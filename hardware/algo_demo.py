"""
hardware/algo_demo.py — Compare Greedy / ILP / GA / PPO on physical fans + LEDs.

Runs all four SRACE optimizers on the same shared observation each cycle,
prints a comparison table (fans AND lights), sends ONE chosen algorithm's
decision to the ESP32, and logs everything to CSV.

Usage:
    # Single algorithm mode:
    python hardware/algo_demo.py --algo greedy --port /dev/ttyUSB0
    python hardware/algo_demo.py --algo ilp
    python hardware/algo_demo.py --algo ga
    python hardware/algo_demo.py --algo ppo

    # Compare all 4, send greedy's decision to the fans + LEDs:
    python hardware/algo_demo.py --algo compare --send greedy

    # Compare all 4, send PPO's decision, fixed occupancy:
    python hardware/algo_demo.py --algo compare --send ppo --occupancy 3,0,2,0

    # Custom interval, API occupancy source:
    python hardware/algo_demo.py --algo compare --send ilp --interval 5 \\
                                 --api http://localhost:8000
"""

import os
import sys
import csv
import time
import argparse
import datetime
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hardware.esp32_bridge import ESP32Bridge, NUM_FANS, NUM_LEDS

# ══════════════════════════════════════════════════════════════
#  ALGORITHM WRAPPERS — decide(state) -> (fan_states, led_states)
# ══════════════════════════════════════════════════════════════

class AlgoWrapper:
    """Base class for all algorithm wrappers."""
    name: str = "base"

    def decide(self, state: dict) -> tuple[list[int], list[int]]:
        """Given shared state dict, return (fans [0/1]*4, leds [0/1]*4)."""
        raise NotImplementedError

    def _solver_to_appliances(self, result: dict,
                              n_fans: int, n_lights: int
                              ) -> tuple[list[int], list[int]]:
        """Convert solver result dict to (fan_list, light_list)."""
        selected = set(result.get("selected_indices", []))
        fans = [1 if fi in selected else 0 for fi in range(n_fans)]
        lights = [1 if (n_fans + li) in selected else 0
                  for li in range(n_lights)]
        return fans, lights


class GreedyWrapper(AlgoWrapper):
    name = "Greedy"

    def decide(self, state: dict) -> tuple[list[int], list[int]]:
        from optimizer.greedy_solver import solve_greedy
        result = solve_greedy(state["coverage"])
        return self._solver_to_appliances(
            result, state["n_fans"], state["n_lights"]
        )


class ILPWrapper(AlgoWrapper):
    name = "ILP"

    def decide(self, state: dict) -> tuple[list[int], list[int]]:
        from optimizer.ilp_solver import solve_ilp
        result = solve_ilp(state["coverage"])
        return self._solver_to_appliances(
            result, state["n_fans"], state["n_lights"]
        )


class GAWrapper(AlgoWrapper):
    name = "GA"

    def __init__(self, pop_size: int = 30, generations: int = 50):
        self.pop_size = pop_size
        self.generations = generations

    def decide(self, state: dict) -> tuple[list[int], list[int]]:
        from optimizer.ga_solver import solve_ga
        result = solve_ga(
            state["coverage"],
            pop_size=self.pop_size,
            n_generations=self.generations,
        )
        return self._solver_to_appliances(
            result, state["n_fans"], state["n_lights"]
        )


class PPOWrapper(AlgoWrapper):
    name = "PPO"

    def __init__(self, model_path: str, config_path: str | None = None):
        from stable_baselines3 import PPO
        from ml.gym_env import SRACEEnv

        self.model = PPO.load(model_path)

        # Create env matching the model's obs/action space
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "config", "default_room.json")
        self.env = SRACEEnv(config_path=config_path)

        # Verify compatibility
        if self.model.observation_space.shape != self.env.observation_space.shape:
            print(f"  ⚠ PPO obs mismatch — rebuilding env with default_room")
            self.env = SRACEEnv()

        self.env.reset()

    def decide(self, state: dict) -> tuple[list[int], list[int]]:
        """Inject occupancy into env, build obs, run PPO predict."""
        zone_occ = state["zone_occupancy"]

        # Map hardware zones into the (possibly larger) env
        self.env.zone_people = np.zeros(self.env.n_zones, dtype=np.float32)
        n = min(len(zone_occ), self.env.n_zones)
        self.env.zone_people[:n] = zone_occ[:n]

        obs = self.env._get_observation()
        action, _ = self.model.predict(obs, deterministic=True)

        # Extract fan actions (first n_fans in action vector)
        fan_actions = action[: self.env.n_fans]
        hw_fans = []
        for i in range(NUM_FANS):
            if i < len(fan_actions):
                hw_fans.append(int(fan_actions[i]))
            else:
                hw_fans.append(0)

        # Extract light actions (next n_lights after fans)
        light_start = self.env.n_fans
        light_end = light_start + self.env.n_lights
        light_actions = action[light_start:light_end]
        hw_leds = []
        for i in range(NUM_LEDS):
            if i < len(light_actions):
                hw_leds.append(int(light_actions[i]))
            else:
                hw_leds.append(0)

        return hw_fans, hw_leds


# ══════════════════════════════════════════════════════════════
#  SHARED STATE BUILDER
# ══════════════════════════════════════════════════════════════

def build_shared_state(cfg, airflow_mat, lux_mat, zone_occ):
    """
    Build the shared state dict that all algorithms consume.
    Uses the same physics as the API and gym env.
    """
    from core.coverage import CoverageResult
    from physics.thermal import simulate_thermal
    from physics.co2_model import simulate_co2

    occupied = {zi for zi, c in enumerate(zone_occ) if c > 0}

    thermal_impact = simulate_thermal(cfg, airflow_mat, zone_occ)
    co2_reduction = simulate_co2(cfg, airflow_mat, zone_occ)
    coverage = CoverageResult(
        cfg, airflow_mat, thermal_impact, co2_reduction, lux_mat, occupied
    )

    return {
        "cfg": cfg,
        "coverage": coverage,
        "zone_occupancy": zone_occ,
        "occupied_zones": occupied,
        "n_fans": cfg.n_fans,
        "n_lights": cfg.n_lights,
        "n_zones": cfg.n_zones,
    }


# ══════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════

def compute_power(fan_states: list[int], led_states: list[int], cfg) -> float:
    """Compute total watts for fans + LEDs."""
    fan_watts = sum(
        cfg.fans[fi].power_watts
        for fi in range(min(len(fan_states), cfg.n_fans))
        if fan_states[fi]
    )
    led_watts = sum(
        cfg.lights[li].power_watts
        for li in range(min(len(led_states), cfg.n_lights))
        if led_states[li]
    )
    return fan_watts + led_watts


def print_header():
    print(f"\n  {'Algorithm':<10} │ {'F1':>3} {'F2':>3} {'F3':>3} {'F4':>3} "
          f"│ {'L1':>3} {'L2':>3} {'L3':>3} {'L4':>3} "
          f"│ {'Power (W)':>10} │ {'Time (ms)':>10}")
    print(f"  {'─' * 10}─┼─{'─' * 15}─┼─{'─' * 15}─┼─{'─' * 10}─┼─{'─' * 10}")


def print_row(name: str, fans: list[int], leds: list[int],
              watts: float, ms: float, is_sent: bool = False):
    fan_str = " ".join(f"{'█' if s else '·':>3}" for s in fans)
    led_str = " ".join(f"{'█' if s else '·':>3}" for s in leds)
    marker = " ◀ SENT" if is_sent else ""
    print(f"  {name:<10} │ {fan_str} │ {led_str} "
          f"│ {watts:>10.1f} │ {ms:>10.2f}{marker}")


def print_cycle_header(cycle: int, zone_occ: np.ndarray):
    n_people = int(zone_occ.sum())
    occ_str = " ".join(f"Z{i}:{int(z)}" for i, z in enumerate(zone_occ))
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n  ┌{'─' * 62}┐")
    print(f"  │  Cycle {cycle:>4}  │  {ts}  │  {occ_str}  ({n_people}p)")
    print(f"  └{'─' * 62}┘")


# ══════════════════════════════════════════════════════════════
#  CSV LOGGER
# ══════════════════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp", "cycle",
            "z0_occ", "z1_occ", "z2_occ", "z3_occ",
            "algorithm",
            "f1", "f2", "f3", "f4",
            "l1", "l2", "l3", "l4",
            "power_w", "time_ms", "sent_to_hw",
        ])

    def log(self, cycle: int, zone_occ: np.ndarray, algo_name: str,
            fans: list[int], leds: list[int],
            watts: float, ms: float, sent: bool):
        occ_vals = [int(zone_occ[i]) if i < len(zone_occ) else 0
                    for i in range(4)]
        self.writer.writerow([
            datetime.datetime.now().isoformat(), cycle,
            *occ_vals,
            algo_name,
            *fans, *leds,
            round(watts, 1), round(ms, 2), int(sent),
        ])
        self.file.flush()

    def close(self):
        self.file.close()


# ══════════════════════════════════════════════════════════════
#  OCCUPANCY SOURCES
# ══════════════════════════════════════════════════════════════

def get_occupancy(mode: str, fixed_occ: np.ndarray | None,
                  api_url: str | None, n_zones: int,
                  cycle: int) -> np.ndarray:
    """Get zone occupancy from the selected source."""
    if fixed_occ is not None:
        return fixed_occ

    if api_url:
        try:
            import requests
            resp = requests.get(f"{api_url}/room_state", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                zones = data.get("zones", [])
                occ = np.zeros(n_zones, dtype=np.float32)
                for i, z in enumerate(zones[:n_zones]):
                    occ[i] = z.get("occupancy", 0)
                return occ
        except Exception:
            pass

    # Random occupancy — varies each cycle for a dynamic demo
    rng = np.random.RandomState(seed=int(time.time()) + cycle)
    n_occupied = rng.randint(0, n_zones + 1)
    occ = np.zeros(n_zones, dtype=np.float32)
    zones = rng.choice(n_zones, size=n_occupied, replace=False)
    for z in zones:
        occ[z] = rng.randint(1, 7)
    return occ


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run(bridge: ESP32Bridge, algos: dict[str, AlgoWrapper],
        send_algo: str, cfg, airflow_mat, lux_mat,
        interval: float, fixed_occ, api_url, csv_logger):
    """Main control loop."""

    max_fan_watts = sum(f.power_watts for f in cfg.fans[:NUM_FANS])
    max_led_watts = sum(l.power_watts for l in cfg.lights[:NUM_LEDS])
    max_watts = max_fan_watts + max_led_watts
    cycle = 0

    try:
        while True:
            cycle += 1

            # ── Get occupancy ──
            zone_occ = get_occupancy(
                "fixed" if fixed_occ is not None else ("api" if api_url else "random"),
                fixed_occ, api_url, cfg.n_zones, cycle
            )

            # ── Build shared state ──
            state = build_shared_state(cfg, airflow_mat, lux_mat, zone_occ)

            # ── Print cycle header ──
            print_cycle_header(cycle, zone_occ)
            print_header()

            # ── Run each algorithm ──
            results = {}
            for name, algo in algos.items():
                t0 = time.perf_counter()
                fan_states, led_states = algo.decide(state)
                dt_ms = (time.perf_counter() - t0) * 1000

                watts = compute_power(fan_states, led_states, cfg)
                is_sent = (name == send_algo)

                results[name] = {
                    "fans": fan_states,
                    "leds": led_states,
                    "watts": watts,
                    "ms": dt_ms,
                    "sent": is_sent,
                }

                print_row(name, fan_states, led_states, watts, dt_ms, is_sent)

                if csv_logger:
                    csv_logger.log(cycle, zone_occ, name, fan_states,
                                   led_states, watts, dt_ms, is_sent)

            # ── Send chosen algo's decision to ESP32 ──
            chosen = results.get(send_algo)
            if chosen:
                bridge.send_fan_states(chosen["fans"][:NUM_FANS])
                bridge.send_led_states(chosen["leds"][:NUM_LEDS])

            # ── Summary line ──
            sent_watts = chosen["watts"] if chosen else 0
            savings = (1 - sent_watts / max_watts) * 100 if max_watts > 0 else 100
            f_str = ",".join(str(s) for s in chosen["fans"]) if chosen else "?"
            l_str = ",".join(str(s) for s in chosen["leds"]) if chosen else "?"
            print(f"\n  ⚡ Sent: {send_algo}")
            print(f"    Fans: [{f_str}]  LEDs: [{l_str}]")
            print(f"    {sent_watts:.0f}W / {max_watts}W  ({savings:.0f}% saved)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {cycle} cycles.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SRACE v2 — Algorithm Comparison on Physical Fans + LEDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hardware/algo_demo.py --algo greedy
  python hardware/algo_demo.py --algo compare --send ppo
  python hardware/algo_demo.py --algo compare --send greedy --occupancy 3,0,2,0
  python hardware/algo_demo.py --algo ppo --interval 5 --api http://localhost:8000
        """,
    )
    parser.add_argument(
        "--algo", type=str, default="compare",
        choices=["greedy", "ilp", "ga", "ppo", "compare"],
        help="Algorithm to run (default: compare)",
    )
    parser.add_argument(
        "--send", type=str, default=None,
        choices=["greedy", "ilp", "ga", "ppo"],
        help="Which algo's decision to send to ESP32 in compare mode "
             "(default: first in list)",
    )
    parser.add_argument("--port", type=str, default=None,
                        help="ESP32 serial port (auto-detect if omitted)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between cycles (default: 5)")
    parser.add_argument("--occupancy", type=str, default=None,
                        help="Fixed occupancy, e.g. '3,0,2,0' (default: random)")
    parser.add_argument("--api", type=str, default=None,
                        help="SRACE API URL for live occupancy")
    parser.add_argument("--model", type=str, default=None,
                        help="PPO model .zip path (default: models/srace_ppo.zip)")
    parser.add_argument("--config", type=str, default=None,
                        help="Room config JSON (default: config/hardware_demo.json)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV log file path (default: algo_demo_log.csv)")
    parser.add_argument("--ga-pop", type=int, default=30,
                        help="GA population size (default: 30)")
    parser.add_argument("--ga-gen", type=int, default=50,
                        help="GA generations (default: 50)")

    args = parser.parse_args()

    print(f"\n{'▓' * 62}")
    print(f"  SRACE v2 — Algorithm Comparison Demo")
    print(f"  Greedy │ ILP (PuLP) │ GA │ PPO → Physical Fans + LEDs")
    print(f"{'▓' * 62}\n")

    # ── Determine which algos to run ──
    algo_map = {
        "greedy": "Greedy",
        "ilp": "ILP",
        "ga": "GA",
        "ppo": "PPO",
    }

    if args.algo == "compare":
        algo_keys = ["greedy", "ilp", "ga", "ppo"]
    else:
        algo_keys = [args.algo]

    # Default --send to first algo
    send_key = args.send or algo_keys[0]
    if send_key not in algo_keys:
        print(f"  ⚠ --send {send_key} not in active algos, using {algo_keys[0]}")
        send_key = algo_keys[0]
    send_name = algo_map[send_key]

    # ── Load room config ──
    config_path = args.config or os.path.join(
        PROJECT_ROOT, "config", "hardware_demo.json"
    )
    from core.room_config import load_config
    from physics.airflow import compute_airflow_matrix
    from physics.lighting import compute_lux_matrix

    cfg = load_config(config_path)
    airflow_mat = compute_airflow_matrix(cfg)
    lux_mat = compute_lux_matrix(cfg)
    print(f"  ✓ Config: {cfg.name} "
          f"({cfg.n_zones}z, {cfg.n_fans}f, {cfg.n_lights}l)")

    # ── Build algorithm wrappers ──
    algos: dict[str, AlgoWrapper] = {}
    for key in algo_keys:
        name = algo_map[key]
        if key == "greedy":
            algos[name] = GreedyWrapper()
        elif key == "ilp":
            algos[name] = ILPWrapper()
        elif key == "ga":
            algos[name] = GAWrapper(
                pop_size=args.ga_pop, generations=args.ga_gen
            )
        elif key == "ppo":
            model_path = args.model or os.path.join(
                PROJECT_ROOT, "models", "srace_ppo.zip"
            )
            algos[name] = PPOWrapper(model_path)
        print(f"  ✓ {name} ready")

    # ── Parse fixed occupancy ──
    fixed_occ = None
    if args.occupancy:
        fixed_occ = np.array(
            [int(x.strip()) for x in args.occupancy.split(",")],
            dtype=np.float32,
        )
        print(f"  ✓ Fixed occupancy: {fixed_occ.tolist()}")

    # ── CSV logger ──
    csv_path = args.csv or os.path.join(PROJECT_ROOT, "algo_demo_log.csv")
    csv_logger = CSVLogger(csv_path)
    print(f"  ✓ CSV log: {csv_path}")

    # ── Connect ESP32 ──
    bridge = ESP32Bridge(port=args.port)
    if not bridge.connect():
        sys.exit(1)

    print(f"\n  Sending: {send_name}'s decisions to ESP32")
    print(f"  Interval: {args.interval}s | Ctrl+C to stop\n")

    try:
        run(bridge, algos, send_name, cfg, airflow_mat, lux_mat,
            args.interval, fixed_occ, args.api, csv_logger)
    finally:
        csv_logger.close()
        bridge.disconnect()
        print(f"  ✓ CSV saved: {csv_path}")

    print("  Done.\n")


if __name__ == "__main__":
    main()
