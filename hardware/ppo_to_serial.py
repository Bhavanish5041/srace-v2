"""
hardware/ppo_to_serial.py — PPO model → ESP32 serial fan controller.

Loads a trained Stable-Baselines3 PPO model, builds observations using
the same SRACEEnv gym environment, runs model.predict() to get fan
actions, and sends them to the ESP32 over serial.

The trained model uses the default_room config (69-dim obs, 21 appliances).
This script:
  1. Instantiates the full SRACEEnv (matching the model's obs/action space)
  2. Injects real occupancy from either CLI input or the SRACE API
  3. Calls model.predict(obs) → gets 21-appliance binary actions
  4. Extracts the first N fan actions → sends to ESP32

Usage:
    # Basic: manual occupancy, auto-detect ESP32
    python hardware/ppo_to_serial.py

    # With specific port and interval
    python hardware/ppo_to_serial.py --port /dev/ttyUSB0 --interval 3

    # With occupancy from the SRACE API
    python hardware/ppo_to_serial.py --api http://localhost:8000

    # Train a hardware-specific model first, then use it
    python hardware/ppo_to_serial.py --model models/hardware_ppo.zip \\
                                     --config config/hardware_demo.json
"""

import os
import sys
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hardware.esp32_bridge import ESP32Bridge, NUM_FANS


def load_ppo_model(model_path: str):
    """Load a trained PPO model from a .zip file."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("✗ stable-baselines3 not installed")
        print("  pip install stable-baselines3")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)

    model = PPO.load(model_path)
    print(f"✓ PPO model loaded: {os.path.basename(model_path)}")
    print(f"  Action space : {model.action_space}")
    print(f"  Obs space    : {model.observation_space}")
    return model


def create_env(config_path: str):
    """Create a SRACEEnv matching the model's training config."""
    from ml.gym_env import SRACEEnv
    env = SRACEEnv(config_path=config_path)
    print(f"✓ Environment: {env.cfg.name}")
    print(f"  Zones: {env.n_zones}, Fans: {env.n_fans}, "
          f"Lights: {env.n_lights}, Appliances: {env.n_appliances}")
    return env


def set_env_occupancy(env, zone_people: np.ndarray):
    """
    Inject real occupancy into the environment state and rebuild obs.

    If the env has more zones than the occupancy array (e.g. env has 12
    zones from default_room but we only have 4 hardware zones), we map
    the hardware zones into the first N env zones.
    """
    n_hw = len(zone_people)
    n_env = env.n_zones

    if n_hw == n_env:
        env.zone_people = zone_people.astype(np.float32)
    elif n_hw < n_env:
        # Map hardware zones into env zones (fill rest with 0)
        env.zone_people = np.zeros(n_env, dtype=np.float32)
        env.zone_people[:n_hw] = zone_people
    else:
        # More hardware zones than env — truncate
        env.zone_people = zone_people[:n_env].astype(np.float32)


def extract_fan_states(action: np.ndarray, env, n_hw_fans: int = NUM_FANS) -> list[int]:
    """
    Extract binary fan states from the PPO action vector.

    PPO with MultiBinary action space outputs binary 0/1 directly
    (not continuous). We just take the first n_hw_fans fan actions.
    """
    # Fans are always the first env.n_fans entries in the action vector
    fan_actions = action[:env.n_fans]

    # Take first n_hw_fans for hardware
    hw_states = []
    for i in range(n_hw_fans):
        if i < len(fan_actions):
            hw_states.append(int(fan_actions[i]))
        else:
            hw_states.append(0)

    return hw_states


def get_api_occupancy(api_url: str, n_zones: int) -> np.ndarray | None:
    """Fetch current occupancy from the SRACE API."""
    try:
        import requests
        resp = requests.get(f"{api_url}/room_state", timeout=3)
        if resp.status_code != 200:
            return None
        data = resp.json()
        zones = data.get("zones", [])
        occ = np.zeros(n_zones, dtype=np.float32)
        for i, z in enumerate(zones[:n_zones]):
            occ[i] = z.get("people", 0)
        return occ
    except Exception:
        return None


def print_cycle(cycle: int, zone_occ: np.ndarray, obs: np.ndarray,
                raw_action: np.ndarray, fan_states: list[int],
                esp_response: bool, env):
    """Print a formatted status line for one control cycle."""
    n_people = int(zone_occ.sum())
    n_occupied = int((zone_occ > 0).sum())
    n_fans_on = sum(fan_states)
    watts = sum(
        f.power_watts for fi, f in enumerate(env.cfg.fans[:NUM_FANS])
        if fi < len(fan_states) and fan_states[fi]
    )
    max_watts = sum(f.power_watts for f in env.cfg.fans[:NUM_FANS])
    savings = (1 - watts / max_watts) * 100 if max_watts > 0 else 100

    # Compact display
    occ_str = " ".join(f"Z{i}:{int(z)}" for i, z in enumerate(zone_occ[:NUM_FANS]))
    fan_str = " ".join(f"F{i+1}:{'ON' if s else '--'}" for i, s in enumerate(fan_states))
    raw_str = ",".join(f"{a}" for a in raw_action[:env.n_fans])

    print(f"\n  ┌─ Cycle {cycle} {'─' * 40}")
    print(f"  │ Occupancy : {occ_str}  ({n_people}p, {n_occupied}z)")
    print(f"  │ PPO raw   : [{raw_str}]")
    print(f"  │ Fan cmd   : {fan_states}  →  {fan_str}")
    print(f"  │ Power     : {watts}W / {max_watts}W  ({savings:.0f}% saved)")
    print(f"  │ ESP32     : {'✓ OK' if esp_response else '✗ FAIL'}")
    print(f"  └{'─' * 50}")


def run_loop(bridge, model, env, interval: float,
             api_url: str | None, manual_occ: np.ndarray | None):
    """Main PPO → serial control loop."""
    print(f"\n{'▓' * 50}")
    print(f"  PPO → ESP32 Control Loop")
    print(f"  Interval: {interval}s  |  Ctrl+C to stop")
    print(f"{'▓' * 50}\n")

    # Reset env to get initial state
    obs, _ = env.reset()

    cycle = 0
    try:
        while True:
            cycle += 1

            # ── Get occupancy ──
            if api_url:
                zone_occ = get_api_occupancy(api_url, NUM_FANS)
                if zone_occ is None:
                    print(f"  ⚠ API unreachable, using last occupancy")
                    zone_occ = env.zone_people[:NUM_FANS]
            elif manual_occ is not None:
                zone_occ = manual_occ
            else:
                # Random occupancy for testing
                zone_occ = np.random.randint(0, 6, size=NUM_FANS).astype(np.float32)

            # ── Inject occupancy into env ──
            set_env_occupancy(env, zone_occ)

            # ── Build observation ──
            obs = env._get_observation()

            # ── PPO predict ──
            action, _states = model.predict(obs, deterministic=True)

            # ── Extract hardware fan states ──
            fan_states = extract_fan_states(action, env)

            # ── Send to ESP32 ──
            ok = bridge.send_fan_states(fan_states)

            # ── Print status ──
            print_cycle(cycle, zone_occ, obs, action, fan_states, ok, env)

            # ── Run one physics tick to update env state ──
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
                set_env_occupancy(env, zone_occ)

            # ── Wait ──
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n  Stopping after {cycle} cycles...")


def main():
    parser = argparse.ArgumentParser(
        description="SRACE PPO → ESP32 Serial Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "srace_ppo.zip"),
                        help="Path to trained PPO model .zip")
    parser.add_argument("--config", type=str, default=None,
                        help="Room config JSON (default: matches trained model)")
    parser.add_argument("--port", type=str, default=None,
                        help="ESP32 serial port (auto-detect if omitted)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Seconds between control cycles (default: 3)")
    parser.add_argument("--api", type=str, default=None,
                        help="SRACE API URL for live occupancy (e.g. http://localhost:8000)")
    parser.add_argument("--occupancy", type=str, default=None,
                        help="Fixed occupancy e.g. '3,0,2,1' (default: random)")

    args = parser.parse_args()

    print(f"\n{'▓' * 50}")
    print(f"  SRACE v2 — PPO → Physical Fan Control")
    print(f"{'▓' * 50}\n")

    # ── Load model ──
    model = load_ppo_model(args.model)

    # ── Create matching environment ──
    # If no config specified, use the default (matching the trained model)
    if args.config is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "default_room.json")
    else:
        config_path = args.config
    env = create_env(config_path)

    # ── Validate compatibility ──
    model_obs_shape = model.observation_space.shape
    env_obs_shape = env.observation_space.shape
    if model_obs_shape != env_obs_shape:
        print(f"\n  ⚠ Observation space mismatch!")
        print(f"    Model expects : {model_obs_shape}")
        print(f"    Env produces  : {env_obs_shape}")
        print(f"  → Using default_room config to match the trained model.")
        print(f"    Hardware fans (first {NUM_FANS}) will be extracted from "
              f"the {model.action_space.n}-action output.\n")
        config_path = os.path.join(PROJECT_ROOT, "config", "default_room.json")
        env = create_env(config_path)

    # ── Parse fixed occupancy if given ──
    manual_occ = None
    if args.occupancy:
        manual_occ = np.array(
            [int(x.strip()) for x in args.occupancy.split(",")],
            dtype=np.float32
        )
        print(f"  Fixed occupancy: {manual_occ.tolist()}")

    # ── Connect ESP32 ──
    bridge = ESP32Bridge(port=args.port)
    if not bridge.connect():
        sys.exit(1)

    try:
        run_loop(bridge, model, env, args.interval, args.api, manual_occ)
    finally:
        bridge.disconnect()

    print("  Done.\n")


if __name__ == "__main__":
    main()
