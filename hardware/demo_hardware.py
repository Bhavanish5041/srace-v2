"""
hardware/demo_hardware.py — Standalone SRACE hardware demo.

One-command launcher: connects ESP32, runs optimizer on occupancy,
and drives 4 physical fans based on which zones are occupied.

Usage:
    # Test wiring (cycles each fan on/off):
    python hardware/demo_hardware.py --test-only

    # Run with manual occupancy input:
    python hardware/demo_hardware.py

    # Run with camera pipeline feeding occupancy:
    python hardware/demo_hardware.py --with-camera --camera-url http://192.168.1.5:8080/video

    # Specify serial port explicitly:
    python hardware/demo_hardware.py --port /dev/ttyUSB0
"""

import os
import sys
import time
import argparse
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from hardware.esp32_bridge import ESP32Bridge, NUM_FANS


def load_room_config():
    """Load the hardware demo room config."""
    from core.room_config import load_config
    config_path = os.path.join(PROJECT_ROOT, "config", "hardware_demo.json")
    if not os.path.exists(config_path):
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)
    return load_config(config_path)


def run_optimizer(cfg, airflow_mat, lux_mat, zone_occ):
    """Run greedy optimizer and return fan states [0/1, 0/1, 0/1, 0/1]."""
    from core.coverage import CoverageResult
    from physics.thermal import simulate_thermal
    from physics.co2_model import simulate_co2
    from optimizer.greedy_solver import solve_greedy

    occupied = {zi for zi, c in enumerate(zone_occ) if c > 0}

    if not occupied:
        return [0] * cfg.n_fans

    thermal_impact = simulate_thermal(cfg, airflow_mat, zone_occ)
    co2_reduction = simulate_co2(cfg, airflow_mat, zone_occ)
    coverage = CoverageResult(
        cfg, airflow_mat, thermal_impact, co2_reduction, lux_mat, occupied
    )
    result = solve_greedy(coverage)

    # Extract fan states from optimizer result
    selected = set(result["selected_indices"])
    fan_states = [1 if fi in selected else 0 for fi in range(cfg.n_fans)]
    return fan_states


def print_status(zone_occ, fan_states, cfg):
    """Print a compact status line."""
    n_people = int(sum(zone_occ))
    n_occupied = sum(1 for z in zone_occ if z > 0)
    n_fans_on = sum(fan_states)
    watts = sum(
        f.power_watts for fi, f in enumerate(cfg.fans) if fan_states[fi]
    )
    max_watts = sum(f.power_watts for f in cfg.fans)
    savings = (1 - watts / max_watts) * 100 if max_watts > 0 else 100

    zones_str = " ".join(f"Z{i}:{int(z)}" for i, z in enumerate(zone_occ))
    fans_str = " ".join(
        f"F{i+1}:{'ON' if s else '--'}" for i, s in enumerate(fan_states)
    )

    print(f"  {zones_str}  |  {fans_str}  |  "
          f"{n_people}p {n_occupied}z  {watts}W  "
          f"({savings:.0f}% saved)")


def run_test(bridge):
    """Cycle each fan on/off for wiring verification."""
    print("\n  ═══════════════════════════════════")
    print("  FAN WIRING TEST")
    print("  ═══════════════════════════════════\n")

    for i in range(NUM_FANS):
        states = [0] * NUM_FANS
        states[i] = 1
        print(f"  → Fan {i+1} (F{i+1}) ON   {states}")
        bridge.send_fan_states(states)
        time.sleep(5.0)
        bridge.all_off()
        time.sleep(0.5)

    print(f"\n  → All fans ON")
    bridge.send_fan_states([1, 1, 1, 1])
    time.sleep(3.0)

    bridge.all_off()
    print(f"  → All fans OFF")
    print("\n  ═══════════════════════════════════")
    print("  TEST COMPLETE — verify each fan spun in order F1→F2→F3→F4")
    print("  ═══════════════════════════════════\n")


def run_manual(bridge, cfg, airflow_mat, lux_mat):
    """Interactive mode: type zone occupancy, see fans react."""
    print("\n  ═══════════════════════════════════")
    print("  MANUAL OCCUPANCY MODE")
    print(f"  Room: {cfg.name} ({cfg.n_zones} zones, {cfg.n_fans} fans)")
    print("  ═══════════════════════════════════")
    print(f"\n  Type {cfg.n_zones} occupancy values (e.g. '3,0,2,1')")
    print("  Or: 'full' / 'empty' / 'test' / 'q'\n")

    try:
        while True:
            cmd = input("  occ> ").strip().lower()

            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "empty":
                zone_occ = np.zeros(cfg.n_zones)
            elif cmd == "full":
                zone_occ = np.full(cfg.n_zones, 5.0)
            elif cmd == "test":
                run_test(bridge)
                continue
            else:
                try:
                    vals = [int(x.strip()) for x in cmd.split(",")]
                    if len(vals) != cfg.n_zones:
                        print(f"    ⚠ Need {cfg.n_zones} values, got {len(vals)}")
                        continue
                    zone_occ = np.array(vals, dtype=float)
                except ValueError:
                    print("    ⚠ Format: 3,0,2,1")
                    continue

            fan_states = run_optimizer(cfg, airflow_mat, lux_mat, zone_occ)
            bridge.send_fan_states(fan_states)
            print_status(zone_occ, fan_states, cfg)

    except (KeyboardInterrupt, EOFError):
        pass


def run_with_api(bridge, cfg, interval=3.0):
    """
    Poll the SRACE API for current room state and drive fans accordingly.
    Requires: uvicorn backend.api:app running on port 8000.
    """
    try:
        import requests
    except ImportError:
        print("✗ requests not installed — pip install requests")
        return

    api_url = "http://localhost:8000"
    print(f"\n  Polling {api_url}/room_state every {interval}s...")
    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            try:
                resp = requests.get(f"{api_url}/room_state", timeout=3)
                if resp.status_code != 200:
                    print(f"  ⚠ API returned {resp.status_code}")
                    time.sleep(interval)
                    continue

                data = resp.json()
                appliances = data.get("appliances", [])

                # Extract fan states (first N appliances of type "fan")
                fan_states = []
                for app in appliances:
                    if app["type"] == "fan" and len(fan_states) < NUM_FANS:
                        fan_states.append(1 if app["active"] else 0)

                # Pad if fewer fans in API than hardware
                while len(fan_states) < NUM_FANS:
                    fan_states.append(0)

                bridge.send_fan_states(fan_states[:NUM_FANS])

                fans_str = " ".join(
                    f"F{i+1}:{'ON' if s else '--'}" for i, s in enumerate(fan_states[:NUM_FANS])
                )
                print(f"\r  {fans_str}  |  "
                      f"{data.get('total_people', 0)}p  "
                      f"{data.get('total_power_watts', 0):.0f}W  "
                      f"({data.get('power_saved_pct', 0):.0f}% saved)  ",
                      end="", flush=True)

            except requests.exceptions.ConnectionError:
                print(f"\r  ⚠ API not reachable at {api_url}  ", end="", flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n  Stopping...")


def main():
    parser = argparse.ArgumentParser(
        description="SRACE v2 — Hardware Demo (ESP32 + 4 fans)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=str, default=None,
                        help="ESP32 serial port (auto-detect if omitted)")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run fan wiring test, then exit")
    parser.add_argument("--api", action="store_true",
                        help="Poll FastAPI server for room state")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Polling interval in seconds (default: 3.0)")

    args = parser.parse_args()

    print("\n" + "▓" * 50)
    print("  SRACE v2 — Physical Fan Demo")
    print("  ESP32 + 2× L298N → 4 DC Fans")
    print("▓" * 50)

    # Connect to ESP32
    bridge = ESP32Bridge(port=args.port)
    if not bridge.connect():
        sys.exit(1)

    try:
        if args.test_only:
            run_test(bridge)
        elif args.api:
            run_with_api(bridge, load_room_config(), interval=args.interval)
        else:
            # Load config and pre-compute physics
            cfg = load_room_config()
            from physics.airflow import compute_airflow_matrix
            from physics.lighting import compute_lux_matrix
            airflow_mat = compute_airflow_matrix(cfg)
            lux_mat = compute_lux_matrix(cfg)
            print(f"  ✓ Physics ready: {airflow_mat.shape} airflow, {lux_mat.shape} lux")

            run_manual(bridge, cfg, airflow_mat, lux_mat)
    finally:
        bridge.disconnect()

    print("  Done.\n")


if __name__ == "__main__":
    main()
