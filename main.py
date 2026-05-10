"""
main.py — SRACE v2 pipeline entry point.

Wires together: config → physics → coverage → optimizers → output.
Runs 4 test scenarios to demonstrate the system.
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.room_config import load_config, print_config_summary
from core.coverage import CoverageResult
from physics.airflow import compute_airflow_matrix
from physics.thermal import simulate_thermal
from physics.co2_model import simulate_co2
from physics.lighting import compute_lux_matrix
from optimizer.greedy_solver import solve_greedy
from optimizer.ilp_solver import solve_ilp
from visualization.room_grid import render_room_state


# ── Test Scenarios ──────────────────────────────────────────────

SCENARIOS = {
    "Empty Room": {},
    "Single Person (Zone 0)": {0: 1},
    "Cluster (Center)": {5: 4, 6: 3, 9: 2, 10: 2},
    "Full Room": {i: 4 for i in range(12)},
    "Front Row Only": {0: 5, 1: 5, 2: 5, 3: 5},
}


def build_occupancy(cfg, scenario: dict) -> np.ndarray:
    """Convert scenario dict {zone_idx: count} to array."""
    occ = np.zeros(cfg.n_zones)
    for zi, count in scenario.items():
        if zi < cfg.n_zones:
            occ[zi] = count
    return occ


def run_scenario(cfg, scenario_name, scenario_zones, airflow_mat, lux_mat,
                  save_viz=False, output_dir=None):
    """Run one scenario through the full pipeline."""
    print(f"\n{'━'*60}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'━'*60}")

    zone_occ = build_occupancy(cfg, scenario_zones)
    occupied = {zi for zi, c in enumerate(zone_occ) if c > 0}

    print(f"  Occupied zones: {sorted(occupied) if occupied else 'none'}")
    print(f"  Total people: {int(zone_occ.sum())}")

    if not occupied:
        print(f"  → All appliances OFF — 0 W consumed")
        print(f"  → Savings: 100%")
        return

    # ── Physics engines ──
    t0 = time.perf_counter()
    thermal_impact = simulate_thermal(cfg, airflow_mat, zone_occ)
    t_thermal = time.perf_counter() - t0

    t0 = time.perf_counter()
    co2_reduction = simulate_co2(cfg, airflow_mat, zone_occ)
    t_co2 = time.perf_counter() - t0

    print(f"  Physics: thermal={t_thermal*1000:.1f}ms  co2={t_co2*1000:.1f}ms")

    # ── Coverage matrix ──
    coverage = CoverageResult(
        cfg, airflow_mat, thermal_impact, co2_reduction, lux_mat, occupied
    )

    # ── Optimizers ──
    t0 = time.perf_counter()
    greedy = solve_greedy(coverage)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    ilp = solve_ilp(coverage)
    t_ilp = time.perf_counter() - t0

    max_watts = sum(a.power_watts for a in cfg.all_appliances)

    # ── Results ──
    print(f"\n  {'':>12s} {'Greedy':>12s} {'ILP (Optimal)':>14s}")
    print(f"  {'─'*42}")
    print(f"  {'Appliances':>12s} {len(greedy['selected']):>12d} {len(ilp['selected']):>14d}")
    print(f"  {'Power (W)':>12s} {greedy['total_watts']:>12.0f} {ilp['total_watts']:>14.0f}")
    greedy_sav = (1 - greedy['total_watts'] / max_watts) * 100
    ilp_sav = (1 - ilp['total_watts'] / max_watts) * 100
    print(f"  {'Savings':>12s} {greedy_sav:>11.1f}% {ilp_sav:>13.1f}%")
    print(f"  {'Solve (ms)':>12s} {t_greedy*1000:>12.2f} {t_ilp*1000:>14.2f}")
    print(f"  {'Zones hit':>12s} {len(greedy['zones_covered']):>12d} {len(ilp['zones_covered']):>14d}")

    print(f"\n  Greedy ON : {', '.join(greedy['selected']) or 'none'}")
    print(f"  ILP ON    : {', '.join(ilp['selected']) or 'none'}")

    # Verify ILP ≤ Greedy
    if ilp['total_watts'] <= greedy['total_watts']:
        print(f"  ✓ ILP ≤ Greedy (as expected — provably optimal)")
    else:
        print(f"  ✗ WARNING: ILP > Greedy! Check coverage constraints.")

    # ── Visualization ──
    if save_viz:
        safe_name = scenario_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = output_dir or os.path.join(os.path.dirname(__file__), "output")
        viz_path = os.path.join(out, f"{safe_name}.png")
        render_room_state(
            cfg, coverage, zone_occ, greedy, ilp,
            airflow_mat, lux_mat,
            save_path=viz_path, show=False,
        )


def main():
    """Run SRACE v2 pipeline."""
    from datetime import datetime

    print("\n" + "▓" * 60)
    print("  SRACE v2 — Smart Room Automation & Control Engine")
    print("  Generalised room optimization via physics + set cover")
    print("▓" * 60)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config", "default_room.json")
    cfg = load_config(config_path)
    print_config_summary(cfg)

    # Create timestamped output directory so each run keeps its own files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(os.path.dirname(__file__), "output", f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Pre-compute static physics (airflow + lux don't depend on occupancy)
    print("  Computing static physics models...")
    airflow_mat = compute_airflow_matrix(cfg)
    lux_mat = compute_lux_matrix(cfg)
    print(f"  ✓ Airflow matrix: {airflow_mat.shape}  (peak {airflow_mat.max():.2f} m/s)")
    print(f"  ✓ Lux matrix: {lux_mat.shape}  (peak {lux_mat.max():.1f} lux)")

    # Run all scenarios
    for name, zones in SCENARIOS.items():
        run_scenario(cfg, name, zones, airflow_mat, lux_mat,
                     save_viz=True, output_dir=run_output_dir)

    print(f"\n{'▓'*60}")
    print(f"  All scenarios complete. Visualizations in {run_output_dir}")
    print(f"{'▓'*60}\n")


if __name__ == "__main__":
    main()
