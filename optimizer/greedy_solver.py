"""
greedy_solver.py — Greedy weighted set cover optimizer.

Classic greedy approximation for the Weighted Set Cover problem.
At each step, selects the appliance with the best ratio of
(new occupied zones covered) / (power in watts).

Guaranteed O(ln n) approximation ratio for the optimum.
Runs in O(n_appliances × n_zones) — suitable for real-time (every 30s).
"""

import numpy as np

from core.coverage import CoverageResult


def _greedy_subset(coverage: CoverageResult, appliance_range: range,
                   occupied: set) -> tuple[list[int], set[int]]:
    """Run greedy over a subset of appliances (e.g. just fans or just lights)."""
    selected: list[int] = []
    uncovered = occupied.copy()
    used = set()

    while uncovered:
        best_idx = -1
        best_ratio = -1.0
        best_new = set()

        for ai in appliance_range:
            if ai in used:
                continue
            zones_this = coverage.covered_zones(ai)
            new_zones = zones_this & uncovered
            if not new_zones:
                continue
            ratio = len(new_zones) / coverage.appliance_watts[ai]
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = ai
                best_new = new_zones

        if best_idx == -1:
            break

        selected.append(best_idx)
        used.add(best_idx)
        uncovered -= best_new

    return selected, occupied - uncovered


def solve_greedy(coverage: CoverageResult) -> dict:
    """
    Greedy weighted set cover — fans and lights optimized SEPARATELY.

    Fans cover airflow needs, lights cover illumination needs.
    Running them together causes lights (cheaper per watt) to "steal"
    coverage from fans, leaving occupied zones with zero airflow.

    Args:
        coverage: CoverageResult with binary matrix and occupied zones.

    Returns:
        dict with:
            - 'selected': list of appliance IDs to turn on
            - 'selected_indices': list of integer indices
            - 'total_watts': total power consumption
            - 'zones_covered': set of covered occupied zone indices
    """
    occupied = coverage.occupied_zones.copy()
    if not occupied:
        return {
            "selected": [],
            "selected_indices": [],
            "total_watts": 0.0,
            "zones_covered": set(),
        }

    n_fans = coverage.cfg.n_fans
    n_total = len(coverage.appliance_ids)

    # Phase 1: Greedy over fans only (airflow coverage)
    fan_selected, fan_covered = _greedy_subset(
        coverage, range(0, n_fans), occupied
    )

    # Phase 2: Greedy over lights only (illumination coverage)
    light_selected, light_covered = _greedy_subset(
        coverage, range(n_fans, n_total), occupied
    )

    # Merge results
    selected_indices = fan_selected + light_selected
    total_watts = sum(coverage.appliance_watts[i] for i in selected_indices)
    selected_ids = [coverage.appliance_ids[i] for i in selected_indices]
    zones_covered = fan_covered | light_covered

    return {
        "selected": selected_ids,
        "selected_indices": selected_indices,
        "total_watts": float(total_watts),
        "zones_covered": zones_covered,
    }

