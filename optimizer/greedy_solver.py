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


def solve_greedy(coverage: CoverageResult) -> dict:
    """
    Greedy weighted set cover.

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

    n_appliances = len(coverage.appliance_ids)
    selected_indices: list[int] = []
    uncovered = occupied.copy()
    used = set()

    while uncovered:
        best_idx = -1
        best_ratio = -1.0
        best_new_zones = set()

        for ai in range(n_appliances):
            if ai in used:
                continue

            zones_this = coverage.covered_zones(ai)
            new_zones = zones_this & uncovered
            if not new_zones:
                continue

            # Efficiency ratio: zones covered per watt
            ratio = len(new_zones) / coverage.appliance_watts[ai]
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = ai
                best_new_zones = new_zones

        if best_idx == -1:
            # No appliance can cover remaining zones
            break

        selected_indices.append(best_idx)
        used.add(best_idx)
        uncovered -= best_new_zones

    total_watts = sum(coverage.appliance_watts[i] for i in selected_indices)
    selected_ids = [coverage.appliance_ids[i] for i in selected_indices]

    return {
        "selected": selected_ids,
        "selected_indices": selected_indices,
        "total_watts": float(total_watts),
        "zones_covered": occupied - uncovered,
    }
