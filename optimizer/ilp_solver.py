"""
ilp_solver.py — Exact ILP optimizer via PuLP CBC.

Formulates the Weighted Set Cover as an Integer Linear Program:

    min  Σ_j  w_j · x_j          (minimize total wattage)
    s.t. Σ_{j: z∈S_j} x_j ≥ 1   for all occupied zones z
         x_j ∈ {0, 1}

Provably finds the optimal (minimum power) solution.
Uses PuLP's built-in CBC (COIN-OR Branch and Cut) solver.
"""

import pulp
import numpy as np

from core.coverage import CoverageResult


def solve_ilp(coverage: CoverageResult) -> dict:
    """
    Exact ILP solution for minimum-power appliance subset.

    Args:
        coverage: CoverageResult with binary matrix and occupied zones.

    Returns:
        dict with:
            - 'selected': list of appliance IDs to turn on
            - 'selected_indices': list of integer indices
            - 'total_watts': total power consumption
            - 'zones_covered': set of covered occupied zone indices
            - 'solver_status': PuLP status string
    """
    occupied = coverage.occupied_zones
    if not occupied:
        return {
            "selected": [],
            "selected_indices": [],
            "total_watts": 0.0,
            "zones_covered": set(),
            "solver_status": "Optimal (trivial)",
        }

    n_appliances = len(coverage.appliance_ids)
    occupied_list = sorted(occupied)

    # --- Create ILP problem ---
    prob = pulp.LpProblem("SRACE_SetCover", pulp.LpMinimize)

    # Binary decision variables: x_j = 1 if appliance j is selected
    x = [
        pulp.LpVariable(f"x_{coverage.appliance_ids[j]}", cat=pulp.LpBinary)
        for j in range(n_appliances)
    ]

    # Objective: minimize total power
    prob += pulp.lpSum(
        coverage.appliance_watts[j] * x[j] for j in range(n_appliances)
    ), "TotalPower"

    # Constraints: every occupied zone must be covered by ≥1 appliance
    for zi in occupied_list:
        covering = [
            j for j in range(n_appliances)
            if coverage.binary[j, zi] == 1
        ]
        if covering:
            prob += (
                pulp.lpSum(x[j] for j in covering) >= 1,
                f"Cover_zone_{zi}",
            )
        # If no appliance covers this zone, constraint is infeasible
        # but we allow it — solver will still find best partial solution

    # Solve (suppress output)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]

    # Extract solution
    selected_indices = [
        j for j in range(n_appliances)
        if x[j].varValue is not None and x[j].varValue > 0.5
    ]

    total_watts = sum(coverage.appliance_watts[j] for j in selected_indices)
    selected_ids = [coverage.appliance_ids[j] for j in selected_indices]

    # Verify which zones are actually covered
    zones_covered = set()
    for j in selected_indices:
        zones_covered |= coverage.covered_zones(j)
    zones_covered &= occupied  # only count occupied zones

    return {
        "selected": selected_ids,
        "selected_indices": selected_indices,
        "total_watts": float(total_watts),
        "zones_covered": zones_covered,
        "solver_status": status,
    }
