"""
ga_solver.py — Genetic Algorithm optimizer for multi-objective appliance selection.

Uses evolutionary optimization to balance power minimisation and zone coverage.
Unlike ILP (exact) or Greedy (heuristic), the GA explores the search space
stochastically and can handle multi-objective trade-offs.

Population-based: each individual is a binary string of appliance on/off states.
Fitness = coverage_fraction - α · power_fraction
"""

import numpy as np
from core.coverage import CoverageResult


def solve_ga(
    coverage: CoverageResult,
    pop_size: int = 50,
    n_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    alpha: float = 0.4,
    verbose: bool = False,
) -> dict:
    """
    Genetic Algorithm for appliance selection.

    Args:
        coverage: CoverageResult with binary matrix and occupied zones.
        pop_size: Population size per generation.
        n_generations: Number of generations to evolve.
        crossover_rate: Probability of crossover between parents.
        mutation_rate: Per-gene mutation probability.
        alpha: Weight for power penalty in fitness (0 = ignore power, 1 = power only).
        verbose: Print generation-by-generation progress.

    Returns:
        dict with same structure as greedy/ilp solvers:
            - 'selected': list of appliance IDs
            - 'selected_indices': list of integer indices
            - 'total_watts': total power consumption
            - 'zones_covered': set of covered zone indices
            - 'generations': number of generations evolved
            - 'best_fitness': final best fitness value
    """
    occupied = coverage.occupied_zones
    if not occupied:
        return {
            "selected": [],
            "selected_indices": [],
            "total_watts": 0.0,
            "zones_covered": set(),
            "generations": 0,
            "best_fitness": 1.0,
        }

    n_appliances = len(coverage.appliance_ids)
    occupied_list = sorted(occupied)
    max_power = sum(coverage.appliance_watts)

    # ── Fitness function ──
    def fitness(individual: np.ndarray) -> float:
        """Higher is better. Balances coverage and power."""
        active_indices = np.where(individual == 1)[0]

        if len(active_indices) == 0:
            return 0.0  # nothing on = no coverage

        # Coverage: fraction of occupied zones covered
        covered = set()
        for ai in active_indices:
            covered |= coverage.covered_zones(ai)
        covered &= set(occupied_list)
        coverage_frac = len(covered) / len(occupied_list)

        # Power: normalised to [0, 1]
        power = sum(coverage.appliance_watts[ai] for ai in active_indices)
        power_frac = power / max_power if max_power > 0 else 0.0

        # Fitness: reward coverage, penalise power
        return coverage_frac - alpha * power_frac

    # ── Initialize population ──
    population = np.random.randint(0, 2, size=(pop_size, n_appliances)).astype(np.int8)

    # Seed one individual with greedy-like heuristic (boost convergence)
    for ai in range(min(n_appliances, len(occupied_list))):
        population[0, ai] = 1

    best_individual = None
    best_fitness_val = -np.inf

    # ── Evolution loop ──
    for gen in range(n_generations):
        # Evaluate fitness
        fitnesses = np.array([fitness(ind) for ind in population])

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness_val:
            best_fitness_val = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            avg_active = population.sum(axis=1).mean()
            print(f"  Gen {gen:3d}  |  Best: {best_fitness_val:.4f}  "
                  f"|  Avg fitness: {fitnesses.mean():.4f}  "
                  f"|  Avg active: {avg_active:.1f}")

        # ── Selection (tournament, size 3) ──
        new_pop = []
        new_pop.append(best_individual.copy())  # elitism

        for _ in range(pop_size - 1):
            # Tournament selection for parent 1
            t1 = np.random.choice(pop_size, size=3, replace=False)
            p1 = population[t1[np.argmax(fitnesses[t1])]]

            # Tournament selection for parent 2
            t2 = np.random.choice(pop_size, size=3, replace=False)
            p2 = population[t2[np.argmax(fitnesses[t2])]]

            # ── Crossover (uniform) ──
            if np.random.random() < crossover_rate:
                mask = np.random.randint(0, 2, size=n_appliances).astype(bool)
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            # ── Mutation ──
            mutations = np.random.random(n_appliances) < mutation_rate
            child[mutations] = 1 - child[mutations]

            new_pop.append(child)

        population = np.array(new_pop, dtype=np.int8)

    # ── Extract best solution ──
    if best_individual is None:
        best_individual = np.zeros(n_appliances, dtype=np.int8)

    selected_indices = list(np.where(best_individual == 1)[0])
    total_watts = sum(coverage.appliance_watts[i] for i in selected_indices)
    selected_ids = [coverage.appliance_ids[i] for i in selected_indices]

    zones_covered = set()
    for j in selected_indices:
        zones_covered |= coverage.covered_zones(j)
    zones_covered &= set(occupied_list)

    return {
        "selected": selected_ids,
        "selected_indices": selected_indices,
        "total_watts": float(total_watts),
        "zones_covered": zones_covered,
        "generations": n_generations,
        "best_fitness": float(best_fitness_val),
    }
