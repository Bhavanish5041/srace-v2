"""
reward.py — Multi-objective reward function for SRACE PPO agent.

Balances eight objectives:
  1. Power minimisation   (α = 0.25)  — use less energy
  2. Comfort maximisation (β = 0.40)  — temperature + CO₂ + lux
  3. Switching penalty     (γ = 0.05)  — don't oscillate appliances
  4. Air quality bonus     (δ = 0.30)  — keep CO₂ low in occupied zones
  5. Danger penalty        (ε = 0.50)  — hard penalty for unsafe conditions
  6. Stability bonus       (fixed 0.3) — reward maintaining good state
  7. Consistency penalty   (ζ = 0.10)  — penalise erratic appliance count swings

Reward = -α·power + β·comfort - γ·switching + δ·air_quality
         - ε·danger + stability - ζ·consistency

The comfort term is internally weighted:
  35% temperature comfort  (affects health)
  40% CO₂ comfort          (ventilation — dominant signal)
  10% coverage fraction    (appliance reach)
  15% lighting comfort     (lux in occupied zones)
"""

import numpy as np


def calculate_reward(
    appliance_states: np.ndarray,
    zone_people: np.ndarray,
    coverage_matrix: np.ndarray,
    prev_appliance_states: np.ndarray,
    appliance_watts: np.ndarray | None = None,
    zone_temps: np.ndarray | None = None,
    zone_co2: np.ndarray | None = None,
    zone_lux: np.ndarray | None = None,
    comfort_targets: dict | None = None,
    alpha: float = 0.25,
    beta: float = 0.40,
    gamma: float = 0.05,
    delta: float = 0.30,
    epsilon: float = 0.50,
    zeta: float = 0.10,
) -> float:
    """
    Compute the RL reward for a given appliance configuration.

    Args:
        appliance_states: Binary array (n_appliances,) — 1 = ON.
        zone_people: Array (n_zones,) — occupant count per zone.
        coverage_matrix: Binary array (n_appliances, n_zones) — coverage.
        prev_appliance_states: Binary array (n_appliances,) — previous step.
        appliance_watts: Array (n_appliances,) — power per appliance.
            Defaults to 75W for fans (first half) and 40W for lights.
        zone_temps: Array (n_zones,) — current zone temperatures °C.
        zone_co2: Array (n_zones,) — current CO₂ levels in ppm.
        zone_lux: Array (n_zones,) — current illuminance in lux.
        comfort_targets: Dict with keys target_temp, target_lux, max_co2.
        alpha: Weight for power penalty.
        beta: Weight for comfort reward.
        gamma: Weight for switching penalty.
        delta: Weight for air quality bonus.
        epsilon: Weight for danger penalty (hard penalty for unsafe conditions).
        zeta: Weight for consistency penalty (appliance count swing).

    Returns:
        Single float reward value.
    """
    n_appliances = len(appliance_states)
    n_zones = len(zone_people)
    stability_bonus = 0.0

    # --- Default appliance wattages (10 fans @ 75W + 10 lights @ 40W) ---
    if appliance_watts is None:
        n_fans = n_appliances // 2
        appliance_watts = np.concatenate([
            np.full(n_fans, 75.0),
            np.full(n_appliances - n_fans, 40.0),
        ])

    # --- Default comfort targets ---
    if comfort_targets is None:
        comfort_targets = {
            "target_temp": 27.0,
            "target_lux": 300.0,
            "max_co2": 1000.0,
        }

    # ═══════════════════════════════════════════════════════════════
    # Term 1: Power penalty  (lower is better)
    # Normalise to [0, 1] range by dividing by max possible power
    # ═══════════════════════════════════════════════════════════════
    total_power = np.dot(appliance_states, appliance_watts)
    max_power = appliance_watts.sum()
    normalised_power = total_power / max_power if max_power > 0 else 0.0

    # Adaptive alpha: double energy penalty for nearly-empty rooms (<30% occupied)
    occupied_mask = zone_people > 0
    n_occupied = occupied_mask.sum()
    occupancy_ratio = n_occupied / n_zones if n_zones > 0 else 0.0
    effective_alpha = alpha * 2.0 if occupancy_ratio < 0.15 and n_occupied > 0 else alpha

    # ═══════════════════════════════════════════════════════════════
    # Term 2: Comfort score  (higher is better)
    # Measures how well occupied zones are served by active appliances
    # ═══════════════════════════════════════════════════════════════


    # Will also track danger penalty
    danger_penalty = 0.0

    if n_occupied == 0:
        # ── EMPTY ROOM: hard early-return ──
        # Running appliances with nobody present is pure waste.
        n_active = appliance_states.sum()
        if n_active == 0:
            # Perfect: everything off, nobody here → strong positive
            return 0.5
        else:
            # Penalise proportionally: -0.5 per active appliance
            return -0.5 * float(n_active)
    else:
        # Coverage check: are occupied zones covered by active appliances?
        active_coverage = coverage_matrix[appliance_states.astype(bool)]
        if active_coverage.size > 0:
            zone_covered = active_coverage.max(axis=0)  # (n_zones,)
        else:
            zone_covered = np.zeros(n_zones)

        # Fraction of occupied zones that are covered
        coverage_fraction = (
            zone_covered[occupied_mask].sum() / n_occupied
        )

        # ── Temperature comfort ──
        # Steeper penalty curve: 1.0 at target, drops fast with deviation.
        # exp(-0.3 * dev²) gives ~0.74 at ±1°C, ~0.30 at ±2°C, ~0.07 at ±3°C
        temp_comfort = 1.0
        if zone_temps is not None:
            target_t = comfort_targets["target_temp"]
            temp_deviations = np.abs(zone_temps[occupied_mask] - target_t)
            mean_dev = temp_deviations.mean()
            temp_comfort = np.exp(-0.3 * mean_dev ** 2)

            # Hard danger penalty: any occupied zone above 35°C
            overheated = (zone_temps[occupied_mask] > 35.0).mean()
            danger_penalty += overheated  # fraction of zones in danger

        # ── CO₂ comfort ──
        # Steeper curve so agent reacts before CO₂ reaches 600+ ppm.
        # exp(-2.5 * norm²): ~0.38 at threshold, drops hard above.
        co2_comfort = 1.0
        if zone_co2 is not None:
            max_co2 = comfort_targets["max_co2"]
            occ_co2 = zone_co2[occupied_mask]
            # Normalise: 0.0 = at ambient, 1.0 = at threshold
            ambient_co2 = 400.0
            co2_norm = np.clip(
                (occ_co2 - ambient_co2) / (max_co2 - ambient_co2), 0.0, 2.0
            )
            co2_comfort = np.exp(-2.5 * co2_norm.mean() ** 2)

            # Hard danger penalty: any occupied zone above 1.2× threshold
            bad_co2 = (occ_co2 > max_co2 * 1.2).mean()
            danger_penalty += bad_co2

        # ── Lighting comfort ──
        lux_comfort = 1.0
        if zone_lux is not None:
            target_lux = comfort_targets["target_lux"]
            occ_lux = zone_lux[occupied_mask]
            # Fraction of occupied zones meeting lux target
            lux_comfort = (occ_lux >= target_lux * 0.7).mean()

        # Weighted comfort: CO₂ is now the dominant signal
        comfort_score = (
            0.35 * temp_comfort
            + 0.40 * co2_comfort
            + 0.10 * coverage_fraction
            + 0.15 * lux_comfort
        )

    # ═══════════════════════════════════════════════════════════════
    # Term 3: Switching penalty  (fewer changes is better)
    # Penalise toggling appliances on/off between steps
    # ═══════════════════════════════════════════════════════════════
    n_switches = np.abs(appliance_states - prev_appliance_states).sum()
    switching_penalty = n_switches / n_appliances  # normalised to [0, 1]

    # ═══════════════════════════════════════════════════════════════
    # Term 3b: Consistency penalty  (stable appliance count is better)
    # Penalise large swings in total active count (e.g. 10→1→8→1)
    # ═══════════════════════════════════════════════════════════════
    prev_n_active = prev_appliance_states.sum()
    curr_n_active = appliance_states.sum()
    count_swing = abs(float(curr_n_active) - float(prev_n_active)) / n_appliances
    consistency_penalty = count_swing ** 2  # quadratic — small diffs OK, big diffs hurt

    # ═══════════════════════════════════════════════════════════════
    # Term 4: Air quality bonus  (lower CO₂ in occupied zones is better)
    # ═══════════════════════════════════════════════════════════════
    air_quality_bonus = 0.0
    if zone_co2 is not None and n_occupied > 0:
        max_co2 = comfort_targets["max_co2"]
        occ_co2 = zone_co2[occupied_mask]
        # Bonus for keeping CO₂ below threshold
        # 1.0 if all zones below threshold, decays as CO₂ rises
        co2_ratio = np.clip(occ_co2 / max_co2, 0.0, 2.0)
        air_quality_bonus = np.exp(-co2_ratio.mean())
    elif n_occupied > 0:
        # No CO₂ data — estimate from fan coverage
        n_fans = n_appliances // 2
        fan_states = appliance_states[:n_fans]
        if fan_states.sum() > 0 and n_occupied > 0:
            # More fans active in occupied zones → better air quality
            fan_coverage = coverage_matrix[:n_fans][fan_states.astype(bool)]
            if fan_coverage.size > 0:
                ventilated = fan_coverage.max(axis=0)[occupied_mask].mean()
                air_quality_bonus = ventilated * 0.5
            else:
                air_quality_bonus = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Term 5: Danger penalty  (hard penalty for unsafe conditions)
    # Normalise to [0, 1]: 0 = safe, 1 = all zones in danger
    # ═══════════════════════════════════════════════════════════════
    danger_penalty = np.clip(danger_penalty / 2.0, 0.0, 1.0)  # avg of temp + co2

    # ═══════════════════════════════════════════════════════════════
    # Term 6: Stability bonus — reward maintaining good state
    # ═══════════════════════════════════════════════════════════════
    total_occupancy = zone_people.sum()
    if zone_temps is not None and zone_co2 is not None:
        avg_temp = zone_temps[occupied_mask].mean() if n_occupied > 0 else zone_temps.mean()
        avg_co2 = zone_co2[occupied_mask].mean() if n_occupied > 0 else zone_co2.mean()
        target_temp = comfort_targets["target_temp"]
        temp_stable = abs(avg_temp - target_temp) < 4.0
        co2_stable = avg_co2 < 800
        if temp_stable and co2_stable and total_occupancy > 0:
            stability_bonus = 0.5

    # ═══════════════════════════════════════════════════════════════
    # Final reward
    # ═══════════════════════════════════════════════════════════════
    occupied_zones = zone_people > 0
    # appliance_states @ coverage_matrix maps (n_appliances) x (n_appliances, n_zones) -> (n_zones)
    coverage_per_zone = appliance_states @ coverage_matrix
    uncovered_occupied = np.sum(occupied_zones & (coverage_per_zone == 0))
    coverage_penalty = -2.0 * uncovered_occupied

    reward = (
        -effective_alpha * normalised_power
        + beta * comfort_score
        - gamma * switching_penalty
        + delta * air_quality_bonus
        - epsilon * danger_penalty
        + stability_bonus
        - zeta * consistency_penalty
        + coverage_penalty
    )

    return float(reward)
