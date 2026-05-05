"""
reward.py — Multi-objective reward function for SRACE PPO agent.

Balances four objectives:
  1. Power minimisation  (α = 0.3)
  2. Comfort maximisation (β = 0.5)
  3. Switching penalty     (γ = 0.1)
  4. Air quality bonus     (δ = 0.1)

Reward = -α · total_power + β · comfort_score
         - γ · switching_penalty + δ · air_quality_bonus
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
    alpha: float = 0.3,
    beta: float = 0.5,
    gamma: float = 0.1,
    delta: float = 0.1,
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

    Returns:
        Single float reward value.
    """
    n_appliances = len(appliance_states)
    n_zones = len(zone_people)

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
            "target_temp": 25.0,
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

    # ═══════════════════════════════════════════════════════════════
    # Term 2: Comfort score  (higher is better)
    # Measures how well occupied zones are served by active appliances
    # ═══════════════════════════════════════════════════════════════
    occupied_mask = zone_people > 0
    n_occupied = occupied_mask.sum()

    if n_occupied == 0:
        # No one in the room — perfect comfort if everything is off
        comfort_score = 1.0 if total_power == 0 else 0.0
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

        # Temperature comfort (if available)
        temp_comfort = 1.0
        if zone_temps is not None:
            target_t = comfort_targets["target_temp"]
            # Penalty grows with deviation from target
            temp_deviations = np.abs(zone_temps[occupied_mask] - target_t)
            # Sigmoid-like comfort: 1.0 at target, ~0.5 at ±3°C
            temp_comfort = np.exp(-0.1 * temp_deviations.mean() ** 2)

        # Lighting comfort (if available)
        lux_comfort = 1.0
        if zone_lux is not None:
            target_lux = comfort_targets["target_lux"]
            occ_lux = zone_lux[occupied_mask]
            # Fraction of occupied zones meeting lux target
            lux_comfort = (occ_lux >= target_lux * 0.7).mean()

        comfort_score = (
            0.5 * coverage_fraction
            + 0.3 * temp_comfort
            + 0.2 * lux_comfort
        )

    # ═══════════════════════════════════════════════════════════════
    # Term 3: Switching penalty  (fewer changes is better)
    # Penalise toggling appliances on/off between steps
    # ═══════════════════════════════════════════════════════════════
    n_switches = np.abs(appliance_states - prev_appliance_states).sum()
    switching_penalty = n_switches / n_appliances  # normalised to [0, 1]

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
    # Final reward
    # ═══════════════════════════════════════════════════════════════
    reward = (
        -alpha * normalised_power
        + beta * comfort_score
        - gamma * switching_penalty
        + delta * air_quality_bonus
    )

    return float(reward)
