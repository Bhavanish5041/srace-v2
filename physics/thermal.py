"""
thermal.py — Thermal model using Method of Lines + SciPy RK45.

Models zone-level temperature evolution over a 5-minute forecast horizon.
Each fan provides convective cooling proportional to its airflow impinging
on the zone.  Occupants contribute ~100 W sensible heat each.

ODE per zone:
    dT_z/dt = −Σ_f α · v(f,z) · (T_z − T_target) + Q_occ / (ρ·c_p·V_z) + k_walls·(T_ambient − T_z)

Returns a thermal impact matrix of shape (n_fans, n_zones) representing
the temperature drop (ΔT) each fan causes over the forecast window.
"""

import numpy as np
from scipy.integrate import solve_ivp

from core.room_config import RoomConfig


# Physical constants
AIR_DENSITY = 1.2  # kg/m³
SPECIFIC_HEAT = 1005.0  # J/(kg·K)
OCCUPANT_HEAT_W = 100.0  # sensible heat per person
CONVECTION_COEFF = 0.15  # tuning: airflow → cooling coefficient
WALL_CONDUCTANCE = 0.01  # heat leak from ambient through walls (1/s)
FORECAST_SECONDS = 300.0  # 5 minute forecast


def _zone_volume(cfg: RoomConfig, zone_idx: int) -> float:
    """Volume of a single zone in m³."""
    zone = cfg.zones[zone_idx]
    return zone.area * cfg.ceiling_height


def simulate_thermal(
    cfg: RoomConfig,
    airflow_matrix: np.ndarray,
    zone_occupancy: np.ndarray,
    initial_temps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run thermal simulation with ALL fans active to measure each fan's
    marginal cooling contribution.

    This simulates temperature evolution with all fans on, then with
    each fan individually removed, to compute per-fan impact.

    Args:
        cfg: Room configuration.
        airflow_matrix: (n_fans, n_zones) airflow contributions.
        zone_occupancy: (n_zones,) number of people per zone.
        initial_temps: (n_zones,) starting temps.  Defaults to ambient.

    Returns:
        np.ndarray of shape (n_fans, n_zones) — ΔT (temperature drop)
        each fan contributes over the forecast window.  Positive = cooling.
    """
    n_fans = cfg.n_fans
    n_zones = cfg.n_zones

    if initial_temps is None:
        initial_temps = np.full(n_zones, cfg.ambient_temp)

    # Compute final temps with ALL fans active
    t_all = _run_ode(cfg, airflow_matrix, zone_occupancy, initial_temps,
                     active_mask=np.ones(n_fans, dtype=bool))

    # Compute marginal contribution: remove one fan at a time
    impact = np.zeros((n_fans, n_zones))
    for fi in range(n_fans):
        mask = np.ones(n_fans, dtype=bool)
        mask[fi] = False
        t_without = _run_ode(cfg, airflow_matrix, zone_occupancy,
                             initial_temps, active_mask=mask)
        # How much warmer zones get without this fan = fan's cooling value
        impact[fi] = t_without - t_all  # positive means this fan helps

    return impact


def _run_ode(
    cfg: RoomConfig,
    airflow_matrix: np.ndarray,
    zone_occupancy: np.ndarray,
    initial_temps: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    Solve the thermal ODE system and return final temperatures.

    Args:
        active_mask: Boolean array — which fans are on.

    Returns:
        Final temperature array (n_zones,).
    """
    n_zones = cfg.n_zones
    target_t = cfg.comfort.target_temp_c
    ambient_t = cfg.ambient_temp

    # Pre-compute total airflow per zone from active fans
    total_airflow = airflow_matrix[active_mask].sum(axis=0)  # (n_zones,)

    # Pre-compute zone volumes
    volumes = np.array([_zone_volume(cfg, zi) for zi in range(n_zones)])
    thermal_mass = AIR_DENSITY * SPECIFIC_HEAT * volumes  # J/K per zone

    # Occupant heat input per zone (W)
    q_occ = zone_occupancy * OCCUPANT_HEAT_W

    def deriv(t, temps):
        dTdt = np.zeros(n_zones)
        for zi in range(n_zones):
            # Convective cooling from fans
            cooling = CONVECTION_COEFF * total_airflow[zi] * (temps[zi] - target_t)
            # Heat from occupants
            heating = q_occ[zi] / thermal_mass[zi]
            # Heat leak from walls
            wall_leak = WALL_CONDUCTANCE * (ambient_t - temps[zi])
            dTdt[zi] = -cooling + heating + wall_leak
        return dTdt

    sol = solve_ivp(
        deriv,
        t_span=(0, FORECAST_SECONDS),
        y0=initial_temps,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
    )

    if not sol.success:
        raise RuntimeError(f"Thermal ODE solver failed: {sol.message}")

    return sol.y[:, -1]  # final temperatures
