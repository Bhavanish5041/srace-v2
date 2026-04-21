"""
co2_model.py — CO₂ mass-balance ODE model.

Models zone-level CO₂ concentration over time.  People exhale CO₂,
fans increase ventilation which dilutes it.

ODE per zone:
    dC_z/dt = (n_z · G_person) / V_z  −  λ_vent(z) · (C_z − C_ambient)

where λ_vent is increased by fan airflow impinging on the zone.

Returns a CO₂ reduction matrix of shape (n_fans, n_zones) representing
how much each fan reduces CO₂ (in ppm) over the forecast window.
"""

import numpy as np
from scipy.integrate import solve_ivp

from core.room_config import RoomConfig

# Constants
CO2_EXHALE_LS = 0.005  # L/s CO₂ per person
CO2_PPM_PER_LS_M3 = 1e6  # conversion: L/s/m³ → ppm/s
BASE_VENTILATION = 0.0005  # 1/s — natural ventilation rate
FAN_VENTILATION_COEFF = 0.002  # extra ventilation per m/s airflow
FORECAST_SECONDS = 300.0  # 5-minute window


def simulate_co2(
    cfg: RoomConfig,
    airflow_matrix: np.ndarray,
    zone_occupancy: np.ndarray,
    initial_co2: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-fan CO₂ reduction impact via marginal analysis.

    Args:
        cfg: Room configuration.
        airflow_matrix: (n_fans, n_zones) airflow in m/s.
        zone_occupancy: (n_zones,) people per zone.
        initial_co2: (n_zones,) starting CO₂ in ppm. Defaults to ambient.

    Returns:
        np.ndarray of shape (n_fans, n_zones) — CO₂ reduction in ppm
        that each fan contributes. Positive = fan reduces CO₂.
    """
    n_fans = cfg.n_fans
    n_zones = cfg.n_zones

    if initial_co2 is None:
        initial_co2 = np.full(n_zones, cfg.ambient_co2)

    # All fans on
    c_all = _run_co2_ode(cfg, airflow_matrix, zone_occupancy, initial_co2,
                          active_mask=np.ones(n_fans, dtype=bool))

    # Marginal: remove each fan
    impact = np.zeros((n_fans, n_zones))
    for fi in range(n_fans):
        mask = np.ones(n_fans, dtype=bool)
        mask[fi] = False
        c_without = _run_co2_ode(cfg, airflow_matrix, zone_occupancy,
                                  initial_co2, active_mask=mask)
        impact[fi] = c_without - c_all  # positive = this fan helps

    return impact


def _run_co2_ode(
    cfg: RoomConfig,
    airflow_matrix: np.ndarray,
    zone_occupancy: np.ndarray,
    initial_co2: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Solve CO₂ ODE and return final concentrations."""
    n_zones = cfg.n_zones
    ambient = cfg.ambient_co2

    # Total airflow per zone from active fans
    total_airflow = airflow_matrix[active_mask].sum(axis=0)

    # Zone volumes
    volumes = np.array([
        cfg.zones[zi].area * cfg.ceiling_height for zi in range(n_zones)
    ])

    # CO₂ generation rate per zone (ppm/s)
    generation = (zone_occupancy * CO2_EXHALE_LS / volumes) * CO2_PPM_PER_LS_M3

    # Ventilation rate per zone (1/s)
    ventilation = BASE_VENTILATION + FAN_VENTILATION_COEFF * total_airflow

    def deriv(t, co2):
        return generation - ventilation * (co2 - ambient)

    sol = solve_ivp(
        deriv,
        t_span=(0, FORECAST_SECONDS),
        y0=initial_co2,
        method="RK45",
        rtol=1e-4,
        atol=1e-2,
    )

    if not sol.success:
        raise RuntimeError(f"CO₂ ODE solver failed: {sol.message}")

    return sol.y[:, -1]
