"""
airflow.py — Gaussian decay fan airflow model.

Each fan produces a peak airflow at its center that decays as a Gaussian
with distance.  σ is set so that 95% of airflow falls within the
configured airflow_radius.

Formula:
    v(f, z) = v_peak · exp(−d² / (2σ²))
    σ = airflow_radius / 2

Returns an airflow matrix of shape (n_fans, n_zones) in m/s.
"""

import numpy as np

from core.room_config import RoomConfig


def compute_airflow_matrix(cfg: RoomConfig) -> np.ndarray:
    """
    Compute airflow contribution of each fan to each zone.

    Args:
        cfg: Room configuration.

    Returns:
        np.ndarray of shape (n_fans, n_zones) — airflow in m/s.
    """
    n_fans = cfg.n_fans
    n_zones = cfg.n_zones
    matrix = np.zeros((n_fans, n_zones))

    for fi, fan in enumerate(cfg.fans):
        sigma = fan.airflow_radius / 2.0  # 95% within radius
        for zi, zone in enumerate(cfg.zones):
            dx = fan.x - zone.cx
            dy = fan.y - zone.cy
            dist_sq = dx * dx + dy * dy
            matrix[fi, zi] = fan.airflow_peak_ms * np.exp(
                -dist_sq / (2.0 * sigma * sigma)
            )

    return matrix


def total_airflow_per_zone(
    airflow_matrix: np.ndarray,
    active_fans: np.ndarray,
) -> np.ndarray:
    """
    Sum airflow across all active fans for each zone.

    Args:
        airflow_matrix: (n_fans, n_zones) from compute_airflow_matrix.
        active_fans: Boolean array of length n_fans.

    Returns:
        np.ndarray of shape (n_zones,) — total airflow per zone in m/s.
    """
    return airflow_matrix[active_fans].sum(axis=0)
