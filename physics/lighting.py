"""
lighting.py — Cosine-law illuminance model for ceiling fixtures.

Each ceiling light emits downward in a Lambertian (cosine) pattern.
Illuminance at a point on the floor:

    E(l, z) = (Φ · cos³θ) / (2π · h²)

where:
    Φ = luminous flux in lumens
    h = mounting height
    θ = angle from nadir: tan(θ) = d/h
    cos θ = h / √(d² + h²)

This is physically correct for ceiling-mounted fixtures radiating
into a hemisphere (not a full sphere like a bare bulb).

Returns a lux matrix of shape (n_lights, n_zones).
"""

import numpy as np

from core.room_config import RoomConfig


def compute_lux_matrix(cfg: RoomConfig) -> np.ndarray:
    """
    Compute illuminance contribution of each light to each zone.

    Uses cosine-law for downward-emitting ceiling fixtures.
    Formula: E = (Φ · cos³θ) / (2π · h²)

    This gives realistic values: ~140 lux directly under an 800-lumen
    fixture at 3m height, decaying with angle.

    Args:
        cfg: Room configuration.

    Returns:
        np.ndarray of shape (n_lights, n_zones) — lux values.
    """
    n_lights = cfg.n_lights
    n_zones = cfg.n_zones
    matrix = np.zeros((n_lights, n_zones))

    for li, light in enumerate(cfg.lights):
        h = light.height_above_floor
        h_sq = h * h
        for zi, zone in enumerate(cfg.zones):
            dx = light.x - zone.cx
            dy = light.y - zone.cy
            horiz_dist_sq = dx * dx + dy * dy
            total_dist_sq = horiz_dist_sq + h_sq
            # cos θ = h / r  where r = √(d² + h²)
            cos_theta = h / np.sqrt(total_dist_sq)
            # Cosine-law: E = (Φ · cos³θ) / (2π · h²)
            matrix[li, zi] = (light.lumens * cos_theta**3) / (2.0 * np.pi * h_sq)

    return matrix


def total_lux_per_zone(
    lux_matrix: np.ndarray,
    active_lights: np.ndarray,
    ambient_lux: float = 0.0,
) -> np.ndarray:
    """
    Total illuminance per zone from active lights + ambient.

    Args:
        lux_matrix: (n_lights, n_zones) from compute_lux_matrix.
        active_lights: Boolean array of length n_lights.
        ambient_lux: Background light level.

    Returns:
        np.ndarray of shape (n_zones,) — total lux per zone.
    """
    return lux_matrix[active_lights].sum(axis=0) + ambient_lux
