"""
coverage.py — Builds the coverage matrix from physics outputs.

Combines airflow, thermal, CO₂, and lux contributions into a single
binary coverage matrix C[appliance][zone] used by the set-cover optimizers.

Also stores raw continuous values for comfort scoring and RL training.
"""

import numpy as np

from core.room_config import RoomConfig


class CoverageResult:
    """
    Complete coverage analysis for the current room state.

    Attributes:
        binary: (n_appliances, n_zones) — 1 if appliance meaningfully covers zone
        airflow_matrix: (n_fans, n_zones) — raw airflow in m/s
        thermal_impact: (n_fans, n_zones) — ΔT cooling per fan
        co2_reduction: (n_fans, n_zones) — CO₂ ppm reduction per fan
        lux_matrix: (n_lights, n_zones) — lux contribution per light
        appliance_ids: ordered list of appliance IDs (fans first, then lights)
        appliance_watts: ordered list of power consumption
    """

    def __init__(
        self,
        cfg: RoomConfig,
        airflow_matrix: np.ndarray,
        thermal_impact: np.ndarray,
        co2_reduction: np.ndarray,
        lux_matrix: np.ndarray,
        occupied_zones: set[int],
    ):
        self.cfg = cfg
        self.airflow_matrix = airflow_matrix
        self.thermal_impact = thermal_impact
        self.co2_reduction = co2_reduction
        self.lux_matrix = lux_matrix
        self.occupied_zones = occupied_zones

        # Build appliance metadata (fans first, then lights)
        self.appliance_ids = (
            [f.id for f in cfg.fans] + [l.id for l in cfg.lights]
        )
        self.appliance_watts = np.array(
            [f.power_watts for f in cfg.fans]
            + [l.power_watts for l in cfg.lights]
        )

        # Build binary coverage matrix
        self.binary = self._build_binary_coverage(cfg)

    def _build_binary_coverage(self, cfg: RoomConfig) -> np.ndarray:
        """
        Construct binary coverage matrix.

        Fan covers a zone if:
          - airflow >= min_airflow_ms, OR
          - thermal_impact >= 0.5°C (meaningful cooling)

        Light covers a zone if:
          - lux contribution >= target_lux * 0.3 (meaningful contribution)
        """
        n_fans = cfg.n_fans
        n_lights = cfg.n_lights
        n_zones = cfg.n_zones
        n_total = n_fans + n_lights

        binary = np.zeros((n_total, n_zones), dtype=int)

        # Fan coverage
        for fi in range(n_fans):
            for zi in range(n_zones):
                airflow_ok = self.airflow_matrix[fi, zi] >= cfg.comfort.min_airflow_ms
                thermal_ok = self.thermal_impact[fi, zi] >= 0.5  # 0.5°C threshold
                if airflow_ok or thermal_ok:
                    binary[fi, zi] = 1

        # Light coverage — threshold: meaningful lux contribution
        # A single light needn't meet the target alone; multiple lights combine.
        # 20 lux = "this light meaningfully contributes to this zone"
        lux_threshold = 20.0
        for li in range(n_lights):
            for zi in range(n_zones):
                if self.lux_matrix[li, zi] >= lux_threshold:
                    binary[n_fans + li, zi] = 1

        return binary

    def covered_zones(self, appliance_idx: int) -> set[int]:
        """Set of zone indices covered by a given appliance."""
        return set(np.where(self.binary[appliance_idx] == 1)[0])

    def print_summary(self):
        """Print coverage matrix summary."""
        cfg = self.cfg
        print(f"\n{'─'*50}")
        print("  Coverage Matrix Summary")
        print(f"{'─'*50}")

        for ai, aid in enumerate(self.appliance_ids):
            zones_covered = self.covered_zones(ai)
            occ_covered = zones_covered & self.occupied_zones
            print(f"  {aid:>4s}  ({self.appliance_watts[ai]:3.0f}W)  "
                  f"covers {len(zones_covered):2d} zones  "
                  f"({len(occ_covered)} occupied)")

        total_coverable = set()
        for ai in range(len(self.appliance_ids)):
            total_coverable |= self.covered_zones(ai)
        uncoverable = self.occupied_zones - total_coverable
        if uncoverable:
            print(f"\n  ⚠ {len(uncoverable)} occupied zones not coverable by any appliance!")
        print(f"{'─'*50}\n")
