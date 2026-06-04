"""
ml/anomaly_detector.py — Streaming anomaly detection using River.

Monitors SRACE sensor data in real-time for:
  - Temperature drift (zones overheating/undercooling)
  - CO₂ spikes (ventilation failures)
  - Occupancy outliers (camera glitches, unusual patterns)
  - Power anomalies (optimizer making wasteful decisions)

Uses River's Half-Space Trees (HST) for online anomaly detection.
No batch retraining needed — adapts continuously to streaming data.

Usage:
    from ml.anomaly_detector import SRACEAnomalyDetector

    detector = SRACEAnomalyDetector()
    for reading in stream:
        score, alerts = detector.update(reading)
        if alerts:
            handle_alerts(alerts)

Requires: river >= 0.21
"""

import time
from typing import Optional
from collections import deque

try:
    from river import anomaly as river_anomaly
    from river import compose, preprocessing
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class SRACEAnomalyDetector:
    """
    Online anomaly detector for SRACE sensor readings.

    Uses Half-Space Trees (O(1) per update, O(1) memory) for streaming
    anomaly scoring, plus rule-based alerts for known danger thresholds.
    """

    def __init__(
        self,
        n_trees: int = 15,
        height: int = 8,
        window_size: int = 250,
        anomaly_threshold: float = 0.80,
        max_history: int = 100,
    ):
        """
        Initialize the anomaly detector.

        Args:
            n_trees: Number of half-space trees.
            height: Tree depth (higher = more sensitive).
            window_size: Sliding window for reference distribution.
            anomaly_threshold: Score threshold (0-1) for flagging anomalies.
            max_history: Max anomaly alerts to keep in memory.
        """
        self.threshold = anomaly_threshold
        self.history = deque(maxlen=max_history)
        self._update_count = 0

        # Track feature statistics for rule-based alerts. This is used even
        # when River is not installed, so initialize it before the optional
        # ML detector setup.
        self._feature_ema = {}  # exponential moving averages
        self._ema_alpha = 0.1   # smoothing factor

        if not RIVER_AVAILABLE:
            self._scaler = None
            self._detector = None
            print("⚠ River not installed — anomaly detection disabled")
            print("  Install with: pip install river>=0.21")
            return

        # Separate scaler and detector for reliable scoring
        self._scaler = preprocessing.StandardScaler()
        self._detector = river_anomaly.HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=42,
        )

    def update(self, reading: dict) -> tuple[float, list[dict]]:
        """
        Process one sensor reading and return anomaly score + alerts.

        Args:
            reading: Dict with keys:
                - avg_temp (float): Average zone temperature °C
                - avg_co2 (float): Average CO₂ ppm
                - avg_lux (float): Average illuminance lux
                - total_power (float): Current power consumption W
                - n_people (int): Total people in room
                - n_active (int): Number of active appliances
                - n_occupied_zones (int): Number of occupied zones

        Returns:
            (anomaly_score, alerts) where:
                anomaly_score: float 0-1 (higher = more anomalous)
                alerts: List of alert dicts with keys: type, severity, message, score
        """
        self._update_count += 1
        alerts = []

        # Extract features for the ML model
        features = {
            "temp": reading.get("avg_temp", 25.0),
            "co2": reading.get("avg_co2", 400.0),
            "lux": reading.get("avg_lux", 0.0),
            "power": reading.get("total_power", 0.0),
            "people": float(reading.get("n_people", 0)),
            "active": float(reading.get("n_active", 0)),
            "occ_zones": float(reading.get("n_occupied_zones", 0)),
        }

        # Derived features
        n_people = features["people"]
        if n_people > 0:
            features["power_per_person"] = features["power"] / n_people
            features["co2_per_person"] = features["co2"] / n_people
        else:
            features["power_per_person"] = features["power"]
            features["co2_per_person"] = 0.0

        # ── ML anomaly scoring ──
        anomaly_score = 0.0
        if self._detector is not None:
            # River estimators mutate in place; some versions return None
            # from learn_one, so keep learning and transforming separate.
            self._scaler.learn_one(features)
            scaled = self._scaler.transform_one(features)
            anomaly_score = float(self._detector.score_one(scaled))
            self._detector.learn_one(scaled)

        # ── Rule-based alerts (always active, even without River) ──

        # 1. Temperature danger
        temp = features["temp"]
        if temp > 35.0:
            alerts.append({
                "type": "temperature_danger",
                "severity": "critical",
                "message": f"🔴 CRITICAL: Avg temperature {temp:.1f}°C exceeds 35°C safety limit",
                "value": temp,
                "threshold": 35.0,
            })
        elif temp > 32.0:
            alerts.append({
                "type": "temperature_warning",
                "severity": "warning",
                "message": f"🟡 WARNING: Avg temperature {temp:.1f}°C approaching danger zone",
                "value": temp,
                "threshold": 32.0,
            })

        # 2. CO₂ danger
        co2 = features["co2"]
        if co2 > 1200.0:
            alerts.append({
                "type": "co2_danger",
                "severity": "critical",
                "message": f"🔴 CRITICAL: CO₂ at {co2:.0f} ppm — ventilation failure",
                "value": co2,
                "threshold": 1200.0,
            })
        elif co2 > 900.0:
            alerts.append({
                "type": "co2_warning",
                "severity": "warning",
                "message": f"🟡 WARNING: CO₂ at {co2:.0f} ppm — approaching limit",
                "value": co2,
                "threshold": 900.0,
            })

        # 3. Power waste detection
        power = features["power"]
        if n_people == 0 and power > 100:
            alerts.append({
                "type": "power_waste",
                "severity": "warning",
                "message": f"🟡 WASTE: {power:.0f}W consumed with no occupants",
                "value": power,
                "threshold": 100.0,
            })

        # 4. Occupancy spike (camera glitch detection)
        if n_people > 80:
            alerts.append({
                "type": "occupancy_spike",
                "severity": "warning",
                "message": f"🟡 ANOMALY: {int(n_people)} people detected — possible sensor error",
                "value": n_people,
                "threshold": 80.0,
            })

        # 5. ML-detected statistical anomaly
        if anomaly_score > self.threshold and self._update_count > 50:
            alerts.append({
                "type": "statistical_anomaly",
                "severity": "info",
                "message": f"🔵 ANOMALY: Unusual sensor pattern detected (score: {anomaly_score:.3f})",
                "value": anomaly_score,
                "threshold": self.threshold,
            })

        # Update EMA tracking
        for key, val in features.items():
            if key in self._feature_ema:
                self._feature_ema[key] = (
                    self._ema_alpha * val + (1 - self._ema_alpha) * self._feature_ema[key]
                )
            else:
                self._feature_ema[key] = val

        # Store alerts in history
        if alerts:
            for alert in alerts:
                alert["timestamp"] = time.time()
                alert["anomaly_score"] = anomaly_score
                alert["reading_index"] = self._update_count
                self.history.append(alert)

        return anomaly_score, alerts

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Get the N most recent anomaly alerts."""
        return list(self.history)[-n:]

    def get_statistics(self) -> dict:
        """Get detector statistics and feature EMAs."""
        return {
            "total_readings": self._update_count,
            "total_alerts": len(self.history),
            "river_available": RIVER_AVAILABLE,
            "threshold": self.threshold,
            "feature_ema": {k: round(v, 2) for k, v in self._feature_ema.items()},
            "recent_alert_count": sum(
                1 for a in self.history
                if time.time() - a.get("timestamp", 0) < 300  # last 5 min
            ),
        }

    @property
    def available(self) -> bool:
        """Whether the ML anomaly detection is available."""
        return RIVER_AVAILABLE and self._detector is not None
