"""
backend/api.py — SRACE v2 FastAPI server.

Exposes room state, physics, and optimizer results as REST endpoints.
Unity, the React dashboard, or any HTTP client can consume these.

Endpoints:
    GET  /room_state     — Run physics + greedy optimizer, return full state
    POST /set_occupancy  — Update zone occupancy counts
    GET  /ppo_action     — Run trained PPO agent, return appliance decisions
    GET  /compare        — Compare greedy vs PPO side-by-side
    GET  /config         — Raw room configuration
    GET  /docs           — Auto-generated Swagger UI (courtesy of FastAPI)

Run:
    uvicorn backend.api:app --reload --port 8000
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.room_config import load_config, RoomConfig
from core.coverage import CoverageResult
from physics.airflow import compute_airflow_matrix
from physics.thermal import simulate_thermal
from physics.co2_model import simulate_co2
from physics.lighting import compute_lux_matrix
from optimizer.greedy_solver import solve_greedy
from optimizer.ilp_solver import solve_ilp

# PPO (loaded lazily — works even if model file is missing)
try:
    from stable_baselines3 import PPO as PPOModel
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("⚠ stable-baselines3 not installed — PPO endpoints disabled")

# MQTT bridge (loaded lazily — works even if paho-mqtt is missing)
try:
    from backend.mqtt_bridge import MQTTBridge, MQTT_AVAILABLE as MQTT_LIB_AVAILABLE
except ImportError:
    MQTT_LIB_AVAILABLE = False
    MQTTBridge = None

# River anomaly detection (loaded lazily)
try:
    from ml.anomaly_detector import SRACEAnomalyDetector
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False
    SRACEAnomalyDetector = None

# ══════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="SRACE v2 API",
    description=(
        "Smart Room Automation & Control Engine — "
        "physics-based room optimization over REST."
    ),
    version="2.0.0",
)

# Allow Unity WebGL, React dashboard, or any local dev tool to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  STARTUP — Load config + pre-compute static physics
# ══════════════════════════════════════════════════════════════

CONFIG_PATH = PROJECT_ROOT / "config" / "classroom_real.json"

cfg: RoomConfig = None
airflow_mat: np.ndarray = None
lux_mat: np.ndarray = None
ppo_model = None  # Loaded at startup if model file exists
mqtt_bridge = None  # MQTT bridge instance
anomaly_detector = None  # River anomaly detector instance

# Current occupancy state (mutable — updated via POST /set_occupancy)
zone_occupancy: np.ndarray = None

# PPO state tracking (persists between API calls)
ppo_appliance_states: np.ndarray = None
ppo_zone_temps: np.ndarray = None
ppo_zone_co2: np.ndarray = None
ppo_zone_lux: np.ndarray = None


@app.on_event("startup")
def startup():
    """Load room config and pre-compute static physics matrices."""
    global cfg, airflow_mat, lux_mat, zone_occupancy

    if not CONFIG_PATH.exists():
        # Fall back to default_room.json if classroom_real not found
        fallback = PROJECT_ROOT / "config" / "default_room.json"
        if not fallback.exists():
            raise RuntimeError(f"No room config found at {CONFIG_PATH} or {fallback}")
        cfg = load_config(fallback)
        print(f"⚠ classroom_real.json not found — loaded {fallback.name}")
    else:
        cfg = load_config(CONFIG_PATH)
        print(f"✓ Loaded room config: {cfg.name}")

    # Static physics — airflow and lux don't depend on occupancy
    airflow_mat = compute_airflow_matrix(cfg)
    lux_mat = compute_lux_matrix(cfg)
    print(f"✓ Airflow matrix: {airflow_mat.shape}  (peak {airflow_mat.max():.2f} m/s)")
    print(f"✓ Lux matrix:     {lux_mat.shape}  (peak {lux_mat.max():.1f} lux)")

    # Default: empty room
    zone_occupancy = np.zeros(cfg.n_zones)

    # Load PPO model if available
    global ppo_model, ppo_appliance_states, ppo_zone_temps, ppo_zone_co2, ppo_zone_lux
    ppo_appliance_states = np.zeros(cfg.n_appliances, dtype=np.int8)
    ppo_zone_temps = np.full(cfg.n_zones, cfg.ambient_temp)
    ppo_zone_co2 = np.full(cfg.n_zones, cfg.ambient_co2)
    ppo_zone_lux = np.full(cfg.n_zones, cfg.ambient_lux)

    if PPO_AVAILABLE:
        model_path = PROJECT_ROOT / "models" / "srace_ppo.zip"
        if model_path.exists():
            ppo_model = PPOModel.load(str(model_path))
            print(f"✓ PPO model loaded: {model_path.name}")
        else:
            print(f"⚠ No PPO model at {model_path} — /ppo_action will return 404")

    # Initialize MQTT bridge
    global mqtt_bridge
    if MQTT_LIB_AVAILABLE and MQTTBridge is not None:
        def on_mqtt_occupancy(zone_people):
            """Handle occupancy updates from MQTT (e.g. camera pipeline)."""
            global zone_occupancy
            if len(zone_people) == cfg.n_zones:
                zone_occupancy = np.array(zone_people, dtype=float)
                print(f"📡 MQTT: Occupancy updated — {int(zone_occupancy.sum())} people")

        mqtt_bridge = MQTTBridge(
            on_occupancy=on_mqtt_occupancy,
        )
        mqtt_bridge.start()
    else:
        print("⚠ MQTT bridge not available")

    # Initialize anomaly detector
    global anomaly_detector
    if ANOMALY_AVAILABLE and SRACEAnomalyDetector is not None:
        anomaly_detector = SRACEAnomalyDetector()
        print("✓ River anomaly detector initialized")
    else:
        print("⚠ Anomaly detection not available (install river>=0.21)")

    print(f"✓ SRACE API ready — {cfg.n_zones} zones, {cfg.n_appliances} appliances\n")


# ══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class OccupancyRequest(BaseModel):
    """POST body for /set_occupancy."""
    zone_people: list[int]

    class Config:
        json_schema_extra = {
            "example": {
                "zone_people": [0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0]
            }
        }


class ZoneInfo(BaseModel):
    index: int
    row: int
    col: int
    cx: float
    cy: float
    occupancy: int
    covered: bool
    airflow_ms: float
    lux: float


class ApplianceInfo(BaseModel):
    id: str
    type: str  # "fan", "light", or "projector"
    x: float
    y: float
    power_watts: float
    active: bool


class RoomStateResponse(BaseModel):
    room_name: str
    dimensions: dict
    zones: list[ZoneInfo]
    appliances: list[ApplianceInfo]
    total_power_watts: float
    max_power_watts: float
    power_saved_pct: float
    solver: str
    solve_time_ms: float
    occupied_zone_count: int
    total_people: int
    avg_temp: float = 0.0
    avg_co2: float = 0.0
    avg_lux: float = 0.0


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/room_state", response_model=RoomStateResponse,
         summary="Get current room state with physics + optimization")
def get_room_state():
    """
    Run the full SRACE pipeline on the current occupancy:
    1. Compute dynamic physics (thermal, CO₂)
    2. Build coverage matrix
    3. Run greedy optimizer
    4. Return structured JSON with zones, appliance states, and power saved
    """
    global zone_occupancy

    occupied = {zi for zi, c in enumerate(zone_occupancy) if c > 0}

    # ── Dynamic physics ──
    thermal_impact = simulate_thermal(cfg, airflow_mat, zone_occupancy)
    co2_reduction = simulate_co2(cfg, airflow_mat, zone_occupancy)

    # ── Coverage matrix ──
    coverage = CoverageResult(
        cfg, airflow_mat, thermal_impact, co2_reduction, lux_mat, occupied
    )

    # ── Optimizer ──
    t0 = time.perf_counter()
    result = solve_greedy(coverage)
    solve_ms = (time.perf_counter() - t0) * 1000

    selected_set = set(result["selected_indices"])
    max_watts = sum(a.power_watts for a in cfg.all_appliances)
    saved_pct = (1 - result["total_watts"] / max_watts) * 100 if max_watts > 0 else 100.0

    # ── Build zone info ──
    # For each zone, compute the total airflow and lux from active appliances
    zones_out = []
    for zi in range(cfg.n_zones):
        z = cfg.zones[zi]

        # Sum airflow from active fans
        total_airflow = 0.0
        for fi in range(cfg.n_fans):
            if fi in selected_set:
                total_airflow += float(airflow_mat[fi, zi])

        # Sum lux from active lights
        total_lux = float(cfg.ambient_lux)
        for li in range(cfg.n_lights):
            ai = cfg.n_fans + li  # appliance index
            if ai in selected_set:
                total_lux += float(lux_mat[li, zi])

        is_covered = zi in result["zones_covered"]

        zones_out.append(ZoneInfo(
            index=zi,
            row=z.row,
            col=z.col,
            cx=round(z.cx, 2),
            cy=round(z.cy, 2),
            occupancy=int(zone_occupancy[zi]),
            covered=is_covered,
            airflow_ms=round(total_airflow, 3),
            lux=round(total_lux, 1),
        ))

    # ── Build appliance info ──
    appliances_out = []
    for fi, fan in enumerate(cfg.fans):
        appliances_out.append(ApplianceInfo(
            id=fan.id,
            type="fan",
            x=round(fan.x, 2),
            y=round(fan.y, 2),
            power_watts=fan.power_watts,
            active=fi in selected_set,
        ))
    for li, light in enumerate(cfg.lights):
        ai = cfg.n_fans + li
        appliances_out.append(ApplianceInfo(
            id=light.id,
            type="light",
            x=round(light.x, 2),
            y=round(light.y, 2),
            power_watts=light.power_watts,
            active=ai in selected_set,
        ))
    for pi, proj in enumerate(cfg.projectors):
        ai = cfg.n_fans + cfg.n_lights + pi
        appliances_out.append(ApplianceInfo(
            id=proj.id,
            type="projector",
            x=round(proj.x, 2),
            y=round(proj.y, 2),
            power_watts=proj.power_watts,
            active=ai in selected_set,
        ))

    # Compute environment metrics for dashboard
    # Run simplified thermal/CO₂ estimate based on current occupancy
    avg_temp = float(cfg.ambient_temp)
    avg_co2 = float(cfg.ambient_co2)
    avg_lux = float(cfg.ambient_lux)
    total_people = int(zone_occupancy.sum())
    if total_people > 0:
        # Rough estimate: each person adds heat, fans cool
        active_fans = np.array([fi in selected_set for fi in range(cfg.n_fans)])
        if active_fans.any():
            from physics.airflow import total_airflow_per_zone
            af = total_airflow_per_zone(airflow_mat, active_fans)
            avg_temp = float(cfg.ambient_temp + 0.05 * total_people - 0.02 * af.sum())
        else:
            avg_temp = float(cfg.ambient_temp + 0.05 * total_people)
        avg_co2 = float(cfg.ambient_co2 + 5.0 * total_people)

    return RoomStateResponse(
        room_name=cfg.name,
        dimensions={
            "width": cfg.width,
            "depth": cfg.depth,
            "ceiling_height": cfg.ceiling_height,
        },
        zones=zones_out,
        appliances=appliances_out,
        total_power_watts=result["total_watts"],
        max_power_watts=max_watts,
        power_saved_pct=round(saved_pct, 1),
        solver="greedy",
        solve_time_ms=round(solve_ms, 2),
        occupied_zone_count=len(occupied),
        total_people=total_people,
        avg_temp=round(avg_temp, 1),
        avg_co2=round(avg_co2, 0),
        avg_lux=round(avg_lux, 1),
    )


@app.post("/set_occupancy", summary="Update zone occupancy counts")
def set_occupancy(req: OccupancyRequest):
    """
    Set the number of people in each zone.
    Expects a list of integers with length == number of zones.

    Unity or the React dashboard can POST here to simulate different scenarios.
    """
    global zone_occupancy

    if len(req.zone_people) != cfg.n_zones:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {cfg.n_zones} zone values, got {len(req.zone_people)}. "
                   f"Room has a {cfg.n_zone_rows}×{cfg.n_zone_cols} zone grid.",
        )

    if any(p < 0 for p in req.zone_people):
        raise HTTPException(
            status_code=422,
            detail="Occupancy values must be non-negative.",
        )

    zone_occupancy = np.array(req.zone_people, dtype=float)
    total = int(zone_occupancy.sum())
    occupied_count = int((zone_occupancy > 0).sum())

    return {
        "status": "ok",
        "total_people": total,
        "occupied_zones": occupied_count,
        "total_zones": cfg.n_zones,
        "message": f"Occupancy updated — {total} people across {occupied_count}/{cfg.n_zones} zones.",
    }


@app.get("/config", summary="Get raw room configuration")
def get_config():
    """Return the loaded room config metadata (no physics, just layout)."""
    return {
        "name": cfg.name,
        "width": cfg.width,
        "depth": cfg.depth,
        "ceiling_height": cfg.ceiling_height,
        "ambient_temp": cfg.ambient_temp,
        "ambient_co2": cfg.ambient_co2,
        "ambient_lux": cfg.ambient_lux,
        "zone_grid": f"{cfg.n_zone_rows}×{cfg.n_zone_cols}",
        "n_zones": cfg.n_zones,
        "n_fans": cfg.n_fans,
        "n_lights": cfg.n_lights,
        "max_power_watts": sum(a.power_watts for a in cfg.all_appliances),
        "comfort": {
            "target_temp_c": cfg.comfort.target_temp_c,
            "target_lux": cfg.comfort.target_lux,
            "max_co2_ppm": cfg.comfort.max_co2_ppm,
            "min_airflow_ms": cfg.comfort.min_airflow_ms,
        },
    }


# ══════════════════════════════════════════════════════════════
#  PPO ENDPOINTS
# ══════════════════════════════════════════════════════════════

# Observation normalisation constants (must match ml/gym_env.py)
_MAX_PEOPLE = 8
_MAX_TEMP = 45.0
_MAX_CO2 = 2000.0
_MAX_LUX = 1500.0

# Simplified physics constants (must match ml/gym_env.py)
_THERMAL_DECAY = 0.02
_CO2_DECAY_FAN = 0.01
_CO2_GENERATION = 5.0
_CO2_NATURAL_DECAY = 0.005
_OCCUPANT_HEAT = 0.05


def _build_ppo_observation():
    """Build a normalised observation vector matching the PPO training format."""
    obs = np.concatenate([
        zone_occupancy / _MAX_PEOPLE,
        ppo_zone_temps / _MAX_TEMP,
        ppo_zone_co2 / _MAX_CO2,
        ppo_zone_lux / _MAX_LUX,
        ppo_appliance_states.astype(np.float32),
    ])
    return obs.astype(np.float32)


def _ppo_physics_tick():
    """Run one step of simplified physics (matches ml/gym_env.py)."""
    global ppo_zone_temps, ppo_zone_co2, ppo_zone_lux

    from physics.airflow import total_airflow_per_zone
    from physics.lighting import total_lux_per_zone

    fan_states = ppo_appliance_states[:cfg.n_fans].astype(bool)
    light_end = cfg.n_fans + cfg.n_lights
    light_states = ppo_appliance_states[cfg.n_fans:light_end].astype(bool)
    proj_states = ppo_appliance_states[light_end:].astype(bool) if cfg.n_projectors > 0 else np.array([], dtype=bool)

    # Airflow → Temperature
    if fan_states.any():
        airflow = total_airflow_per_zone(airflow_mat, fan_states)
    else:
        airflow = np.zeros(cfg.n_zones)

    target_t = cfg.comfort.target_temp_c
    cooling = _THERMAL_DECAY * airflow * (ppo_zone_temps - target_t)
    ppo_zone_temps = ppo_zone_temps - cooling
    ppo_zone_temps = ppo_zone_temps + _OCCUPANT_HEAT * zone_occupancy
    ppo_zone_temps = ppo_zone_temps + 0.005 * (cfg.ambient_temp - ppo_zone_temps)
    ppo_zone_temps = np.clip(ppo_zone_temps, 15.0, _MAX_TEMP)

    # CO₂
    ppo_zone_co2 = ppo_zone_co2 + _CO2_GENERATION * zone_occupancy
    co2_removal = _CO2_DECAY_FAN * airflow * (ppo_zone_co2 - cfg.ambient_co2)
    ppo_zone_co2 = ppo_zone_co2 - co2_removal
    ppo_zone_co2 = ppo_zone_co2 - _CO2_NATURAL_DECAY * (ppo_zone_co2 - cfg.ambient_co2)
    ppo_zone_co2 = np.clip(ppo_zone_co2, cfg.ambient_co2, _MAX_CO2)

    # Lighting (lights + projectors)
    if light_states.any():
        ppo_zone_lux = total_lux_per_zone(lux_mat, light_states, cfg.ambient_lux)
    else:
        ppo_zone_lux = np.full(cfg.n_zones, cfg.ambient_lux)

    # Add projector lux contribution
    if cfg.n_projectors > 0 and proj_states.any():
        for pi, proj in enumerate(cfg.projectors):
            if not proj_states[pi]:
                continue
            for zi in range(cfg.n_zones):
                z = cfg.zones[zi]
                dx = proj.x - z.cx
                dy = proj.y - z.cy
                dist = np.sqrt(dx*dx + dy*dy)
                if dist <= proj.coverage_radius:
                    falloff = 1.0 - (dist / proj.coverage_radius)
                    ppo_zone_lux[zi] += proj.screen_lux * falloff


@app.get("/ppo_action", summary="Run PPO agent to get appliance decisions")
def ppo_action():
    """
    Use the trained PPO reinforcement learning agent to decide which
    appliances to activate based on the current room state.

    Unlike the greedy optimizer (static snapshot), the PPO agent maintains
    internal state and makes temporally-aware decisions — it knows that
    turning fans on NOW prevents overheating later.

    Returns the same structure as /room_state but using PPO decisions.
    """
    global ppo_appliance_states

    if ppo_model is None:
        raise HTTPException(
            status_code=404,
            detail="PPO model not loaded. Train with: python3 ml/train_ppo.py --timesteps 500000"
        )

    # Build observation from current state
    obs = _build_ppo_observation()

    # Get PPO action (deterministic)
    action, _ = ppo_model.predict(obs, deterministic=True)
    ppo_appliance_states = np.array(action, dtype=np.int8)

    # Run one physics tick to update state
    _ppo_physics_tick()

    # Build response
    total_power = float(np.dot(ppo_appliance_states, np.array(
        [f.power_watts for f in cfg.fans]
        + [l.power_watts for l in cfg.lights]
        + [p.power_watts for p in cfg.projectors]
    )))
    max_watts = sum(a.power_watts for a in cfg.all_appliances)
    saved_pct = (1 - total_power / max_watts) * 100 if max_watts > 0 else 100.0

    appliances_out = []
    for fi, fan in enumerate(cfg.fans):
        appliances_out.append({
            "id": fan.id,
            "type": "fan",
            "x": round(fan.x, 2),
            "y": round(fan.y, 2),
            "power_watts": fan.power_watts,
            "active": bool(ppo_appliance_states[fi]),
        })
    for li, light in enumerate(cfg.lights):
        appliances_out.append({
            "id": light.id,
            "type": "light",
            "x": round(light.x, 2),
            "y": round(light.y, 2),
            "power_watts": light.power_watts,
            "active": bool(ppo_appliance_states[cfg.n_fans + li]),
        })
    for pi, proj in enumerate(cfg.projectors):
        appliances_out.append({
            "id": proj.id,
            "type": "projector",
            "x": round(proj.x, 2),
            "y": round(proj.y, 2),
            "power_watts": proj.power_watts,
            "active": bool(ppo_appliance_states[cfg.n_fans + cfg.n_lights + pi]),
        })

    return {
        "solver": "ppo",
        "room_name": cfg.name,
        "appliances": appliances_out,
        "total_power_watts": total_power,
        "max_power_watts": max_watts,
        "power_saved_pct": round(saved_pct, 1),
        "avg_temp": round(float(ppo_zone_temps.mean()), 1),
        "avg_co2": round(float(ppo_zone_co2.mean()), 0),
        "avg_lux": round(float(ppo_zone_lux.mean()), 1),
        "n_active": int(ppo_appliance_states.sum()),
        "n_fans_on": int(ppo_appliance_states[:cfg.n_fans].sum()),
        "n_lights_on": int(ppo_appliance_states[cfg.n_fans:cfg.n_fans + cfg.n_lights].sum()),
        "n_projectors_on": int(ppo_appliance_states[cfg.n_fans + cfg.n_lights:].sum()),
        "total_people": int(zone_occupancy.sum()),
        "occupied_zones": int((zone_occupancy > 0).sum()),
    }


@app.post("/ppo_reset", summary="Reset PPO agent state")
def ppo_reset():
    """Reset the PPO agent's internal physics state (temps, CO₂, lux)."""
    global ppo_appliance_states, ppo_zone_temps, ppo_zone_co2, ppo_zone_lux
    ppo_appliance_states = np.zeros(cfg.n_appliances, dtype=np.int8)
    ppo_zone_temps = np.full(cfg.n_zones, cfg.ambient_temp)
    ppo_zone_co2 = np.full(cfg.n_zones, cfg.ambient_co2)
    ppo_zone_lux = np.full(cfg.n_zones, cfg.ambient_lux)
    return {"status": "ok", "message": "PPO state reset to ambient conditions."}


@app.get("/compare", summary="Compare greedy vs PPO side-by-side")
def compare_solvers():
    """
    Run both the greedy optimizer and PPO agent on the current occupancy,
    and return their results side-by-side for comparison.
    """
    # Get greedy result
    greedy_result = get_room_state()

    # Get PPO result (or placeholder if unavailable)
    if ppo_model is not None:
        ppo_result = ppo_action()
    else:
        ppo_result = {"error": "PPO model not loaded"}

    return {
        "occupancy": {
            "total_people": int(zone_occupancy.sum()),
            "occupied_zones": int((zone_occupancy > 0).sum()),
            "total_zones": cfg.n_zones,
        },
        "greedy": {
            "power_watts": greedy_result.total_power_watts,
            "power_saved_pct": greedy_result.power_saved_pct,
            "n_active": len([a for a in greedy_result.appliances if a.active]),
            "fans_on": [a.id for a in greedy_result.appliances if a.active and a.type == "fan"],
            "lights_on": [a.id for a in greedy_result.appliances if a.active and a.type == "light"],
            "projectors_on": [a.id for a in greedy_result.appliances if a.active and a.type == "projector"],
        },
        "ppo": {
            "power_watts": ppo_result.get("total_power_watts", 0),
            "power_saved_pct": ppo_result.get("power_saved_pct", 0),
            "n_active": ppo_result.get("n_active", 0),
            "avg_temp": ppo_result.get("avg_temp", 0),
            "avg_co2": ppo_result.get("avg_co2", 0),
            "fans_on": ppo_result.get("n_fans_on", 0),
            "lights_on": ppo_result.get("n_lights_on", 0),
            "projectors_on": ppo_result.get("n_projectors_on", 0),
        } if "error" not in ppo_result else ppo_result,
    }


# ══════════════════════════════════════════════════════════════
#  MQTT ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/mqtt_status", summary="Get MQTT connection status")
def get_mqtt_status():
    """Return the current MQTT bridge connection status."""
    if mqtt_bridge is not None:
        return mqtt_bridge.status
    return {
        "connected": False,
        "available": False,
        "message": "MQTT bridge not initialized (install paho-mqtt>=2.0)",
    }


# ══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION ENDPOINTS
# ══════════════════════════════════════════════════════════════

def _feed_anomaly_detector(state: dict):
    """Feed a room state snapshot to the anomaly detector."""
    if anomaly_detector is None:
        return 0.0, []

    reading = {
        "avg_temp": state.get("avg_temp", 25.0),
        "avg_co2": state.get("avg_co2", 400.0),
        "avg_lux": state.get("avg_lux", 0.0),
        "total_power": state.get("total_power_watts", 0.0),
        "n_people": state.get("total_people", 0),
        "n_active": state.get("n_active", 0),
        "n_occupied_zones": state.get("occupied_zones", 0),
    }
    return anomaly_detector.update(reading)


@app.get("/anomalies", summary="Get recent anomaly alerts")
def get_anomalies(n: int = 20):
    """Return the N most recent anomaly alerts from the River detector."""
    if anomaly_detector is None:
        return {
            "available": False,
            "message": "Anomaly detection not available (install river>=0.21)",
            "alerts": [],
        }
    return {
        "available": True,
        "alerts": anomaly_detector.get_recent_alerts(n),
        "statistics": anomaly_detector.get_statistics(),
    }


@app.get("/anomaly_stats", summary="Get anomaly detection statistics")
def get_anomaly_stats():
    """Return anomaly detector statistics and feature EMAs."""
    if anomaly_detector is None:
        return {"available": False}
    return anomaly_detector.get_statistics()
