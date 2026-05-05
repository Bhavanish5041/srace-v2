"""
backend/api.py — SRACE v2 FastAPI server.

Exposes room state, physics, and optimizer results as REST endpoints.
Unity, the React dashboard, or any HTTP client can consume these.

Endpoints:
    GET  /room_state     — Run physics + greedy optimizer, return full state
    POST /set_occupancy  — Update zone occupancy counts
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

# Current occupancy state (mutable — updated via POST /set_occupancy)
zone_occupancy: np.ndarray = None


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
    type: str  # "fan" or "light"
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
        total_people=int(zone_occupancy.sum()),
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
