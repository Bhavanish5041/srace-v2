"""
room_config.py — Generalised room configuration loader.

Loads any room layout from a JSON file and computes zone geometry.
Works for classrooms, offices, auditoriums — anything with fans and lights.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Zone:
    """A rectangular zone within the room."""
    row: int
    col: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cx: float  # center x
    cy: float  # center y
    area: float  # m²

    @property
    def index(self) -> int:
        """Flat index for matrix operations (row-major)."""
        return self.row * self._total_cols + self.col

    def __post_init__(self):
        self._total_cols = 0  # set after construction

    def set_grid_info(self, total_cols: int):
        self._total_cols = total_cols


@dataclass
class Fan:
    """A ceiling fan appliance."""
    id: str
    x: float
    y: float
    power_watts: float
    airflow_radius: float  # metres — 95% of airflow within this radius
    airflow_peak_ms: float  # peak airflow speed at fan centre (m/s)
    idx: int = 0  # index in the appliance list


@dataclass
class Light:
    """A ceiling light appliance."""
    id: str
    x: float
    y: float
    power_watts: float
    lumens: float
    height_above_floor: float  # metres
    idx: int = 0


@dataclass
class Projector:
    """A ceiling-mounted projector appliance."""
    id: str
    x: float
    y: float
    power_watts: float
    screen_lux: float
    coverage_radius: float
    height_above_floor: float
    idx: int = 0


@dataclass
class ComfortParams:
    """Target comfort thresholds."""
    target_temp_c: float
    target_lux: float
    max_co2_ppm: float
    min_airflow_ms: float


@dataclass
class RoomConfig:
    """
    Complete room configuration — loaded from JSON, works for any room.

    Attributes:
        name: Human-readable room name
        width: Room width in metres (x-axis)
        depth: Room depth in metres (y-axis)
        ceiling_height: Ceiling height in metres
        ambient_temp: Baseline ambient temperature °C
        ambient_co2: Outdoor / baseline CO₂ in ppm
        ambient_lux: Ambient light level without artificial lighting
        n_zone_cols: Number of zone columns
        n_zone_rows: Number of zone rows
        zones: List of Zone objects with computed geometry
        fans: List of Fan objects
        lights: List of Light objects
        comfort: Target comfort parameters
    """
    name: str
    width: float
    depth: float
    ceiling_height: float
    ambient_temp: float
    ambient_co2: float
    ambient_lux: float
    n_zone_cols: int
    n_zone_rows: int
    zones: list[Zone] = field(default_factory=list)
    fans: list[Fan] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    projectors: list[Projector] = field(default_factory=list)
    comfort: ComfortParams = None

    @property
    def n_zones(self) -> int:
        return self.n_zone_rows * self.n_zone_cols

    @property
    def n_fans(self) -> int:
        return len(self.fans)

    @property
    def n_lights(self) -> int:
        return len(self.lights)

    @property
    def n_projectors(self) -> int:
        return len(self.projectors)

    @property
    def n_appliances(self) -> int:
        return self.n_fans + self.n_lights + self.n_projectors

    @property
    def all_appliances(self) -> list:
        """All appliances in order: fans, lights, projectors."""
        return self.fans + self.lights + self.projectors

    def zone_at(self, row: int, col: int) -> Zone:
        """Get zone by grid coordinates."""
        return self.zones[row * self.n_zone_cols + col]

    def zone_flat(self, idx: int) -> Zone:
        """Get zone by flat index."""
        return self.zones[idx]


def _validate_bounds(item: Any, width: float, depth: float, label: str):
    """Ensure an appliance is within room bounds."""
    if not (0 <= item.x <= width):
        raise ValueError(
            f"{label} '{item.id}' x={item.x} is outside room width [0, {width}]"
        )
    if not (0 <= item.y <= depth):
        raise ValueError(
            f"{label} '{item.id}' y={item.y} is outside room depth [0, {depth}]"
        )


def load_config(path: str | Path) -> RoomConfig:
    """
    Load a room configuration from a JSON file.

    This is the single entry point — give it any valid room JSON and it
    returns a fully populated RoomConfig ready for physics + optimization.

    Args:
        path: Path to the JSON config file.

    Returns:
        RoomConfig with computed zone geometry and validated appliances.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # --- Room basics ---
    room = data["room"]
    width = float(room["width"])
    depth = float(room["depth"])
    ceiling = float(room.get("ceiling_height", 3.0))

    # --- Zone grid ---
    zone_cfg = data["zones"]
    n_cols = int(zone_cfg["cols"])
    n_rows = int(zone_cfg["rows"])

    zone_w = width / n_cols
    zone_h = depth / n_rows

    zones: list[Zone] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x_min = c * zone_w
            x_max = (c + 1) * zone_w
            y_min = r * zone_h
            y_max = (r + 1) * zone_h
            z = Zone(
                row=r, col=c,
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                cx=(x_min + x_max) / 2,
                cy=(y_min + y_max) / 2,
                area=zone_w * zone_h,
            )
            z.set_grid_info(n_cols)
            zones.append(z)

    # --- Fans ---
    fans: list[Fan] = []
    for i, fd in enumerate(data.get("fans", [])):
        fan = Fan(
            id=fd["id"],
            x=float(fd["x"]),
            y=float(fd["y"]),
            power_watts=float(fd["power_watts"]),
            airflow_radius=float(fd["airflow_radius"]),
            airflow_peak_ms=float(fd["airflow_peak_ms"]),
            idx=i,
        )
        _validate_bounds(fan, width, depth, "Fan")
        fans.append(fan)

    # --- Lights ---
    lights: list[Light] = []
    for i, ld in enumerate(data.get("lights", [])):
        light = Light(
            id=ld["id"],
            x=float(ld["x"]),
            y=float(ld["y"]),
            power_watts=float(ld["power_watts"]),
            lumens=float(ld["lumens"]),
            height_above_floor=float(ld.get("height_above_floor", ceiling)),
            idx=i,
        )
        _validate_bounds(light, width, depth, "Light")
        lights.append(light)

    # --- Projectors ---
    projectors: list[Projector] = []
    for i, pd in enumerate(data.get("projectors", [])):
        projector = Projector(
            id=pd["id"],
            x=float(pd["x"]),
            y=float(pd["y"]),
            power_watts=float(pd["power_watts"]),
            screen_lux=float(pd["screen_lux"]),
            coverage_radius=float(pd["coverage_radius"]),
            height_above_floor=float(pd.get("height_above_floor", ceiling)),
            idx=i,
        )
        _validate_bounds(projector, width, depth, "Projector")
        projectors.append(projector)

    # --- Comfort ---
    cc = data.get("comfort", {})
    comfort = ComfortParams(
        target_temp_c=float(cc.get("target_temp_c", 25.0)),
        target_lux=float(cc.get("target_lux", 300.0)),
        max_co2_ppm=float(cc.get("max_co2_ppm", 1000.0)),
        min_airflow_ms=float(cc.get("min_airflow_ms", 0.5)),
    )

    config = RoomConfig(
        name=room.get("name", "Unnamed Room"),
        width=width,
        depth=depth,
        ceiling_height=ceiling,
        ambient_temp=float(room.get("ambient_temp", 30.0)),
        ambient_co2=float(room.get("ambient_co2", 400.0)),
        ambient_lux=float(room.get("ambient_lux", 50.0)),
        n_zone_cols=n_cols,
        n_zone_rows=n_rows,
        zones=zones,
        fans=fans,
        lights=lights,
        projectors=projectors,
        comfort=comfort,
    )

    return config


def print_config_summary(cfg: RoomConfig):
    """Pretty-print a config summary to terminal."""
    print(f"\n{'='*60}")
    print(f"  SRACE Room: {cfg.name}")
    print(f"{'='*60}")
    print(f"  Dimensions : {cfg.width} × {cfg.depth} m  (ceiling {cfg.ceiling_height} m)")
    print(f"  Zone grid  : {cfg.n_zone_rows} rows × {cfg.n_zone_cols} cols = {cfg.n_zones} zones")
    print(f"  Fans       : {cfg.n_fans}  (total {sum(f.power_watts for f in cfg.fans):.0f} W)")
    print(f"  Lights     : {cfg.n_lights}  (total {sum(l.power_watts for l in cfg.lights):.0f} W)")
    print(f"  Projectors : {cfg.n_projectors}  (total {sum(p.power_watts for p in cfg.projectors):.0f} W)")
    print(f"  Max power  : {sum(a.power_watts for a in cfg.all_appliances):.0f} W")
    print(f"  Comfort    : {cfg.comfort.target_temp_c}°C  {cfg.comfort.target_lux} lux"
          f"  <{cfg.comfort.max_co2_ppm} ppm CO₂")
    print(f"{'='*60}\n")
