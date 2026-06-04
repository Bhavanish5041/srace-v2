"""
gym_env.py — Gymnasium environment for SRACE PPO training.

Observation space:
    Flat vector of:
        zone_people (n_zones,)        — occupants per zone
        zone_temps  (n_zones,)        — temperature °C
        zone_co2    (n_zones,)        — CO₂ ppm
        zone_lux    (n_zones,)        — illuminance lux
        appliance_states (n_appliances,)  — current on/off

Action space:
    MultiBinary(n_appliances) — toggle each appliance on/off

Episode:
    Runs for MAX_STEPS ticks (one tick = 5 seconds simulated).
    Occupancy is randomised at reset.
"""

import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.room_config import RoomConfig, load_config
from core.coverage import CoverageResult
from physics.airflow import compute_airflow_matrix, total_airflow_per_zone
from physics.thermal import simulate_thermal
from physics.co2_model import simulate_co2
from physics.lighting import compute_lux_matrix, total_lux_per_zone
from ml.reward import calculate_reward


# ── Constants ──────────────────────────────────────────────────
MAX_STEPS = 200          # steps per episode
TICK_SECONDS = 5.0       # simulated seconds per step
MAX_PEOPLE_PER_ZONE = 8  # for observation space bounds
MAX_TEMP = 45.0          # °C upper bound
MAX_CO2 = 2000.0         # ppm upper bound
MAX_LUX = 1500.0         # lux upper bound

# Simplified physics constants for fast per-step updates
THERMAL_DECAY = 0.02     # per-step fractional temp change from fans
CO2_DECAY_FAN = 0.01     # per-step CO₂ decay boost per unit airflow
CO2_GENERATION = 5.0     # ppm per person per step
CO2_NATURAL_DECAY = 0.005  # natural ventilation per step
OCCUPANT_HEAT = 0.05     # °C per person per step


class SRACEEnv(gym.Env):
    """
    SRACE Gymnasium environment for PPO training.

    The agent controls n_appliances (fans + lights) to minimise power
    consumption while maintaining comfort in occupied zones.

    Each step simulates one tick of simplified physics (fast enough
    for millions of RL steps without calling scipy.integrate).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str | None = None, render_mode=None):
        """
        Initialise the SRACE environment.

        Args:
            config_path: Path to room JSON config.
                         Defaults to config/default_room.json.
            render_mode: Gymnasium render mode (unused for now).
        """
        super().__init__()
        self.render_mode = render_mode

        # ── Load room config ──
        if config_path is None:
            config_path = os.path.join(_project_root, "config", "default_room.json")
        self.cfg = load_config(config_path)

        self.n_zones = self.cfg.n_zones
        self.n_fans = self.cfg.n_fans
        self.n_lights = self.cfg.n_lights
        self.n_projectors = self.cfg.n_projectors
        self.n_appliances = self.cfg.n_appliances  # fans + lights + projectors

        # ── Pre-compute static physics matrices ──
        self.airflow_matrix = compute_airflow_matrix(self.cfg)
        self.lux_matrix = compute_lux_matrix(self.cfg)

        # ── Build appliance wattages ──
        self.appliance_watts = np.array(
            [f.power_watts for f in self.cfg.fans]
            + [l.power_watts for l in self.cfg.lights]
            + [p.power_watts for p in self.cfg.projectors]
        )

        # ── Action space: binary on/off for each appliance ──
        self.action_space = spaces.MultiBinary(self.n_appliances)

        # ── Observation space ──
        # [zone_people | zone_temps | zone_co2 | zone_lux | appliance_states]
        obs_size = 4 * self.n_zones + self.n_appliances
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,  # all observations normalised to [0, 1]
            shape=(obs_size,),
            dtype=np.float32,
        )

        # ── Build coverage matrix (static — depends only on room layout) ──
        self._build_static_coverage()

        # ── State variables (set in reset) ──
        self.zone_people = np.zeros(self.n_zones)
        self.zone_temps = np.full(self.n_zones, self.cfg.ambient_temp)
        self.zone_co2 = np.full(self.n_zones, self.cfg.ambient_co2)
        self.zone_lux = np.zeros(self.n_zones)
        self.appliance_states = np.zeros(self.n_appliances, dtype=np.int8)
        self.prev_appliance_states = np.zeros(self.n_appliances, dtype=np.int8)
        self.step_count = 0

    def _build_static_coverage(self):
        """Pre-compute a binary coverage matrix from physics."""
        # Use marginal thermal/CO₂ analysis with dummy full occupancy
        dummy_occ = np.full(self.n_zones, 4.0)
        thermal_impact = simulate_thermal(
            self.cfg, self.airflow_matrix, dummy_occ
        )
        co2_reduction = simulate_co2(
            self.cfg, self.airflow_matrix, dummy_occ
        )
        coverage_result = CoverageResult(
            self.cfg, self.airflow_matrix, thermal_impact,
            co2_reduction, self.lux_matrix,
            occupied_zones=set(range(self.n_zones)),
        )
        self.coverage_matrix = coverage_result.binary

    def reset(self, seed=None, options=None):
        """
        Reset the environment with randomised occupancy.

        Returns:
            observation: Normalised flat array.
            info: Empty dict.
        """
        super().reset(seed=seed)

        # ── Randomise occupancy ──
        # Random scenario: 0-60% zone occupation, 0-8 people per zone
        n_occupied = self.np_random.integers(0, self.n_zones + 1)
        occupied_zones = self.np_random.choice(
            self.n_zones, size=n_occupied, replace=False
        )
        self.zone_people = np.zeros(self.n_zones)
        for zi in occupied_zones:
            self.zone_people[zi] = self.np_random.integers(1, MAX_PEOPLE_PER_ZONE + 1)

        # ── Reset physics state ──
        self.zone_temps = np.full(
            self.n_zones,
            self.cfg.ambient_temp + self.np_random.uniform(-1.0, 2.0),
        )
        self.zone_co2 = np.full(
            self.n_zones,
            self.cfg.ambient_co2 + self.np_random.uniform(0, 100),
        )
        self.zone_lux = np.full(self.n_zones, self.cfg.ambient_lux)

        # ── All appliances start OFF ──
        self.appliance_states = np.zeros(self.n_appliances, dtype=np.int8)
        self.prev_appliance_states = np.zeros(self.n_appliances, dtype=np.int8)
        self.step_count = 0

        return self._get_observation(), {}

    def step(self, action):
        """
        Apply action, run one physics tick, compute reward.

        Args:
            action: Binary array (n_appliances,) — 1 = ON.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # ── Apply action ──
        self.prev_appliance_states = self.appliance_states.copy()
        self.appliance_states = np.array(action, dtype=np.int8)

        # ── Run simplified physics (fast, no ODE solver) ──
        self._physics_tick()

        # ── Calculate reward ──
        reward = calculate_reward(
            appliance_states=self.appliance_states,
            zone_people=self.zone_people,
            coverage_matrix=self.coverage_matrix,
            prev_appliance_states=self.prev_appliance_states,
            appliance_watts=self.appliance_watts,
            zone_temps=self.zone_temps,
            zone_co2=self.zone_co2,
            zone_lux=self.zone_lux,
            comfort_targets={
                "target_temp": self.cfg.comfort.target_temp_c,
                "target_lux": self.cfg.comfort.target_lux,
                "max_co2": self.cfg.comfort.max_co2_ppm,
            },
        )

        # ── Episode termination ──
        self.step_count += 1
        terminated = False  # no early termination
        truncated = self.step_count >= MAX_STEPS

        info = {
            "total_power_w": float(np.dot(self.appliance_states, self.appliance_watts)),
            "n_active": int(self.appliance_states.sum()),
            "avg_temp": float(self.zone_temps.mean()),
            "avg_co2": float(self.zone_co2.mean()),
            "n_occupied": int((self.zone_people > 0).sum()),
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _physics_tick(self):
        """
        Run one simplified physics step.

        Uses fast analytical updates instead of ODE solvers so RL
        training can do millions of steps without bottlenecking on
        scipy.integrate.
        """
        fan_states = self.appliance_states[:self.n_fans].astype(bool)
        light_end = self.n_fans + self.n_lights
        light_states = self.appliance_states[self.n_fans:light_end].astype(bool)
        proj_states = self.appliance_states[light_end:].astype(bool) if self.n_projectors > 0 else np.array([], dtype=bool)

        # ── Airflow → Temperature ──
        if fan_states.any():
            airflow = total_airflow_per_zone(self.airflow_matrix, fan_states)
        else:
            airflow = np.zeros(self.n_zones)

        # Cooling from fans (proportional to airflow and temp delta)
        target_t = self.cfg.comfort.target_temp_c
        cooling = THERMAL_DECAY * airflow * (self.zone_temps - target_t)
        self.zone_temps -= cooling

        # Heating from occupants
        self.zone_temps += OCCUPANT_HEAT * self.zone_people

        # Heat leak from ambient (walls)
        self.zone_temps += 0.005 * (self.cfg.ambient_temp - self.zone_temps)

        # Clamp to physical range
        self.zone_temps = np.clip(self.zone_temps, 15.0, MAX_TEMP)

        # ── CO₂ ──
        # Generation from people
        self.zone_co2 += CO2_GENERATION * self.zone_people

        # Removal from fan ventilation
        co2_removal = CO2_DECAY_FAN * airflow * (self.zone_co2 - self.cfg.ambient_co2)
        self.zone_co2 -= co2_removal

        # Natural ventilation
        self.zone_co2 -= CO2_NATURAL_DECAY * (self.zone_co2 - self.cfg.ambient_co2)

        # Clamp
        self.zone_co2 = np.clip(self.zone_co2, self.cfg.ambient_co2, MAX_CO2)

        # ── Lighting ──
        # Combine lights + projectors for lux computation
        if light_states.any():
            self.zone_lux = total_lux_per_zone(
                self.lux_matrix, light_states, self.cfg.ambient_lux
            )
        else:
            self.zone_lux = np.full(self.n_zones, self.cfg.ambient_lux)

        # Add projector lux contribution
        if self.n_projectors > 0 and proj_states.any():
            for pi, proj in enumerate(self.cfg.projectors):
                if not proj_states[pi]:
                    continue
                for zi in range(self.n_zones):
                    z = self.cfg.zones[zi]
                    dx = proj.x - z.cx
                    dy = proj.y - z.cy
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= proj.coverage_radius:
                        # Linear falloff within coverage radius
                        falloff = 1.0 - (dist / proj.coverage_radius)
                        self.zone_lux[zi] += proj.screen_lux * falloff

    def _get_observation(self) -> np.ndarray:
        """
        Flatten all state into a single normalised observation vector.

        Layout: [zone_people | zone_temps | zone_co2 | zone_lux | appliance_states]
        All normalised to [0, 1] for stable RL training.
        """
        obs = np.concatenate([
            self.zone_people / MAX_PEOPLE_PER_ZONE,
            self.zone_temps / MAX_TEMP,
            self.zone_co2 / MAX_CO2,
            self.zone_lux / MAX_LUX,
            self.appliance_states.astype(np.float32),
        ])
        return obs.astype(np.float32)
