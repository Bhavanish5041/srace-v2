# SRACE v2 - Smart Room Automation & Control Engine

> Generalised intelligent room automation platform that works for **any room** —
> classroom, office, staff room, auditorium — controlled via a single JSON config file.
> Not a fixed installation but a deployable system.

---

## The Problem

Institutional buildings waste **40% electricity** running all fans and lights
regardless of how many people are present or where they sit. Existing systems use
dumb binary thresholds - everything on or everything off.

SRACE fixes this with **real intelligence**: physics-based models, mathematical
optimization, and reinforcement learning to run only the appliances that matter.

---

## System Architecture

```
S23 Ultra / Unity NavMesh Agents
  → YOLOv8 / ArUco / Zone triggers
  → Zone counts every 5 seconds
        ↓
Physics Engine (Python)
  → Airflow: Gaussian decay
  → Thermal: PDE solved with RK45
  → CO₂: Mass balance ODE
  → Lighting: Cosine-law (inverse square)
  → Output: Coverage matrix C[appliance][zone]
        ↓
Set Cover Optimizer
  → Greedy (real time, every 30s)
  → ILP via PuLP (exact optimal, every 60s)
        ↓
PPO RL Agent (Stable-Baselines3)
  → State: crowd + temps + CO₂ + lux + appliance states
  → Action: which fans/lights to toggle
  → Reward: −power + comfort − switching + air_quality − danger
        ↓
FastAPI Backend
  → REST endpoints (GET /room_state, POST /set_occupancy)
  → Unity polls every 5 seconds
        ↓
Two simultaneous outputs:
  Unity 3D → code-generated room, spinning fans, glowing lights, zone heatmap
  FastAPI Swagger UI → live data, power metrics
```

---

## Current Progress

### ✅ Week 1 — Python Core (Complete)
- [x] Generalised room config loader (JSON → dataclasses)
- [x] 4 physics models (airflow, thermal, CO₂, lighting)
- [x] Coverage matrix builder
- [x] Greedy weighted set cover optimizer
- [x] ILP exact optimizer (PuLP CBC)
- [x] Full pipeline orchestrator (`main.py` with 5 test scenarios)
- [x] Matplotlib room grid visualization

### ✅ Week 2 — Unity Simulation (Complete)
- [x] Code-generated 3D room from JSON config (floor, walls, ceiling)
- [x] Fan objects with spinning blades, spin-up/down animation, color states
- [x] Light objects with Unity Light component, emission glow, halos
- [x] Zone heatmap (red=occupied+uncovered, green=covered, grey=empty)
- [x] Orbit camera with mouse controls
- [x] Keyboard presets (1-5) for occupancy scenarios
- [x] C# ports of airflow and lighting physics
- [x] Binary coverage matrix (C# port)
- [x] Two-phase greedy optimizer (fans and lights run separately)
- [x] Analytical thermal impact estimation for fan coverage

### ✅ Week 3 — RL + Backend + Integration (Complete)
- [x] Custom Gymnasium environment (`SRACEEnv`) with simplified physics
- [x] PPO training pipeline (Stable-Baselines3, configurable timesteps)
- [x] Multi-objective reward function (5-term: power, comfort, switching, air quality, danger)
- [x] PPO evaluation framework with baseline comparison
- [x] FastAPI REST backend (`/room_state`, `/set_occupancy`, `/config`)
- [x] Unity ↔ Python bridge (`SRACEApiClient` polls backend every 5s)
- [x] Trained PPO model saved to `models/srace_ppo.zip`

---

## Repository Structure

```
srace-v2/
├── config/
│   ├── default_room.json             # 10×8m classroom, 4×3 grid, 10 fans, 10 lights
│   └── classroom_real.json           # Real classroom dimensions (10.8×7.6m)
│
├── core/
│   ├── room_config.py                # RoomConfig, Zone, Fan, Light, ComfortParams
│   └── coverage.py                   # Binary coverage matrix from physics
│
├── physics/
│   ├── airflow.py                    # Gaussian decay fan airflow model
│   ├── thermal.py                    # Thermal ODE (Method of Lines + RK45)
│   ├── co2_model.py                  # CO₂ mass-balance ODE
│   └── lighting.py                   # Cosine-law illuminance model
│
├── optimizer/
│   ├── greedy_solver.py              # Greedy weighted set cover (fans/lights separate)
│   └── ilp_solver.py                 # Exact ILP via PuLP CBC
│
├── ml/
│   ├── gym_env.py                    # Custom Gymnasium environment for PPO
│   ├── reward.py                     # 5-term reward: power, comfort, switching, air, danger
│   ├── train_ppo.py                  # PPO training (SB3, EvalCallback, resume support)
│   └── test_ppo.py                   # Evaluation with scenarios + baseline comparison
│
├── backend/
│   └── api.py                        # FastAPI server (room_state, set_occupancy, config)
│
├── models/
│   ├── srace_ppo.zip                 # Trained PPO model
│   ├── best/                         # Best model checkpoint (from EvalCallback)
│   └── eval_logs/                    # Training evaluation logs
│
├── visualization/
│   └── room_grid.py                  # Matplotlib room grid visualization
│
├── Assets/SRACE/Scripts/             # Unity C# simulation
│   ├── Core/
│   │   ├── RoomConfig.cs             # C# data model (Zone, Fan, Light, ComfortParams)
│   │   ├── RoomConfigLoader.cs       # JSON → RoomConfig parser
│   │   ├── CoverageMatrix.cs         # Binary coverage matrix (C# port)
│   │   ├── SRACEManager.cs           # Main orchestrator (physics → greedy → visuals)
│   │   └── SRACEApiClient.cs         # Unity ↔ Python REST bridge
│   ├── Environment/
│   │   ├── RoomBuilder.cs            # Code-generates floor, walls, ceiling
│   │   ├── FanObject.cs              # Ceiling fan with spinning blades
│   │   ├── LightObject.cs            # Ceiling light with glow + halo
│   │   ├── ZoneHeatmap.cs            # Color-coded zone overlay
│   │   └── ClassroomCamera.cs        # Orbit camera with mouse control
│   └── Physics/
│       ├── AirflowModel.cs           # Gaussian airflow (C# port)
│       └── LightingModel.cs          # Cosine-law lux (C# port)
│
├── output/                           # Generated Matplotlib visualizations
├── main.py                           # Full pipeline: config → physics → optimizer → output
└── requirements.txt                  # Python dependencies
```

---

## Mathematics & Physics

| Formula | Used For | Module |
|---------|----------|--------|
| `v(f,z) = v_peak · exp(−d²/(2σ²))`, σ = radius/2 | Airflow per zone from each fan | `physics/airflow.py` |
| `dT/dt = −α·v·(T−T_target) + Q_occ/(ρ·cp·V) + k·(T_amb−T)` | Temperature forecast (5 min, RK45) | `physics/thermal.py` |
| `dC/dt = (n·G)/(V) − λ_vent·(C−C_ambient)` | CO₂ per zone over time | `physics/co2_model.py` |
| `E = (Φ·cos³θ)/(2π·h²)` | Lux per zone from each ceiling light | `physics/lighting.py` |
| `min Σ w_j·x_j  s.t. coverage ≥ 1 ∀ occupied zones` | Minimum power appliance subset | `optimizer/ilp_solver.py` |

### PPO Reward Function

```
R = −α·power + β·comfort − γ·switching + δ·air_quality − ε·danger

α = 0.15  (power penalty)
β = 0.55  (comfort: 40% temp + 30% CO₂ + 15% coverage + 15% lux)
γ = 0.05  (switching penalty)
δ = 0.15  (air quality bonus)
ε = 0.50  (danger penalty: >35°C or >1200 ppm CO₂)
```

### Physical Constants

```
AIR_DENSITY        = 1.2 kg/m³
SPECIFIC_HEAT      = 1005 J/(kg·K)
OCCUPANT_HEAT      = 100 W sensible per person
CONVECTION_COEFF   = 0.15 (airflow → cooling)
WALL_CONDUCTANCE   = 0.01 (1/s, heat leak)
CO2_EXHALE_RATE    = 0.005 L/s per person
BASE_VENTILATION   = 0.0005 (1/s, natural)
FAN_VENT_COEFF     = 0.002 (extra ventilation per m/s airflow)
FORECAST_WINDOW    = 300 seconds (5 minutes)
```

---

## Running

### Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Python Pipeline

```bash
# Run full physics + optimizer pipeline with 5 test scenarios
python3 main.py
# Output: terminal results + visualizations in ./output/
```

### PPO Training & Evaluation

```bash
# Train PPO agent (quick test)
python3 ml/train_ppo.py --timesteps 1000

# Full training run
python3 ml/train_ppo.py --timesteps 500000

# Resume training from checkpoint
python3 ml/train_ppo.py --timesteps 500000 --resume models/srace_ppo.zip

# Evaluate trained model (with per-step details)
python3 ml/test_ppo.py --episodes 10 --render

# Test specific occupancy scenarios
python3 ml/test_ppo.py --scenario full --render     # all zones occupied
python3 ml/test_ppo.py --scenario sparse --render    # 2-3 zones only
python3 ml/test_ppo.py --scenario empty --render     # no occupants

# Test on a completely different room (zero-shot generalisation)
python3 ml/test_ppo.py --config config/office_small.json --scenario full --render
python3 ml/test_ppo.py --config config/auditorium.json --scenario sparse --render
```

### FastAPI Backend

```bash
# Start API server
source venv/bin/activate
uvicorn backend.api:app --reload --port 8000

# Swagger UI: http://localhost:8000/docs
# Endpoints:
#   GET  /room_state      — physics + greedy optimizer result
#   POST /set_occupancy   — update zone occupancy
#   GET  /ppo_action      — PPO agent decision (with physics state)
#   POST /ppo_reset       — reset PPO agent state
#   GET  /compare         — greedy vs PPO side-by-side
#   GET  /config          — raw room configuration
```

### Unity Simulation

1. Open the project in Unity (2022.3+ with URP)
2. Drag `default_room.json` or `classroom_real.json` into `Assets/SRACE/Resources/`
3. Add `SRACEManager` to an empty GameObject
4. Press Play

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `1` | Empty room (all off) |
| `2` | Single person in zone 0 |
| `3` | Center cluster (zones 5,6,9,10) |
| `4` | Full room (4 per zone) |
| `5` | Front row only |
| `Space` | Re-run optimizer |
| `F` | Toggle all fans |
| `L` | Toggle all lights |

**API mode:** Add `SRACEApiClient` component to enable Python backend polling.

---

## Test Scenarios

| Scenario | Occupancy | Expected Behavior |
|----------|-----------|-------------------|
| Empty Room | 0 people | All appliances OFF, 0W |
| Single Person | 1 in zone 0 | 1-2 fans + 1-2 lights near zone 0 |
| Center Cluster | 11 in zones 5,6,9,10 | Center fans + lights only |
| Full Room | 48 people (4/zone) | Most appliances ON, near max power |
| Front Row Only | 20 in zones 0-3 | Front-row appliances only, back OFF |

---

## Key Design Decisions

### Optimizer: Two-Phase Greedy

The greedy optimizer runs fans and lights in **separate phases** instead of together.
Without this, lights (40W) always beat fans (75W) on coverage-per-watt, causing the
optimizer to never select fans — leaving rooms with zero airflow despite full
lighting coverage.

### PPO Reward Engineering

The reward function uses a **5-term design** with a hard danger penalty. Early
versions used only 4 terms (power α=0.3, comfort β=0.5, switching γ=0.1, air δ=0.1),
which caused the agent to prefer lights over fans because:
1. Power penalty was too dominant (fans cost more watts)
2. Temperature comfort penalty was too gentle (`exp(-0.1·dev²)`)
3. No punishment for unsafe conditions

The fix: reduced power weight, steeper comfort curves (`exp(-0.3·dev²)`), and
a new danger term (ε=0.5) for zones above 35°C or 1200 ppm CO₂.

### Unity: Code-Generated Room

The entire 3D room (floor, walls, ceiling, fans, lights, heatmap) is **generated
from the JSON config at runtime** — no manual scene setup. Change the JSON config
and the room rebuilds automatically.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Physics | Python, NumPy, SciPy (`solve_ivp` RK45) |
| Optimization | PuLP ILP (CBC solver) |
| Machine Learning | PyTorch, Stable-Baselines3 PPO, Gymnasium |
| Backend | FastAPI, Uvicorn |
| 3D Simulation | Unity 2022.3+ (URP), C# |
| Vision (planned) | YOLOv8, OpenCV ArUco |
| Hardware (planned) | Raspberry Pi 4, 20 LEDs |
| Dashboard (planned) | React, Recharts |

---

## Subject Mappings (Academic)

| Subject | SRACE Component |
|---------|----------------|
| **Computer Networks** | REST API, Unity ↔ Python polling, JSON protocol |
| **Data Analysis & Algorithms** | Greedy O(n log n), ILP solver comparison, coverage analysis |
| **Discrete Mathematics** | Weighted Set Cover (NP-hard), ILP binary variables, bipartite coverage graph |
| **AI and ML** | PPO RL agent, custom Gymnasium env, multi-objective reward shaping |

---

## Team

| Member | Owns |
|--------|------|
| **Bhavanish** | Physics engine, ILP + Greedy optimizer, PPO RL agent, reward engineering, FastAPI backend, Unity integration |
| Person 2 | ArUco detector, YOLOv8 integration, zone mapper, GPIO controller |
| Person 3 | FastAPI gateway extensions, WebSocket, InfluxDB, Redis, Docker |
| Person 4 | React dashboard, Grafana, UE5 assets |

---

## What Makes This Project Stand Out

1. **Generalised** - any room via JSON config, not a one-off hack
2. **Multi-domain** - CV + RL + combinatorics + physics + IoT + distributed systems
3. **Mathematically rigorous** - ILP gives provably optimal solution
4. **Physically accurate** - ODEs and PDEs, not just thresholds
5. **Dual optimizer** — Greedy for real-time + ILP for verification
6. **RL with reward engineering** — 5-term reward with danger penalties
7. **Full-stack** — Python physics → REST API → Unity 3D simulation
