# SRACE v2 - Smart Room Automation & Control Engine

> Generalised intelligent room automation platform that works for **any room**
> (classroom, office, staff room, auditorium) controlled via a single JSON config file.
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
  > YOLOv8 / ArUco / Zone triggers
  > Zone counts every 5 seconds
        |
Physics Engine (Python + C#)
  > Airflow: Gaussian decay
  > Thermal: ODE solved with RK45 (Python) / RK4 (C#)
  > CO2: Mass balance ODE (Python RK45 / C# RK4)
  > Lighting: Cosine-law (inverse square)
  > Output: Coverage matrix C[appliance][zone] (fans + lights + projectors)
        |
Optimizer Tier (3 solvers)
  > Greedy (real time, every 30s) — 3-phase: fans → lights → projectors
  > ILP via PuLP (exact optimal, every 60s)
  > GA evolutionary (multi-objective, 80 generations)
        |
PPO RL Agent (Stable-Baselines3)
  > State: crowd + temps + CO2 + lux + appliance states (incl. projectors)
  > Action: which fans/lights/projectors to toggle
  > Reward: -power + comfort - switching + air_quality - danger + stability
        |
Real-Time Communication
  > MQTT bridge (paho-mqtt) — pub/sub for RPi, Unity, dashboard
  > FastAPI REST — 10+ endpoints (room_state, ppo_action, anomalies, mqtt_status)
  > River anomaly detection — streaming ML for sensor drift alerts
        |
Four simultaneous outputs:
  Unity 3D with Power HUD > fans, lights, projectors, temp/CO₂ bars, zone heatmap
  Live HTML Dashboard     > projector chips, anomaly alerts, MQTT status
  FastAPI Swagger UI      > live data, power metrics, anomaly stats
  RPi LED Panel           > MQTT-driven appliance state indicators
```

---

## Current Progress

### Week 1: Python Core (Complete)
- [x] Generalised room config loader (JSON to dataclasses)
- [x] 4 physics models (airflow, thermal, CO2, lighting)
- [x] Coverage matrix builder
- [x] Greedy weighted set cover optimizer
- [x] ILP exact optimizer (PuLP CBC)
- [x] Full pipeline orchestrator (`main.py` with 5 test scenarios)
- [x] Matplotlib room grid visualization

### Week 2: Unity Simulation (Complete)
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
- [x] Live Power HUD overlay with animated bar and savings percentage

### Week 3: RL + Backend + Integration (Complete)
- [x] Custom Gymnasium environment (`SRACEEnv`) with simplified physics
- [x] PPO training pipeline (Stable-Baselines3, configurable timesteps)
- [x] Multi-objective reward function (7-term: power, comfort, switching, air quality, danger, stability, consistency)
- [x] Empty room fix: hard penalty for wasted energy when nobody is present
- [x] PPO evaluation framework with baseline comparison
- [x] Cross-room generalisation (test on unseen office/auditorium layouts)
- [x] FastAPI REST backend with 10+ endpoints
- [x] Unity to Python bridge (`SRACEApiClient` polls backend every 5s)
- [x] Genetic Algorithm optimizer (tournament selection, crossover, mutation)
- [x] Live HTML dashboard (polls API, scenario buttons, greedy/PPO toggle)
- [x] Visualization charts (PPO step-by-step + baseline comparison bar chart)
- [x] Trained PPO model saved to `models/srace_ppo.zip`
- [x] 4 room configs (classroom, office, auditorium, conference room)

### Week 4: Advanced Features (Complete)
- [x] Projector appliance type (3rd appliance alongside fans/lights)
- [x] C# Thermal ODE model (RK4 solver, marginal analysis per fan)
- [x] C# CO₂ mass-balance ODE model (RK4 solver)
- [x] Real-time temp/CO₂/lux tracking in Unity HUD (live bars + color-coded)
- [x] PPO environment updated for projector awareness (21 appliances)
- [x] Coverage matrix expanded for projectors (distance-based coverage)
- [x] MQTT real-time bridge (`paho-mqtt`, 6 topics, thread-safe)
- [x] River streaming anomaly detection (HalfSpaceTrees + rule-based alerts)
- [x] Dashboard upgrade: projector chips, MQTT badge, anomaly alerts, dynamic grid
- [x] PPO diagnostics toolkit (`diagnose_ppo.py`, 6-section report)
- [x] Backend: projector support in all endpoints, anomaly/MQTT endpoints

---

## Repository Structure

```
srace-v2/
├── config/
│   ├── default_room.json             # 10x8m classroom, 4x3 grid, 10F/10L/1P
│   ├── classroom_real.json           # Real classroom dimensions (10.8x7.6m)
│   ├── office_small.json             # 6x5m office, 2x2 grid, 4F/4L/1P
│   ├── auditorium.json               # 15x10m hall, 5x4 grid, 16F/16L/2P
│   └── conference_room.json          # 8x6m room, 3x2 grid, 6F/6L/2P
│
├── core/
│   ├── room_config.py                # RoomConfig, Zone, Fan, Light, Projector
│   └── coverage.py                   # Coverage matrix (fans + lights + projectors)
│
├── physics/
│   ├── airflow.py                    # Gaussian decay fan airflow model
│   ├── thermal.py                    # Thermal ODE (RK45)
│   ├── co2_model.py                  # CO₂ mass-balance ODE (RK45)
│   └── lighting.py                   # Cosine-law illuminance model
│
├── optimizer/
│   ├── greedy_solver.py              # Greedy set cover (3-phase: fans→lights→projectors)
│   ├── ilp_solver.py                 # Exact ILP via PuLP CBC
│   └── ga_solver.py                  # Genetic Algorithm
│
├── ml/
│   ├── gym_env.py                    # Gymnasium env (fans+lights+projectors)
│   ├── reward.py                     # 7-term reward function
│   ├── train_ppo.py                  # PPO training (SB3, EvalCallback)
│   ├── test_ppo.py                   # Evaluation + cross-room testing
│   ├── diagnose_ppo.py               # 6-section diagnostic report generator
│   └── anomaly_detector.py           # River streaming anomaly detection
│
├── backend/
│   ├── api.py                        # FastAPI (10+ endpoints, MQTT, anomaly)
│   └── mqtt_bridge.py                # Paho MQTT pub/sub bridge (6 topics)
│
├── models/
│   ├── srace_ppo.zip                 # Trained PPO model
│   └── best/                         # Best model checkpoint
│
├── Assets/SRACE/Scripts/             # Unity C# simulation
│   ├── Core/
│   │   ├── RoomConfig.cs             # C# data model (Fan, Light, Projector)
│   │   ├── RoomConfigLoader.cs       # JSON → RoomConfig parser
│   │   ├── CoverageMatrix.cs         # Coverage matrix (C#)
│   │   ├── SRACEManager.cs           # Orchestrator + real-time physics tick
│   │   └── SRACEApiClient.cs         # REST + PPO/Greedy toggle
│   ├── Environment/
│   │   ├── RoomBuilder.cs            # Code-generated 3D room
│   │   ├── FanObject.cs              # Spinning fan with states
│   │   ├── LightObject.cs            # Glowing light with halo
│   │   ├── ProjectorObject.cs        # Projector with spot lights
│   │   ├── ZoneHeatmap.cs            # Color-coded zone overlay
│   │   ├── PowerHUD.cs               # Power + Temp + CO₂ HUD
│   │   └── ClassroomCamera.cs        # Orbit camera
│   └── Physics/
│       ├── AirflowModel.cs           # Gaussian airflow (C#)
│       ├── LightingModel.cs          # Cosine-law lux (C#)
│       ├── ThermalModel.cs           # Thermal ODE — RK4 solver (C#)
│       └── CO2Model.cs               # CO₂ ODE — RK4 solver (C#)
│
├── dashboard.html                    # Live dashboard (MQTT badge, anomaly alerts)
├── main.py                           # Full pipeline orchestrator
└── requirements.txt                  # numpy, scipy, paho-mqtt, river, sb3
```

---

## Mathematics & Physics

| Formula | Used For | Module |
|---------|----------|--------|
| `v(f,z) = v_peak * exp(-d^2/(2*sigma^2))`, sigma = radius/2 | Airflow per zone from each fan | `physics/airflow.py` |
| `dT/dt = -a*v*(T-T_target) + Q_occ/(rho*cp*V) + k*(T_amb-T)` | Temperature forecast (5 min, RK45) | `physics/thermal.py` |
| `dC/dt = (n*G)/(V) - lambda_vent*(C-C_ambient)` | CO2 per zone over time | `physics/co2_model.py` |
| `E = (Phi*cos^3(theta))/(2*pi*h^2)` | Lux per zone from each ceiling light | `physics/lighting.py` |
| `min sum(w_j * x_j) s.t. coverage >= 1 for all occupied zones` | Minimum power appliance subset | `optimizer/ilp_solver.py` |
| Fitness = coverage_fraction - alpha * power_fraction | GA multi-objective evolution | `optimizer/ga_solver.py` |

### PPO Reward Function

```
R = -alpha*power + beta*comfort - gamma*switching + delta*air_quality - epsilon*danger

alpha   = 0.15  (power penalty, doubled for <30% occupancy)
beta    = 0.55  (comfort: 40% temp + 30% CO2 + 15% coverage + 15% lux)
gamma   = 0.05  (switching penalty)
delta   = 0.15  (air quality bonus)
epsilon = 0.50  (danger penalty: >35C or >1200 ppm CO2)

Empty room: all off = +0.5, each active appliance = -0.5 penalty
```

### Physical Constants

```
AIR_DENSITY        = 1.2 kg/m^3
SPECIFIC_HEAT      = 1005 J/(kg*K)
OCCUPANT_HEAT      = 100 W sensible per person
CONVECTION_COEFF   = 0.15 (airflow to cooling)
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
# Shows Greedy vs ILP vs GA comparison for each scenario
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

### Visualization Charts

```bash
# Generate PPO comparison charts (saved to output/)
python3 visualize_results.py
# Output: ppo_sparse_episode.png, ppo_full_episode.png, ppo_vs_baselines.png
```

### FastAPI Backend

```bash
# Start API server
source venv/bin/activate
uvicorn backend.api:app --reload --port 8000

# Swagger UI: http://localhost:8000/docs
# Endpoints:
#   GET  /room_state      physics + greedy optimizer (with temp/CO₂/lux)
#   POST /set_occupancy   update zone occupancy
#   GET  /ppo_action      PPO agent decision (with physics state)
#   POST /ppo_reset       reset PPO agent state
#   GET  /compare         greedy vs PPO side-by-side
#   GET  /config          raw room configuration
#   GET  /mqtt_status     MQTT bridge connection status
#   GET  /anomalies       recent anomaly alerts from River
#   GET  /anomaly_stats   anomaly detector statistics
```

### Live Dashboard

```bash
# Start the API first, then open dashboard in browser
uvicorn backend.api:app --port 8000
# Open dashboard.html in any browser
# Features: scenario buttons, greedy/PPO toggle, live power meter, zone grid
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
| `P` | Toggle all projectors |
| `M` | Switch greedy ↔ PPO solver |

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

### Optimizer: Three-Tier Architecture

- **Greedy** (O(n log n)): Real-time, runs every 30 seconds. Two-phase approach (fans first, then lights) to prevent lights from stealing coverage from fans.
- **ILP** (exact): Branch and Bound via PuLP CBC. Provably optimal but slower. Used for verification.
- **GA** (evolutionary): Tournament selection + uniform crossover + bit-flip mutation. 50 individuals over 80 generations. Multi-objective fitness balancing coverage and power.

### PPO Reward Engineering

The reward function uses a **5-term design** with a hard danger penalty. Early
versions used only 4 terms (power alpha=0.3, comfort beta=0.5, switching gamma=0.1, air delta=0.1),
which caused the agent to prefer lights over fans because:
1. Power penalty was too dominant (fans cost more watts)
2. Temperature comfort penalty was too gentle (`exp(-0.1*dev^2)`)
3. No punishment for unsafe conditions

The fix: reduced power weight, steeper comfort curves (`exp(-0.3*dev^2)`), and
a new danger term (epsilon=0.5) for zones above 35C or 1200 ppm CO2.

**Empty room fix:** Hard early-return giving -0.5 per active appliance when nobody is present, and +0.5 for correctly turning everything off. Also doubles the energy penalty for rooms below 30% occupancy.

### Unity: Code-Generated Room

The entire 3D room (floor, walls, ceiling, fans, lights, heatmap, power HUD) is **generated
from the JSON config at runtime**, no manual scene setup. Change the JSON config
and the room rebuilds automatically.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Physics | Python (SciPy RK45), C# (RK4 ODE solver) |
| Optimization | PuLP ILP (CBC), GA, Greedy 3-phase |
| Machine Learning | PyTorch, Stable-Baselines3 PPO, Gymnasium |
| Anomaly Detection | River (HalfSpaceTrees, streaming ML) |
| Backend | FastAPI, Uvicorn |
| Real-Time Comms | Paho MQTT (6 topics, pub/sub) |
| 3D Simulation | Unity 2022.3+ (URP), C# |
| Dashboard | Single-page HTML (REST + MQTT status) |
| Vision (planned) | YOLOv8, OpenCV ArUco |
| Hardware (planned) | Raspberry Pi 4, LEDs via MQTT |

---

## Subject Mappings (Academic)

| Subject | SRACE Component |
|---------|----------------|
| **Computer Networks** | REST API, Unity to Python polling, JSON protocol |
| **Data Analysis & Algorithms** | Greedy O(n log n), ILP solver, GA evolutionary, coverage analysis |
| **Discrete Mathematics** | Weighted Set Cover (NP-hard), ILP binary variables, bipartite coverage graph |
| **AI and ML** | PPO RL agent, custom Gymnasium env, multi-objective reward shaping |

---

## Team

| Member | Owns |
|--------|------|
|  | Physics engine, ILP + Greedy + GA optimizers, PPO RL agent, reward engineering, FastAPI backend, Unity integration, live dashboard |
|  | ArUco detector, YOLOv8 integration, zone mapper, GPIO controller |
|  | FastAPI gateway extensions, WebSocket, InfluxDB, Redis, Docker |
|  | React dashboard, Grafana, UE5 assets |

---

## What Makes This Project Stand Out

1. **Generalised** - any room via JSON config, not a one-off hack
2. **Multi-domain** - CV + RL + combinatorics + physics + IoT + distributed systems
3. **Mathematically rigorous** - ILP gives provably optimal solution
4. **Physically accurate** - ODEs and PDEs, not just thresholds
5. **Triple optimizer** - Greedy for real-time + ILP for verification + GA for multi-objective
6. **RL with reward engineering** - 5-term reward with danger penalties and empty room fix
7. **Full-stack** - Python physics > REST API > Unity 3D > Browser dashboard
8. **Cross-room generalisation** - trained on classroom, tested on office and auditorium
