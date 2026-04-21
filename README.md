# SRACE v2 — Smart Room Automation & Control Engine

> Generalised intelligent room automation platform that works for any room —
> classroom, office, staff room, auditorium — controlled via a JSON config file.
> Not a fixed installation but a deployable system.

---

## The Problem

Institutional buildings waste **40–70% electricity** running all fans and lights
regardless of how many people are present or where they sit. Existing systems use
dumb binary thresholds — everything on or everything off.

SRACE fixes this with **real intelligence**: physics-based models, mathematical
optimization, and reinforcement learning to run only the appliances that matter.

---

## System Architecture

```
S23 Ultra / UE5 NavMesh Agents
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
  → Genetic Algorithm (layout changes)
        ↓
PPO RL Agent (Stable-Baselines3)
  → State: crowd + temps + CO₂ + lux + appliance states
  → Action: which fans/lights to toggle
  → Reward: −energy + comfort − switching + air quality
        ↓
MQTT Broker (Mosquitto)
  → Publishes control commands
        ↓
Three simultaneous outputs:
  RPi GPIO → LED panel (physical)
  UE5 simulation → fans spin, lights glow, heatmap
  React dashboard → live data, power gauge, RL log
```

---

## Key Design Decisions

### Crowd Detection
| Considered | Decision | Reason |
|-----------|----------|--------|
| IR beam sensors | ❌ Dropped | Too primitive, no spatial info |
| PIR sensors per zone | ❌ Dropped | Can't count, only detect presence |
| LiDAR | ❌ Dropped | Too expensive |
| **Samsung S23 Ultra + YOLOv8** | ✅ Production | Real-time counting with spatial zones |
| ArUco markers on paper figurines | ✅ Physical demo | Cheap, reliable for static demo |
| Unity NavMesh agents | → UE5 NavMesh | Simulation crowd |

### Hardware
| Considered | Decision | Reason |
|-----------|----------|--------|
| ESP32 | → RPi Zero 2W | More compute for edge ML |
| RPi Zero 2W per zone | → Single RPi 4 | Simplified demo |
| Real fans/lights | ❌ Dropped | Too risky, college property |
| Relay module (220V) | ❌ Dropped | Safety concern |
| **20 LEDs on breadboard** | ✅ Final | Safe, cheap (₹180), clear visual |

### ML Approach
| Considered | Decision | Reason |
|-----------|----------|--------|
| LSTM for crowd prediction | ❌ Dropped | Redundant with real-time YOLO |
| **PPO reinforcement learning** | ✅ | Learns optimal policy through simulation |
| **River online ML** | ✅ | Anomaly detection, no batch retraining |
| **ILP optimizer** | ✅ | Mathematical core, provably optimal |

### Visualization
| Evolution | Final |
|-----------|-------|
| React dashboard only → Pygame 2D → Unity 3D → **Unreal Engine 5** | UE5 for Lumen lighting, Nanite, NavMesh, existing experience |

---

## Mathematics & Physics

| Formula | Used For | Module |
|---------|----------|--------|
| `v(f,z) = v_peak · exp(−d²/(2σ²))`, σ = radius/2 | Airflow per zone from each fan | `physics/airflow.py` |
| `dT/dt = −α·v·(T−T_target) + Q_occ/(ρ·cp·V) + k·(T_amb−T)` | Temperature forecast 5 min ahead | `physics/thermal.py` |
| `dC/dt = (n·G)/(V) − λ_vent·(C−C_ambient)` | CO₂ per zone over time | `physics/co2_model.py` |
| `E = (Φ·cos³θ)/(2π·h²)` | Lux per zone from each ceiling light | `physics/lighting.py` |
| `min Σ w_j·x_j  s.t. coverage ≥ 1 ∀ occupied zones` | Minimum power appliance subset | `optimizer/ilp_solver.py` |
| `R = −α·P + β·comfort − γ·switches + δ·air_quality` | PPO reward function | `ml/gym_env.py` (Week 2+) |

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

## Subject Mappings (Academic)

| Subject | SRACE Component |
|---------|----------------|
| **Computer Networks** | MQTT QoS1, WebSocket, topic tree, RPi edge node |
| **Data Analysis & Algorithms** | InfluxDB time-series, Greedy O(n log n), solver comparison, River streaming ML |
| **Discrete Mathematics** | Weighted Set Cover NP-hard, ILP binary variables, bipartite coverage graph, GA chromosome encoding |
| **AI and ML** | PPO RL agent, YOLO CV, zero-shot generalisation, multi-objective reward |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision | YOLOv8, OpenCV ArUco, Samsung S23 Ultra |
| Physics | NumPy, SciPy (`solve_ivp` RK45) |
| Optimization | PuLP ILP (CBC), DEAP Genetic Algorithm |
| ML | PyTorch, Stable-Baselines3 PPO, River |
| Backend | FastAPI, Mosquitto MQTT, WebSocket |
| Database | InfluxDB 2.0, Redis 7 |
| Hardware | Raspberry Pi 4, 20 LEDs, breadboard |
| Simulation | **Unreal Engine 5** (Lumen, Nanite, NavMesh) |
| Dashboard | React, Recharts, Tailwind, Grafana |

---

## Repository Structure

### Week 1 — Python Core (DONE)

```
srace-v2/
├── config/
│   └── default_room.json          # Room config: 10×8m classroom, 10 fans, 10 lights
├── core/
│   ├── room_config.py             # RoomConfig, Zone, Fan, Light, ComfortParams dataclasses
│   └── coverage.py                # Binary coverage matrix from physics outputs
├── physics/
│   ├── airflow.py                 # Gaussian decay fan airflow model
│   ├── thermal.py                 # Thermal ODE via Method of Lines + RK45
│   ├── co2_model.py               # CO₂ mass-balance ODE
│   └── lighting.py                # Cosine-law illuminance model
├── optimizer/
│   ├── greedy_solver.py           # Greedy weighted set cover (O(n log n))
│   └── ilp_solver.py              # Exact ILP via PuLP CBC
├── visualization/
│   └── room_grid.py               # Matplotlib room grid visualization
├── output/                        # Generated visualizations
├── main.py                        # Pipeline: config → physics → coverage → optimizer → output
└── requirements.txt               # numpy, scipy, pulp, matplotlib
```

### Planned Structure (Full System)

```
srace-v2/
├── (above) ...
├── ml/
│   ├── gym_env.py                 # Custom Gymnasium environment
│   ├── train_ppo.py               # PPO training script
│   └── reward.py                  # Multi-objective reward function
├── detection/
│   ├── yolo_detector.py           # YOLOv8 real-time person detection
│   ├── aruco_detector.py          # ArUco marker detection for demo
│   └── zone_mapper.py             # Map detections to zone grid
├── backend/
│   ├── api.py                     # FastAPI gateway
│   ├── mqtt_client.py             # MQTT publisher/subscriber
│   └── ws_server.py               # WebSocket for dashboard
├── hardware/
│   ├── gpio_controller.py         # RPi GPIO LED control
│   └── led_mapper.py              # Map appliance states to LED pins
├── ue5/                           # Unreal Engine 5 project (separate repo)
└── dashboard/                     # React dashboard (separate repo)
```

---

## Team Split

| Member | Owns |
|--------|------|
| **Bhavanish** | PPO agent, ILP + Greedy + GA optimizer, River anomaly detection, physics engine integration, MQTT backbone |
| Person 2 | ArUco detector, YOLOv8 integration, zone mapper, GPIO controller, sensor scripts |
| Person 3 | FastAPI gateway, WebSocket, InfluxDB, Redis, Docker |
| Person 4 | React dashboard, UE5 simulation, Grafana |

---

## Demo Setup

```
Demo table layout:
  ┌─────────────────────────────────────────────────────────┐
  │  Laptop 1          Laptop 2        RPi 4    S23 Ultra   │
  │  (UE5 sim)         (Dashboard)     (LEDs)   (overhead)  │
  │                                                         │
  │                    ┌──────────┐                         │
  │                    │ Cardboard│                         │
  │                    │ Mini Room│                         │
  │                    │ (ArUco   │                         │
  │                    │ figurines│                         │
  │                    └──────────┘                         │
  └─────────────────────────────────────────────────────────┘
```

### Demo Flow
1. **Empty room** — all LEDs off, UE5 dark, power = 0W
2. **Place figurines** in zones → ArUco detects → optimizer runs
3. **Specific LEDs light up** → UE5 fans spin, lights glow
4. **Move figurines** → different LEDs respond in real time
5. **Fill all zones** → power rises → all LEDs on
6. **Remove everyone** → LEDs off → savings % displayed

---

## Week 1 Plan

### Python Track (Bhavanish's actual work)
- [x] Day 1: Project setup, folder structure, Git repo
- [x] Day 2: RoomConfig JSON loader
- [x] Day 3: Physics engine (all 4 models)
- [x] Day 4: Set Cover optimizer (Greedy + ILP)
- [x] Day 5: Wire everything in main.py, print results
- [x] Day 6: Matplotlib room grid visualization

### UE5 Track (impress mentor)
- [ ] Day 1–2: Room scene from Fab.com assets
- [ ] Day 3: Fan rotation + Lumen lighting
- [ ] Day 4: Blueprint logic, zone heatmap
- [ ] Day 5: SRACE HUD widget
- [ ] Day 6: Fake demo sequence Blueprint
- [ ] Day 7: Record cinematic with Sequencer

### Show Mentor Strategy
1. UE5 demo video first (wow factor)
2. Python terminal output second (technical depth)
3. GitHub repo with clean commits

---

## What Makes This Project Stand Out

1. **Generalised** — any room via JSON config, not a one-off hack
2. **Multi-domain** — CV + RL + combinatorics + physics + IoT + distributed systems
3. **Mathematically rigorous** — ILP gives provably optimal solution
4. **Physically accurate** — PDEs and ODEs, not just thresholds
5. **Production grade** — MQTT, InfluxDB, Redis, WebSocket, Docker
6. **Visually stunning** — UE5 Lumen photorealistic simulation
7. **Publication worthy** — zero-shot PPO generalisation across room types is a genuine research contribution

---

## Key Concepts

- **PPO** — RL algorithm that learns optimal appliance policy through 2M simulated room interactions, using clipped policy updates for stable training
- **ILP** — Mathematical optimization that finds exact minimum power appliance combination using Branch and Bound, solved in milliseconds via PuLP CBC
- **Set Cover** — NP-hard combinatorial problem: minimum cost subset of appliances covering all occupied zones
- **River** — Online streaming ML, learns anomaly patterns live without batch retraining
- **MQTT** — Lightweight IoT pub/sub protocol, QoS1 guaranteed delivery, hierarchical topic tree
- **Lumen** — UE5 real-time global illumination, photorealistic lighting with zero shader code

---

## Running

```bash
# Install dependencies
pip install numpy scipy pulp matplotlib

# Run full pipeline with all test scenarios
python main.py

# Output: terminal results + visualizations in ./output/
```

### Test Scenarios (built-in)
| Scenario | Occupancy |
|----------|-----------|
| Empty Room | No one |
| Single Person (Zone 0) | 1 person in zone 0 |
| Cluster (Center) | 11 people in zones 5,6,9,10 |
| Full Room | 48 people, 4 per zone |
| Front Row Only | 20 people in zones 0–3 |
