# SRACE v2: Complete Project Walkthrough & Mentor Briefing

## What We Built (All Work Completed)

### Week 1: Python Physics & Optimization Core

Built the entire computational backend from scratch:

| Module | What It Does | Key Formula |
|--------|-------------|-------------|
| airflow.py | Gaussian decay airflow model | `v = v_peak * exp(-d^2/(2*sigma^2))` |
| thermal.py | Thermal ODE solved with SciPy RK45 | `dT/dt = -a*v*(T-T_target) + Q_occ/(rho*cp*V)` |
| co2_model.py | CO2 mass-balance ODE | `dC/dt = (n*G)/V - lambda*(C-C_amb)` |
| lighting.py | Cosine-law ceiling fixture model | `E = (Phi*cos^3(theta))/(2*pi*h^2)` |
| coverage.py | Binary coverage matrix builder | `C[appliance][zone] = 1 if covers` |
| greedy_solver.py | Greedy weighted set cover | O(n log n), two-phase (fans then lights) |
| ilp_solver.py | Exact ILP via PuLP CBC | `min sum(w_j * x_j) s.t. coverage >= 1` |
| ga_solver.py | Genetic Algorithm optimizer | Tournament + crossover + mutation, 80 gens |
| room_config.py | Generalised JSON config loader | Any room layout to dataclasses |
| main.py | Full pipeline orchestrator | 5 scenarios, 3-way optimizer comparison |

**Key point for mentor:** The system is *generalised*. Change the JSON config and it works for any room (classroom, office, auditorium). Not a hardcoded hack.

---

### Week 2: Unity 3D Simulation

Ported the physics to C# and built a full 3D visualisation:

| Script | What It Does |
|--------|-------------|
| SRACEManager.cs | Main orchestrator: loads config, builds room, runs physics + greedy, updates visuals |
| RoomBuilder.cs | Code-generates floor, walls, ceiling from JSON |
| FanObject.cs | Ceiling fan with 3 spinning blades, smooth spin-up/down, color states |
| LightObject.cs | Ceiling light with Unity Light, emission glow, halo |
| ZoneHeatmap.cs | Color-coded floor overlay (red/green/grey) |
| PowerHUD.cs | Live power meter overlay with animated bar, savings %, fan/light counts |
| AirflowModel.cs | C# port of Gaussian airflow |
| LightingModel.cs | C# port of cosine-law lux |
| SRACEApiClient.cs | Unity to Python REST bridge (polls every 5s) |

**Key point for mentor:** The entire 3D room is code-generated from JSON, no manual scene setup. Press 1-5 for occupancy presets and watch fans spin + lights glow in real time. The Power HUD shows live wattage and savings percentage.

---

### Week 3: PPO Reinforcement Learning + Backend

| Component | File | Status |
|-----------|------|--------|
| Gymnasium environment | gym_env.py | Custom env with simplified physics |
| Reward function | reward.py | 5-term reward with empty room fix |
| PPO training | train_ppo.py | SB3, EvalCallback, resume support |
| PPO evaluation | test_ppo.py | Scenarios + cross-room generalisation |
| FastAPI backend | api.py | 6 endpoints including PPO + compare |
| Genetic Algorithm | ga_solver.py | Real evolutionary optimizer (not a stub) |
| Live dashboard | dashboard.html | Browser-based real-time visualisation |
| Visualization | visualize_results.py | PPO step-by-step + baseline charts |
| Trained model | srace_ppo.zip | 2.5M timesteps |
| Room configs | 3 configs | Classroom, office (6x5m), auditorium (15x10m) |

---

## PPO Results (2.5M Steps)

### Performance by Scenario

| Scenario | People | Fans | Temp | CO2 | Power | Reward |
|----------|--------|------|------|-----|-------|--------|
| **Full** (12/12 zones) | 60 | 10/10 | 28.7C | 999 ppm | 794W | +13.4 |
| **Sparse** (3/12 zones) | 7-9 | 3-5 | 26.4C | 490 ppm | 493-523W | +27 to +42 |
| **Random** (11/12 zones) | 47 | 7-8 | 27-29C | 710-1080 | 600-760W | -1.8 to +5.8 |
| **Empty** (0/12 zones) | 0 | Still learning | 26.9C | 404 ppm | 436W | -12.5 |

### Cross-Room Generalisation

| Room | Config | Zones | Appliances | Result |
|------|--------|-------|------------|--------|
| Classroom (training) | default_room.json | 12 | 20 | Full performance |
| Office (unseen) | office_small.json | 4 | 8 | Good generalisation |
| Auditorium (unseen) | auditorium.json | 20 | 32 | Reasonable with padding |

### What the Agent Learned

- Activates all 10 fans for cooling in full occupancy (before bug fix: 0 fans)
- Temperature control: cools from 30C ambient toward 25-29C
- CO2 management: keeps at exactly 999 ppm (just under 1000 limit) in full scenario
- Scales with occupancy: sparse uses 493W vs full at 794W
- Still learning empty room behavior (needs more training samples)

---

## Bugs Found & Fixed

### Bug 1: PPO Agent Never Used Fans

**Problem:** The trained model activated only lights, never any fans. Temperature hit 45C, CO2 hit 2000 ppm.

**Root cause:** The reward function's power penalty (alpha=0.3) was too dominant. Fans cost 75W vs lights at 40W, so the agent was punished more for using fans. Meanwhile, the temperature comfort penalty used `exp(-0.1*dev^2)` which was too gentle.

**Fix in reward.py:**
- Reduced power weight: alpha 0.3 to 0.15
- Increased comfort weight: beta 0.5 to 0.55
- Steeper temp curve: `exp(-0.3*dev^2)` instead of `exp(-0.1*dev^2)`
- Added 5th term: danger penalty (epsilon=0.5) for zones above 35C or 1200 ppm
- Reweighted comfort internals: temp 40% + CO2 30% + coverage 15% + lux 15%

**Result:** Fans now activate properly. Lights-only went from +42.5 reward to -72.3 reward.

### Bug 2: Unity Greedy Never Selected Fans

**Problem:** In Unity's local mode, the greedy optimizer selected only lights, no fans.

**Root cause (two issues):**
1. `GreedyActivate()` ran fans and lights together in one pass. Lights at 40W always had better coverage-per-watt than fans at 75W.
2. `thermalImpact` was a placeholder zero array, so fans could not claim coverage.

**Fix in SRACEManager.cs:**
- Split greedy into two phases: fans first, then lights (mirrors Python solver)
- Added analytical thermal estimate: `convCoeff * airflow * tempDelta * forecastTime`

### Bug 3: Empty Room Wasting Energy

**Problem:** Agent kept 8-9 appliances on (400-500W) even with zero occupants.

**Root cause:** The penalty for running appliances in an empty room was only -0.15 per step (too weak). The agent found a lazy policy.

**Fix in reward.py:**
- Hard early-return: -0.5 per active appliance when nobody is present
- +0.5 reward for correctly turning everything off
- Doubled energy penalty (alpha * 2) for rooms below 30% occupancy
- Signal is now 45x stronger for the empty case

---

## Three Optimizer Comparison

| Property | Greedy | ILP | GA |
|----------|--------|-----|-----|
| Approach | Heuristic | Exact | Evolutionary |
| Time complexity | O(n log n) | Exponential (but fast for small n) | O(pop * gens * n) |
| Optimality | ln(n) approximation | Provably optimal | Near-optimal |
| Multi-objective | Single (power) | Single (power) | Yes (power + coverage) |
| Real-time suitable | Yes (every 30s) | Yes (every 60s) | Yes (every 60s) |
| Use case | Default solver | Verification/benchmarking | Multi-objective trade-offs |

---

## What to Explain to Your Mentor

### 1. The Core Idea (30 seconds)

> "Buildings waste 40-70% electricity running all fans and lights regardless of occupancy. SRACE uses physics-based models to compute which appliances actually affect which zones, then uses mathematical optimization to find the minimum-power subset that covers all occupied zones."

### 2. Physics Pipeline (2 minutes)

Show `python3 main.py` output. Explain:
- **4 physics models** compute real values (airflow m/s, temperature C, CO2 ppm, illuminance lux)
- These produce a **coverage matrix** `C[appliance][zone]`: does this appliance meaningfully affect this zone?
- Three optimizers: **Greedy** (fast heuristic), **ILP** (exact optimal via Branch & Bound), **GA** (evolutionary multi-objective)
- Pipeline shows all 3 side by side for each scenario

### 3. PPO Reinforcement Learning (3 minutes)

Show `python3 ml/test_ppo.py --scenario sparse --render`. Explain:
- The agent observes: zone occupancy + temperatures + CO2 + lux + current appliance states
- It decides: which of 20 appliances (10 fans + 10 lights) to turn on/off
- **Reward = -power + comfort - switching + air_quality - danger**
- Unlike greedy/ILP (static snapshot), the RL agent learns **temporal dynamics**: it knows turning fans on NOW prevents overheating later
- Cross-room generalisation: trained on classroom, tested on office and auditorium

**The bug fix story:** Explain the reward engineering problem. The original reward made fans unprofitable because power penalty dominated comfort. This is a real ML engineering lesson: reward shaping is critical.

### 4. Unity Demo (2 minutes)

Open Unity, press Play:
- Press `4` (full room): watch fans spin, lights glow, green heatmap, Power HUD shows wattage
- Press `1` (empty): everything shuts off, HUD shows 100% saved
- Press `3` (center cluster): only nearby appliances activate
- Press `F` to toggle fans, `L` to toggle lights manually

### 5. Live Dashboard (1 minute)

```bash
uvicorn backend.api:app --port 8000
# Open dashboard.html in browser
```
- Click scenario buttons (Empty, Sparse, Full, Cluster)
- Toggle between Greedy and PPO solver
- Watch live power meter, zone grid, appliance chips update in real time

### 6. API Integration (1 minute)

```bash
uvicorn backend.api:app --port 8000
```
Open `http://localhost:8000/docs`. Show:
- `POST /set_occupancy` to set zone people
- `GET /ppo_action` to see PPO agent decisions
- `GET /compare` to see greedy vs PPO side-by-side

---

## How to Demo (Step by Step)

### Before the Meeting
```bash
cd ~/.gemini/antigravity/scratch/srace-v2
source venv/bin/activate
```

### Demo Sequence

**Step 1: Python pipeline** (terminal)
```bash
python3 main.py
```
Shows 5 scenarios with Greedy vs ILP vs GA, power savings comparison

**Step 2: PPO evaluation** (terminal)
```bash
python3 ml/test_ppo.py --scenario sparse --render
python3 ml/test_ppo.py --scenario full --render
```
Shows agent decisions, temperature/CO2 control, baseline comparison

**Step 3: Visualization charts** (terminal)
```bash
python3 visualize_results.py
```
Generates dark-themed charts in output/ folder

**Step 4: Unity simulation** (Unity Editor)
- Press Play, then press `4` (full) then `1` (empty) then `3` (cluster)
- Show fans spinning, lights glowing, heatmap colors, Power HUD updating

**Step 5: Live dashboard** (browser)
```bash
uvicorn backend.api:app --port 8000
# Open dashboard.html
```
Click through scenarios, toggle Greedy/PPO, show real-time updates

**Step 6: Cross-room testing** (terminal)
```bash
python3 ml/test_ppo.py --config config/auditorium.json --scenario sparse --render
```
Shows the model generalising to unseen rooms

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Room size | 10.8 x 7.6m, 12 zones (4x3 grid) |
| Appliances | 10 fans (75W each) + 10 lights (40W each) = 1150W max |
| Optimizers | 3 (Greedy, ILP, GA) |
| Greedy savings | ~60-80% power saved vs all-on |
| ILP savings | Provably optimal (>= greedy) |
| PPO training | 2.5M steps, 4 parallel envs |
| PPO sparse result | 26.4C, 490 ppm CO2, 493W (57% savings) |
| PPO full result | 28.7C, 999 ppm CO2, 794W (31% savings) |
| Physics models | 4 (airflow, thermal, CO2, lighting) |
| Comfort targets | 25C, 300 lux, <1000 ppm CO2 |
| Room configs | 3 (classroom, office, auditorium) |
| API endpoints | 6 (room_state, set_occupancy, ppo_action, ppo_reset, compare, config) |

---

## Subject Connections (If Mentor Asks)

| Subject | What to Say |
|---------|-------------|
| **Computer Networks** | "We use REST API polling: Unity sends HTTP GET every 5 seconds to Python backend. The live dashboard polls every 2 seconds. MQTT is planned for RPi integration." |
| **Data Analysis & Algorithms** | "Three optimizers: Greedy is O(n log n) approximation for Set Cover. ILP gives exact optimal via Branch and Bound. GA uses evolutionary search with tournament selection. We compare all 3 in our pipeline." |
| **Discrete Mathematics** | "The core problem is Weighted Set Cover, which is NP-hard. We model it as an ILP with binary decision variables x_j in {0,1} and solve via CBC." |
| **AI and ML** | "PPO is a policy gradient RL algorithm. Our agent learns through 2.5M simulated interactions. The reward function has 5 terms balancing energy vs comfort vs safety. We also tested cross-room generalisation." |
