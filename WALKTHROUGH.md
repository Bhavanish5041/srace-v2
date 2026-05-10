# SRACE v2 — Complete Project Walkthrough & Mentor Briefing

## What We've Built (Completed Work)

### Week 1 — Python Physics & Optimization Core ✅

Built the entire computational backend from scratch:

| Module | What It Does | Key Formula |
|--------|-------------|-------------|
| [airflow.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/physics/airflow.py) | Gaussian decay airflow model | `v = v_peak · exp(−d²/(2σ²))` |
| [thermal.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/physics/thermal.py) | Thermal ODE solved with SciPy RK45 | `dT/dt = −α·v·(T−T_target) + Q_occ/(ρ·cp·V)` |
| [co2_model.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/physics/co2_model.py) | CO₂ mass-balance ODE | `dC/dt = (n·G)/V − λ·(C−C_amb)` |
| [lighting.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/physics/lighting.py) | Cosine-law ceiling fixture model | `E = (Φ·cos³θ)/(2π·h²)` |
| [coverage.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/core/coverage.py) | Binary coverage matrix builder | `C[appliance][zone] = 1 if covers` |
| [greedy_solver.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/optimizer/greedy_solver.py) | Greedy weighted set cover | O(n log n), two-phase (fans then lights) |
| [ilp_solver.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/optimizer/ilp_solver.py) | Exact ILP via PuLP CBC | `min Σ wⱼ·xⱼ s.t. coverage ≥ 1` |
| [room_config.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/core/room_config.py) | Generalised JSON config loader | Any room layout → dataclasses |
| [main.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/main.py) | Full pipeline orchestrator | 5 test scenarios end-to-end |

**Key point for mentor:** The system is *generalised* — change the JSON config and it works for any room (classroom, office, auditorium). Not a hardcoded hack.

---

### Week 2 — Unity 3D Simulation ✅

Ported the physics to C# and built a full 3D visualisation:

| Script | What It Does |
|--------|-------------|
| [SRACEManager.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Core/SRACEManager.cs) | Main orchestrator — loads config, builds room, runs physics + greedy, updates visuals |
| [RoomBuilder.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Environment/RoomBuilder.cs) | Code-generates floor, walls, ceiling from JSON |
| [FanObject.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Environment/FanObject.cs) | Ceiling fan with 3 spinning blades, spin-up/down, color states |
| [LightObject.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Environment/LightObject.cs) | Ceiling light with Unity Light, emission glow, halo |
| [ZoneHeatmap.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Environment/ZoneHeatmap.cs) | Color-coded floor overlay (red/green/grey) |
| [AirflowModel.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Physics/AirflowModel.cs) | C# port of Gaussian airflow |
| [LightingModel.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Physics/LightingModel.cs) | C# port of cosine-law lux |
| [SRACEApiClient.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Core/SRACEApiClient.cs) | Unity ↔ Python REST bridge (polls every 5s) |

**Key point for mentor:** The entire 3D room is code-generated from JSON — no manual scene setup. Press 1-5 for occupancy presets and watch fans spin + lights glow in real time.

---

### Week 3 — PPO Reinforcement Learning + Backend ✅

| Component | File | Status |
|-----------|------|--------|
| Gymnasium environment | [gym_env.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/ml/gym_env.py) | ✅ Custom env with simplified physics |
| Reward function | [reward.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/ml/reward.py) | ✅ 5-term reward (fixed fan bias bug) |
| PPO training | [train_ppo.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/ml/train_ppo.py) | ✅ SB3, EvalCallback, resume support |
| PPO evaluation | [test_ppo.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/ml/test_ppo.py) | ✅ Scenarios + baseline comparison |
| FastAPI backend | [api.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/backend/api.py) | ✅ /room_state, /set_occupancy, /config |
| Trained model | `models/srace_ppo.zip` | ✅ 500K timesteps |

---

## PPO Results (Latest, 500K Steps)

### Performance by Scenario

| Scenario | People | Fans | Temp | CO₂ | Power | Reward |
|----------|--------|------|------|-----|-------|--------|
| **Full** (12/12 zones) | 60 | 9-10 ✅ | 28.9°C | 1031 ppm | 795W | +10.2 |
| **Sparse** (3/12 zones) | 7-9 | 6-7 ✅ | 26.2°C | 508 ppm | 610W | +36-42 |
| **Random** (11/12 zones) | 47 | 7-8 ✅ | 27-29°C | 710-1080 | 600-760W | -1.8 to +5.8 |
| **Empty** (0/12 zones) | 0 | 9 ❌ | 25.5°C | 400 ppm | 599W | -17.3 |

### What the Agent Learned

- ✅ **Activates fans for cooling** — before the bug fix: 0 fans, now: 6-10 fans
- ✅ **Temperature control** — cools from 30°C ambient toward 25-29°C
- ✅ **CO₂ management** — keeps below 1000 ppm in sparse scenarios
- ✅ **Scales with occupancy** — sparse uses 610W vs full at 795W
- ⚠️ **Empty room** — hasn't learned to shut down (needs more training)
- ⚠️ **Underuses lights** — prioritizes fans due to danger penalty

---

## Bugs Found & Fixed

### Bug 1: PPO Agent Never Used Fans

**Problem:** The trained model activated only lights, never any fans. Temperature hit 45°C, CO₂ hit 2000 ppm.

**Root cause:** The reward function's power penalty (α=0.3) was too dominant. Fans cost 75W vs lights at 40W, so the agent was punished more for using fans. Meanwhile, the temperature comfort penalty used `exp(-0.1·dev²)` which was too gentle — a 20°C deviation barely mattered.

**Fix in [reward.py](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/ml/reward.py):**
- Reduced power weight: α 0.3 → 0.15
- Increased comfort weight: β 0.5 → 0.55
- Steeper temp curve: `exp(-0.3·dev²)` instead of `exp(-0.1·dev²)`
- Added 5th term: danger penalty (ε=0.5) for zones >35°C or >1200 ppm
- Reweighted comfort internals: temp 40% + CO₂ 30% + coverage 15% + lux 15%

**Result:** Fans now activate properly. Lights-only went from +42.5 reward to **−72.3** reward.

### Bug 2: Unity Greedy Never Selected Fans

**Problem:** In Unity's local mode, the greedy optimizer selected only lights, no fans.

**Root cause (two issues):**
1. `GreedyActivate()` ran fans and lights **together** in one pass. Lights at 40W always had better coverage-per-watt than fans at 75W, so lights were always picked first.
2. `thermalImpact` was a **placeholder zero array**, so fans couldn't claim coverage via the thermal check.

**Fix in [SRACEManager.cs](file:///home/bhavanish/.gemini/antigravity/scratch/srace-v2/Assets/SRACE/Scripts/Core/SRACEManager.cs):**
- Split greedy into two phases: fans first, then lights (mirrors Python solver)
- Added analytical thermal estimate: `convCoeff × airflow × tempDelta × forecastTime`

---

## What to Explain to Your Mentor

### 1. The Core Idea (30 seconds)

> "Buildings waste 40-70% electricity running all fans and lights regardless of occupancy. SRACE uses physics-based models to compute which appliances actually affect which zones, then uses mathematical optimization to find the minimum-power subset that covers all occupied zones."

### 2. Physics Pipeline (2 minutes)

Show `python3 main.py` output. Explain:
- **4 physics models** compute real values (airflow m/s, temperature °C, CO₂ ppm, illuminance lux)
- These produce a **coverage matrix** `C[appliance][zone]` — does this appliance meaningfully affect this zone?
- The **ILP optimizer** solves the Weighted Set Cover problem (NP-hard) exactly using Branch & Bound
- The **greedy** gives O(ln n) approximation, good enough for real-time (every 30 seconds)

### 3. PPO Reinforcement Learning (3 minutes)

Show `python3 ml/test_ppo.py --scenario sparse --render`. Explain:
- The agent observes: zone occupancy + temperatures + CO₂ + lux + current appliance states
- It decides: which of 20 appliances (10 fans + 10 lights) to turn on/off
- **Reward = −power + comfort − switching + air_quality − danger**
- Unlike greedy/ILP (static snapshot), the RL agent learns **temporal dynamics** — it knows turning fans on NOW prevents overheating later

**The bug fix story:** Explain the reward engineering problem. The original reward made fans unprofitable because power penalty dominated comfort. This is a real ML engineering lesson — reward shaping is critical.

### 4. Unity Demo (2 minutes)

Open Unity, press Play:
- Press `4` (full room) → watch fans spin, lights glow, green heatmap
- Press `1` (empty) → everything shuts off
- Press `3` (center cluster) → only nearby appliances activate
- Press `F` to toggle fans, `L` to toggle lights manually

### 5. API Integration (1 minute)

```bash
source venv/bin/activate
uvicorn backend.api:app --port 8000
```
Open `http://localhost:8000/docs` — Swagger UI. Show:
- `POST /set_occupancy` → set zone people
- `GET /room_state` → physics + optimizer result with active appliances and power savings

---

## Remaining Work

### High Priority (Needed for Final Demo)

| Task | Owner | Effort | Status |
|------|-------|--------|--------|
| Train PPO longer (2M+ steps) | Bhavanish | 1-2 hours compute | Not started |
| Fix empty room behavior (curriculum learning or reward tweak) | Bhavanish | 2 hours | Not started |
| Connect PPO model to FastAPI (add `/ppo_action` endpoint) | Bhavanish | 3 hours | Not started |
| YOLOv8 crowd detection | Person 2 | In progress | - |
| React dashboard | Person 4 | In progress | - |
| RPi LED hardware | Person 2 | In progress | - |

### Medium Priority (Polish)

| Task | Description | Effort |
|------|-------------|--------|
| PPO vs Greedy vs ILP comparison endpoint | API returns all 3 solutions side-by-side | 3 hours |
| Unity HUD overlay | Show power, savings %, active count on screen | 2 hours |
| Tensorboard logging | Uncomment TB callback in train_ppo.py, add training curves | 1 hour |
| More room configs | Create office.json, auditorium.json to show generalisation | 1 hour |
| PPO light usage improvement | Tune reward to better balance fans vs lights | 2 hours |

### Low Priority (Nice-to-Have)

| Task | Description |
|------|-------------|
| MQTT integration | Pub/sub for RPi ↔ backend communication |
| InfluxDB logging | Store historical room state for Grafana |
| Genetic algorithm | Layout optimization (fan/light placement) |
| River anomaly detection | Online ML for unusual occupancy patterns |

---

## How to Demo (Step by Step)

### Before the Meeting
```bash
cd ~/.gemini/antigravity/scratch/srace-v2
source venv/bin/activate

# Optional: retrain with more steps for better results
python3 ml/train_ppo.py --timesteps 2000000
```

### Demo Sequence

**Step 1: Python pipeline** (terminal)
```bash
python3 main.py
```
→ Shows 5 scenarios, Greedy vs ILP, power savings

**Step 2: PPO evaluation** (terminal)
```bash
python3 ml/test_ppo.py --scenario sparse --render
python3 ml/test_ppo.py --scenario full --render
```
→ Shows agent decisions, temperature/CO₂ control, baseline comparison

**Step 3: Unity simulation** (Unity Editor)
- Press Play → Press `4` (full room) → Press `1` (empty) → Press `3` (cluster)
- Show fans spinning, lights glowing, heatmap colors changing

**Step 4: API** (browser)
```bash
uvicorn backend.api:app --port 8000
# Open http://localhost:8000/docs
```
→ Live POST/GET with Swagger

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Room size | 10.8 × 7.6m, 12 zones (4×3 grid) |
| Appliances | 10 fans (75W each) + 10 lights (40W each) = 1150W max |
| Greedy savings | ~60-80% power saved vs all-on |
| ILP savings | Provably optimal (≥ greedy) |
| PPO training | 500K steps, 4 parallel envs |
| PPO sparse result | 26.2°C, 508 ppm CO₂, 610W (47% savings) |
| Physics models | 4 (airflow, thermal, CO₂, lighting) |
| Comfort targets | 25°C, 300 lux, <1000 ppm CO₂ |

---

## Subject Connections (If Mentor Asks)

| Subject | What to Say |
|---------|-------------|
| **Computer Networks** | "We use REST API polling — Unity sends HTTP GET every 5 seconds to Python backend. MQTT is planned for RPi integration." |
| **Data Analysis & Algorithms** | "The greedy is O(n log n) approximation for Set Cover. ILP gives exact optimal via Branch and Bound. We compared both in our pipeline." |
| **Discrete Mathematics** | "The core problem is Weighted Set Cover — NP-hard. We model it as an ILP with binary decision variables xⱼ ∈ {0,1} and solve via CBC." |
| **AI and ML** | "PPO is a policy gradient RL algorithm. Our agent learns through 500K simulated interactions. The reward function has 5 terms balancing energy vs comfort vs safety." |
