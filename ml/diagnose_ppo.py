"""
ml/diagnose_ppo.py — SRACE PPO Diagnostics Logger
====================================================
Run this to generate a full diagnostic report you can share.

Usage:
    python ml/diagnose_ppo.py

Output:
    Prints to terminal AND saves to: logs/ppo_diagnosis_<timestamp>.txt

The report captures:
  1. Model architecture & training metadata
  2. Reward breakdown per step (what each term is doing)
  3. Action patterns  (is agent stuck always-on or always-off?)
  4. Physics drift    (temp / CO2 over episode)
  5. Per-scenario summary: random / full / sparse / empty
  6. Baseline comparison: greedy vs PPO vs all-on vs all-off

Share the generated .txt file to get targeted fixes.
"""

import os
import sys
import time
import json
from datetime import datetime
from io import StringIO

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import PPO
from ml.gym_env import SRACEEnv, MAX_STEPS
from ml.reward import calculate_reward

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

LOG_DIR = os.path.join(_project_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"ppo_diagnosis_{timestamp}.txt")

_lines = []  # buffer every line so we can write to file at end

# Global adaptation state — set in main()
_model_obs_size = 0
_model_n_appliances = 0
_env_obs_size = 0
_env_n_appliances = 0


def log(line=""):
    print(line)
    _lines.append(line)


def write_log():
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_lines))
    print(f"\n✅  Report saved to: {LOG_PATH}")
    print(f"    Share this file to get targeted fixes.\n")


def scenario_reset(env, scenario, seed=None):
    obs, info = env.reset(seed=seed)
    if scenario == "full":
        env.zone_people = np.full(env.n_zones, 5.0)
    elif scenario == "sparse":
        env.zone_people = np.zeros(env.n_zones)
        chosen = np.random.choice(env.n_zones, size=min(3, env.n_zones), replace=False)
        for z in chosen:
            env.zone_people[z] = np.random.randint(2, 6)
    elif scenario == "empty":
        env.zone_people = np.zeros(env.n_zones)
    obs = env._get_observation()
    return obs, info


def _predict(model, obs):
    """Predict with obs/action adaptation for model/env size mismatch."""
    # Adapt observation
    if _env_obs_size != _model_obs_size:
        adapted_obs = np.zeros(_model_obs_size, dtype=obs.dtype)
        n_copy = min(_env_obs_size, _model_obs_size)
        adapted_obs[:n_copy] = obs[:n_copy]
    else:
        adapted_obs = obs

    raw_action, info = model.predict(adapted_obs, deterministic=True)

    # Adapt action
    if _model_n_appliances != _env_n_appliances:
        action = np.zeros(_env_n_appliances, dtype=raw_action.dtype)
        n_copy = min(_model_n_appliances, _env_n_appliances)
        action[:n_copy] = raw_action[:n_copy]
    else:
        action = raw_action

    return action


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — Model Info
# ══════════════════════════════════════════════════════════════

def section_model_info(model, model_path, env):
    log("=" * 70)
    log("  SECTION 1: Model & Environment Info")
    log("=" * 70)
    log(f"  Model path    : {model_path}")
    log(f"  Room          : {env.cfg.name}")
    log(f"  Config        : {env.cfg.width}×{env.cfg.depth}m  "
        f"ceil={env.cfg.ceiling_height}m")
    log(f"  Zones         : {env.n_zones}  "
        f"({env.cfg.n_zone_rows}r × {env.cfg.n_zone_cols}c)")
    log(f"  Fans          : {env.n_fans}  "
        f"({sum(f.power_watts for f in env.cfg.fans):.0f}W total)")
    log(f"  Lights        : {env.n_lights}  "
        f"({sum(l.power_watts for l in env.cfg.lights):.0f}W total)")
    log(f"  Projectors    : {env.cfg.n_projectors}")
    max_pow = sum(a.power_watts for a in env.cfg.all_appliances)
    log(f"  Max power     : {max_pow:.0f}W")
    log(f"  Model obs     : {_model_obs_size}  (env obs: {_env_obs_size})")
    log(f"  Model actions : {_model_n_appliances}  (env actions: {_env_n_appliances})")
    if _model_n_appliances != _env_n_appliances:
        log(f"  ⚠  MISMATCH: Model trained for {_model_n_appliances} appliances, "
            f"env has {_env_n_appliances}. Adapting actions.")
    log(f"  Policy        : {type(model.policy).__name__}")

    # Try to extract training info from model
    try:
        log(f"  Gamma         : {model.gamma}")
        log(f"  Learning rate : {model.learning_rate}")
        log(f"  n_steps       : {model.n_steps}")
        log(f"  n_epochs      : {model.n_epochs}")
        log(f"  Clip range    : {model.clip_range}")
        log(f"  Total timesteps (num_timesteps): {model.num_timesteps}")
    except Exception:
        log("  (Could not extract hyperparameters)")

    log(f"  Comfort target: {env.cfg.comfort.target_temp_c}°C  "
        f"{env.cfg.comfort.target_lux} lux  "
        f"<{env.cfg.comfort.max_co2_ppm} ppm CO₂")
    log()


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — Reward Breakdown
# ══════════════════════════════════════════════════════════════

def section_reward_breakdown(model, env, n_episodes=3):
    log("=" * 70)
    log("  SECTION 2: Reward Breakdown (what each term contributes)")
    log("=" * 70)
    log("  Coverage, Energy, Comfort (Temp), Comfort (CO2), Comfort (Lux)")
    log()

    from ml import reward as rmod

    orig_fn = rmod.calculate_reward
    captured_components = []

    def patched_reward(**kwargs):
        r = orig_fn(**kwargs)
        states = kwargs["appliance_states"]
        people = kwargs["zone_people"]
        cm = kwargs["coverage_matrix"]
        watts = kwargs["appliance_watts"]
        temps = kwargs["zone_temps"]
        co2 = kwargs["zone_co2"]
        lux = kwargs["zone_lux"]
        comfort = kwargs["comfort_targets"]

        n_appliances = len(states)
        n_zones = len(people)
        occupied = np.where(people > 0)[0]
        n_occ = len(occupied)

        # Coverage score
        if n_occ > 0:
            covered = 0
            for zi in occupied:
                if any(states[ai] and cm[ai, zi] for ai in range(n_appliances)):
                    covered += 1
            cov_score = covered / n_occ
        else:
            cov_score = 1.0

        # Energy score
        total_w = float(np.dot(states, watts))
        max_w = float(watts.sum())
        energy_score = 1.0 - (total_w / max_w) if max_w > 0 else 1.0

        # Comfort scores
        target_t = comfort.get("target_temp", 25.0)
        temp_comfort = float(np.mean(np.clip(
            1.0 - np.abs(temps - target_t) / 10.0, 0, 1)))

        max_co2 = comfort.get("max_co2", 1000.0)
        co2_comfort = float(np.mean(np.clip(
            1.0 - np.maximum(0, co2 - max_co2) / 500.0, 0, 1)))

        target_lux = comfort.get("target_lux", 300.0)
        if n_occ > 0:
            lux_comfort = float(np.mean(np.clip(
                lux[occupied] / target_lux, 0, 1)))
        else:
            lux_comfort = 1.0

        captured_components.append({
            "coverage": cov_score,
            "energy": energy_score,
            "temp": temp_comfort,
            "co2": co2_comfort,
            "lux": lux_comfort,
            "total": r,
            "power_w": total_w,
            "n_active": int(states.sum()),
        })
        return r

    rmod.calculate_reward = patched_reward

    for ep in range(n_episodes):
        captured_components.clear()
        obs, _ = scenario_reset(env, "random", seed=ep + 42)

        for step in range(MAX_STEPS):
            action = _predict(model, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if not captured_components:
            continue

        cov  = np.mean([c["coverage"] for c in captured_components])
        eng  = np.mean([c["energy"]   for c in captured_components])
        temp = np.mean([c["temp"]     for c in captured_components])
        co2  = np.mean([c["co2"]      for c in captured_components])
        lux  = np.mean([c["lux"]      for c in captured_components])
        tot  = np.mean([c["total"]    for c in captured_components])
        pw   = np.mean([c["power_w"]  for c in captured_components])
        act  = np.mean([c["n_active"] for c in captured_components])

        log(f"  Ep {ep+1}  total_reward={tot:+.3f}  "
            f"power={pw:.0f}W  active={act:.1f}/{env.n_appliances}")
        log(f"    coverage={cov:.3f}  energy={eng:.3f}  "
            f"temp={temp:.3f}  co2={co2:.3f}  lux={lux:.3f}")

    rmod.calculate_reward = orig_fn
    log()


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — Action Pattern Analysis
# ══════════════════════════════════════════════════════════════

def section_action_patterns(model, env, n_episodes=5):
    log("=" * 70)
    log("  SECTION 3: Action Patterns (is agent stuck?)")
    log("=" * 70)

    appliance_on_counts = np.zeros(env.n_appliances)
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = scenario_reset(env, "random", seed=ep + 99)
        for step in range(MAX_STEPS):
            action = _predict(model, obs)
            obs, _, terminated, truncated, _ = env.step(action)

            appliance_on_counts += action[:env.n_appliances]
            total_steps += 1

            if terminated or truncated:
                break

    log(f"  Over {total_steps} steps across {n_episodes} episodes:")
    log()
    log("  FAN activation rates:")
    for i, fan in enumerate(env.cfg.fans):
        pct = 100 * appliance_on_counts[i] / total_steps
        bar = "█" * int(pct / 2)
        log(f"    {fan.id:4s}  {pct:5.1f}%  {bar}")

    log()
    log("  LIGHT activation rates:")
    for i, light in enumerate(env.cfg.lights):
        idx = env.n_fans + i
        pct = 100 * appliance_on_counts[idx] / total_steps
        bar = "█" * int(pct / 2)
        log(f"    {light.id:4s}  {pct:5.1f}%  {bar}")

    if env.cfg.n_projectors > 0:
        log()
        log("  PROJECTOR activation rates:")
        for i, proj in enumerate(env.cfg.projectors):
            idx = env.n_fans + env.n_lights + i
            pct = 100 * appliance_on_counts[idx] / total_steps
            bar = "█" * int(pct / 2)
            log(f"    {proj.id:4s}  {pct:5.1f}%  {bar}")

    # Diagnose
    log()
    all_rates = appliance_on_counts / total_steps
    log(f"  Avg activation rate : {100*all_rates.mean():.1f}%")
    if all_rates.mean() > 0.9:
        log("  ⚠  DIAGNOSIS: Agent is almost always ON — not learning energy savings")
    elif all_rates.mean() < 0.1:
        log("  ⚠  DIAGNOSIS: Agent is almost always OFF — ignoring comfort constraints")
    else:
        log("  ✓  Activation rate looks healthy (10–90%)")

    if all_rates.std() < 0.05:
        log("  ⚠  DIAGNOSIS: Very low variance — agent not differentiating appliances")
    log()


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — Physics Drift Over Episode
# ══════════════════════════════════════════════════════════════

def section_physics_drift(model, env, scenario="full"):
    log("=" * 70)
    log(f"  SECTION 4: Physics Drift (scenario={scenario})")
    log("=" * 70)
    log(f"  {'Step':>5}  {'AvgTemp':>8}  {'AvgCO2':>8}  {'AvgLux':>8}  "
        f"{'Active':>7}  {'Power':>8}  {'Reward':>8}")
    log(f"  {'-'*65}")

    obs, _ = scenario_reset(env, scenario, seed=7)

    for step in range(MAX_STEPS):
        action = _predict(model, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0 or step == MAX_STEPS - 1:
            log(f"  {step:>5}  "
                f"{info['avg_temp']:>7.1f}°  "
                f"{info['avg_co2']:>7.0f}p  "
                f"{env.zone_lux.mean():>7.0f}l  "
                f"{info['n_active']:>5}/{env.n_appliances}  "
                f"{info['total_power_w']:>6.0f}W  "
                f"{reward:>+8.3f}")

        if terminated or truncated:
            break

    log()
    final_temp = env.zone_temps.mean()
    final_co2  = env.zone_co2.mean()
    target_t   = env.cfg.comfort.target_temp_c
    max_co2    = env.cfg.comfort.max_co2_ppm

    if final_temp > target_t + 3:
        log(f"  ⚠  DIAGNOSIS: Temp drifted to {final_temp:.1f}°C (target {target_t}°C) — fans underused")
    if final_co2 > max_co2:
        log(f"  ⚠  DIAGNOSIS: CO₂ reached {final_co2:.0f} ppm (max {max_co2}) — ventilation failing")
    log()


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — Per-Scenario Summary
# ══════════════════════════════════════════════════════════════

def section_scenario_summary(model, env, n_episodes=5):
    log("=" * 70)
    log("  SECTION 5: Per-Scenario Summary")
    log("=" * 70)
    log(f"  {'Scenario':<10}  {'AvgReward':>10}  {'StdReward':>10}  "
        f"{'AvgPower':>10}  {'AvgActive':>10}  {'AvgTemp':>8}  {'AvgCO2':>8}")
    log(f"  {'-'*75}")

    for scenario in ["random", "full", "sparse", "empty"]:
        rewards, powers, actives, temps, co2s = [], [], [], [], []

        for ep in range(n_episodes):
            obs, _ = scenario_reset(env, scenario, seed=ep + 200)
            ep_r, ep_p, ep_a, ep_t, ep_c = 0, [], [], [], []

            for step in range(MAX_STEPS):
                action = _predict(model, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_r += reward
                ep_p.append(info["total_power_w"])
                ep_a.append(info["n_active"])
                ep_t.append(info["avg_temp"])
                ep_c.append(info["avg_co2"])
                if terminated or truncated:
                    break

            rewards.append(ep_r)
            powers.append(np.mean(ep_p))
            actives.append(np.mean(ep_a))
            temps.append(np.mean(ep_t))
            co2s.append(np.mean(ep_c))

        log(f"  {scenario:<10}  "
            f"{np.mean(rewards):>+10.2f}  "
            f"{np.std(rewards):>10.2f}  "
            f"{np.mean(powers):>9.0f}W  "
            f"{np.mean(actives):>9.1f}  "
            f"{np.mean(temps):>7.1f}°  "
            f"{np.mean(co2s):>7.0f}p")

    log()


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — Greedy vs PPO Comparison
# ══════════════════════════════════════════════════════════════

def section_greedy_vs_ppo(model, env, n_episodes=5):
    log("=" * 70)
    log("  SECTION 6: PPO vs Baselines (random scenario)")
    log("=" * 70)

    strategies = {
        "PPO (RL)": None,
        "All ON":   "all_on",
        "All OFF":  "all_off",
        "Random":   "random_act",
    }
    results = {}

    for name, strategy in strategies.items():
        rewards, powers = [], []
        for ep in range(n_episodes):
            obs, _ = scenario_reset(env, "random", seed=ep + 300)
            ep_r = 0.0
            ep_p = []

            for step in range(MAX_STEPS):
                if strategy is None:
                    action = _predict(model, obs)
                elif strategy == "all_on":
                    action = np.ones(env.n_appliances, dtype=np.int8)
                elif strategy == "all_off":
                    action = np.zeros(env.n_appliances, dtype=np.int8)
                else:
                    action = np.random.randint(0, 2, size=env.n_appliances).astype(np.int8)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_r += reward
                ep_p.append(info["total_power_w"])
                if terminated or truncated:
                    break

            rewards.append(ep_r)
            powers.append(np.mean(ep_p))

        results[name] = {
            "mean_r": np.mean(rewards),
            "std_r":  np.std(rewards),
            "mean_p": np.mean(powers),
        }

    ppo_r = results["PPO (RL)"]["mean_r"]
    log(f"  {'Strategy':<12}  {'AvgReward':>12}  {'StdReward':>12}  "
        f"{'AvgPower':>10}  {'vs PPO':>8}")
    log(f"  {'-'*62}")
    for name, r in results.items():
        delta = r["mean_r"] - ppo_r if name != "PPO (RL)" else 0.0
        sign  = "+" if delta >= 0 else ""
        log(f"  {name:<12}  "
            f"{r['mean_r']:>+12.2f}  "
            f"{r['std_r']:>12.2f}  "
            f"{r['mean_p']:>9.0f}W  "
            f"{'—' if name == 'PPO (RL)' else sign + f'{delta:.2f}':>8}")

    log()
    # Diagnose
    if ppo_r < results["All OFF"]["mean_r"]:
        log("  ⚠  CRITICAL: PPO is WORSE than doing nothing (all-off) — severe reward issue")
    elif ppo_r < results["Random"]["mean_r"]:
        log("  ⚠  SERIOUS: PPO is WORSE than random actions — policy not converging")
    elif ppo_r < results["All ON"]["mean_r"]:
        log("  ⚠  WARNING: PPO is WORSE than always-on — energy savings not worth comfort loss")
    else:
        log("  ✓  PPO beats all baselines")
    log()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global _model_obs_size, _model_n_appliances, _env_obs_size, _env_n_appliances

    model_path = os.path.join(_project_root, "models", "srace_ppo.zip")
    if not os.path.exists(model_path):
        # Try best checkpoint
        best_path = os.path.join(_project_root, "models", "best", "best_model.zip")
        if os.path.exists(best_path):
            model_path = best_path
        else:
            print(f"✗ No model found at {model_path}")
            print("  Train first: python ml/train_ppo.py --timesteps 500000")
            sys.exit(1)

    log(f"SRACE v2 — PPO Diagnostics Report")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model: {model_path}")
    log()

    print("Loading model...")
    model = PPO.load(model_path)

    print("Creating environment...")
    env = SRACEEnv()

    # Set up adaptation globals
    _model_obs_size = model.observation_space.shape[0]
    _model_n_appliances = model.action_space.shape[0]
    _env_obs_size = env.observation_space.shape[0]
    _env_n_appliances = env.n_appliances

    section_model_info(model, model_path, env)
    section_reward_breakdown(model, env, n_episodes=3)
    section_action_patterns(model, env, n_episodes=5)
    section_physics_drift(model, env, scenario="full")
    section_scenario_summary(model, env, n_episodes=5)
    section_greedy_vs_ppo(model, env, n_episodes=5)

    write_log()


if __name__ == "__main__":
    main()
