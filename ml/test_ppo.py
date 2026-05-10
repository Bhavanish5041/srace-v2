"""
test_ppo.py — Evaluate a trained PPO agent for SRACE room automation.

Usage:
    python ml/test_ppo.py                              # Quick eval (5 episodes)
    python ml/test_ppo.py --episodes 50                # Extended eval
    python ml/test_ppo.py --model models/best/best_model.zip  # Use best checkpoint
    python ml/test_ppo.py --render                     # Print per-step details
    python ml/test_ppo.py --scenario sparse            # Test specific occupancy
    python ml/test_ppo.py --config config/office_small.json    # Test on a different room
    python ml/test_ppo.py --config config/auditorium.json      # Large room generalisation

Scenarios:
    random  — Random occupancy each episode (default)
    full    — All zones occupied (stress test)
    sparse  — Only 2-3 zones occupied (efficiency test)
    empty   — No occupants (should learn to turn everything off)
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import PPO
from ml.gym_env import SRACEEnv, MAX_STEPS


# ── ANSI colours for terminal output ──────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    END    = "\033[0m"


def make_scenario_env(scenario: str, config_path: str = None) -> SRACEEnv:
    """Create an environment, optionally with fixed occupancy."""
    env = SRACEEnv(config_path=config_path)
    env._scenario = scenario  # tag for custom reset
    return env


def scenario_reset(env: SRACEEnv, scenario: str, seed=None):
    """
    Reset env with a specific occupancy scenario.
    Returns (obs, info).
    """
    obs, info = env.reset(seed=seed)

    if scenario == "random":
        return obs, info  # default random occupancy

    # Override occupancy after reset
    if scenario == "full":
        env.zone_people = np.full(env.n_zones, 5.0)
    elif scenario == "sparse":
        env.zone_people = np.zeros(env.n_zones)
        chosen = np.random.choice(env.n_zones, size=min(3, env.n_zones), replace=False)
        for z in chosen:
            env.zone_people[z] = np.random.randint(2, 6)
    elif scenario == "empty":
        env.zone_people = np.zeros(env.n_zones)

    # Recompute observation with new occupancy
    obs = env._get_observation()
    return obs, info


def run_baseline(env: SRACEEnv, strategy: str, n_episodes: int, scenario: str):
    """
    Run a baseline strategy for comparison.

    Strategies: 'all_on', 'all_off', 'random'
    """
    rewards = []
    powers = []

    for ep in range(n_episodes):
        obs, _ = scenario_reset(env, scenario, seed=ep + 1000)
        ep_reward = 0.0
        ep_power = 0.0

        for step in range(MAX_STEPS):
            if strategy == "all_on":
                action = np.ones(env.n_appliances, dtype=np.int8)
            elif strategy == "all_off":
                action = np.zeros(env.n_appliances, dtype=np.int8)
            else:  # random
                action = np.random.randint(0, 2, size=env.n_appliances).astype(np.int8)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_power += info["total_power_w"]

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        powers.append(ep_power / MAX_STEPS)  # avg per step

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_power": np.mean(powers),
    }


def adapt_obs(obs, env_obs_size, model_obs_size):
    """
    Adapt an environment observation to match the model's expected input size.
    Pads with zeros or truncates as needed.
    """
    if env_obs_size == model_obs_size:
        return obs
    adapted = np.zeros(model_obs_size, dtype=obs.dtype)
    n_copy = min(env_obs_size, model_obs_size)
    adapted[:n_copy] = obs[:n_copy]
    return adapted


def adapt_action(action, model_n_appliances, env_n_appliances):
    """
    Adapt a model's action to a different-sized environment.
    If the new room has fewer appliances, truncate.
    If more, pad with zeros (off).
    """
    if model_n_appliances == env_n_appliances:
        return action
    adapted = np.zeros(env_n_appliances, dtype=action.dtype)
    n_copy = min(model_n_appliances, env_n_appliances)
    adapted[:n_copy] = action[:n_copy]
    return adapted


def run_evaluation(model, env, n_episodes, scenario, render=False, model_n_appliances=None):
    """
    Run the PPO model through episodes and collect metrics.
    Handles observation/action space mismatches for cross-room generalisation.
    """
    all_rewards = []
    all_powers = []
    all_active = []
    all_temps = []
    all_co2 = []
    all_steps_data = []

    # Detect space sizes for adaptation
    env_obs_size = env.observation_space.shape[0]
    model_obs_size = model.observation_space.shape[0]
    needs_adaptation = (model_n_appliances is not None and
                        model_n_appliances != env.n_appliances)

    if needs_adaptation:
        print(f"  {C.YELLOW}Adapting: obs {env_obs_size}→{model_obs_size}, "
              f"action {model_n_appliances}→{env.n_appliances}{C.END}\n")

    for ep in range(n_episodes):
        obs, _ = scenario_reset(env, scenario, seed=ep)
        ep_reward = 0.0
        ep_power = []
        ep_active = []
        ep_temps = []
        ep_co2 = []

        initial_occupancy = env.zone_people.copy()
        n_occupied = int((initial_occupancy > 0).sum())

        if render:
            print(f"\n{C.BOLD}{'─' * 70}{C.END}")
            print(f"{C.CYAN}  Episode {ep + 1}  |  Occupied zones: {n_occupied}/{env.n_zones}  "
                  f"|  People: {initial_occupancy[initial_occupancy > 0].astype(int).tolist()}{C.END}")
            print(f"{'─' * 70}")

        for step in range(MAX_STEPS):
            # Adapt observation for model if room configs differ
            model_obs = adapt_obs(obs, env_obs_size, model_obs_size) if needs_adaptation else obs
            raw_action, _ = model.predict(model_obs, deterministic=True)
            action = adapt_action(raw_action, model_n_appliances, env.n_appliances) if needs_adaptation else raw_action
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_power.append(info["total_power_w"])
            ep_active.append(info["n_active"])
            ep_temps.append(info["avg_temp"])
            ep_co2.append(info["avg_co2"])

            if render and step % 20 == 0:
                fan_states = env.appliance_states[:env.n_fans]
                light_states = env.appliance_states[env.n_fans:]
                fans_on = [env.cfg.fans[i].id for i in range(env.n_fans) if fan_states[i]]
                lights_on = [env.cfg.lights[i].id for i in range(env.n_lights) if light_states[i]]

                power_color = C.GREEN if info["total_power_w"] < 400 else (
                    C.YELLOW if info["total_power_w"] < 700 else C.RED)

                print(
                    f"  Step {step:3d}  │  "
                    f"R={reward:+.3f}  │  "
                    f"{power_color}Power={info['total_power_w']:5.0f}W{C.END}  │  "
                    f"T={info['avg_temp']:.1f}°C  │  "
                    f"CO₂={info['avg_co2']:.0f}  │  "
                    f"Active={info['n_active']:2d}  │  "
                    f"Fans: {fans_on}  Lights: {lights_on}"
                )

            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_powers.append(np.mean(ep_power))
        all_active.append(np.mean(ep_active))
        all_temps.append(np.mean(ep_temps))
        all_co2.append(np.mean(ep_co2))

        if render:
            print(f"  {'─' * 68}")
            print(f"  {C.BOLD}Episode reward: {ep_reward:+.2f}  │  "
                  f"Avg power: {np.mean(ep_power):.0f}W  │  "
                  f"Avg active: {np.mean(ep_active):.1f}{C.END}")

    return {
        "rewards": all_rewards,
        "powers": all_powers,
        "active": all_active,
        "temps": all_temps,
        "co2": all_co2,
    }


def print_report(results, baselines, scenario, n_episodes, model_path):
    """Print a formatted evaluation report."""
    print(f"\n\n{'═' * 70}")
    print(f"{C.BOLD}  SRACE v2 — PPO Evaluation Report{C.END}")
    print(f"{'═' * 70}")
    print(f"  Model     : {model_path}")
    print(f"  Scenario  : {scenario}")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Steps/ep  : {MAX_STEPS}")
    print(f"{'═' * 70}")

    # ── PPO Agent Results ──
    mean_r = np.mean(results["rewards"])
    std_r  = np.std(results["rewards"])
    mean_p = np.mean(results["powers"])
    mean_a = np.mean(results["active"])
    mean_t = np.mean(results["temps"])
    mean_c = np.mean(results["co2"])

    print(f"\n{C.CYAN}  ┌─ PPO Agent ────────────────────────────────────────────┐{C.END}")
    print(f"{C.CYAN}  │{C.END}  Mean reward      : {C.BOLD}{mean_r:+.2f} ± {std_r:.2f}{C.END}")
    print(f"{C.CYAN}  │{C.END}  Avg power        : {C.BOLD}{mean_p:.0f} W{C.END}")
    print(f"{C.CYAN}  │{C.END}  Avg active appl. : {C.BOLD}{mean_a:.1f} / 20{C.END}")
    print(f"{C.CYAN}  │{C.END}  Avg temperature  : {C.BOLD}{mean_t:.1f} °C{C.END}  (target: 25.0)")
    print(f"{C.CYAN}  │{C.END}  Avg CO₂          : {C.BOLD}{mean_c:.0f} ppm{C.END}  (max: 1000)")
    print(f"{C.CYAN}  └────────────────────────────────────────────────────────┘{C.END}")

    # ── Baseline Comparison ──
    print(f"\n{C.YELLOW}  ┌─ Baseline Comparison ──────────────────────────────────┐{C.END}")
    print(f"{C.YELLOW}  │{C.END}  {'Strategy':<12}  {'Mean Reward':>13}  {'Avg Power':>11}  {'vs PPO':>8}")
    print(f"{C.YELLOW}  │{C.END}  {'─' * 50}")

    for name, bl in baselines.items():
        delta = mean_r - bl["mean_reward"]
        delta_color = C.GREEN if delta > 0 else C.RED
        print(
            f"{C.YELLOW}  │{C.END}  {name:<12}  "
            f"{bl['mean_reward']:+13.2f}  "
            f"{bl['mean_power']:9.0f} W  "
            f"{delta_color}{delta:+8.2f}{C.END}"
        )

    print(f"{C.YELLOW}  └────────────────────────────────────────────────────────┘{C.END}")

    # ── Per-episode breakdown ──
    print(f"\n{C.DIM}  Per-episode rewards:{C.END}")
    for i, r in enumerate(results["rewards"]):
        bar_len = max(0, int((r + 10) * 2))  # scale bar
        bar = "█" * min(bar_len, 40)
        color = C.GREEN if r > mean_r else C.DIM
        print(f"  {color}  Ep {i+1:3d}  {r:+8.2f}  {bar}{C.END}")

    print(f"\n{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="SRACE PPO Evaluation")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model .zip (default: models/srace_ppo.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--scenario", type=str, default="random",
        choices=["random", "full", "sparse", "empty"],
        help="Occupancy scenario to test (default: random)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Print per-step details during evaluation"
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip baseline comparison"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to room JSON config (default: config/default_room.json). "
             "Use to test generalisation on unseen rooms."
    )
    args = parser.parse_args()

    # ── Resolve model path ──
    if args.model is None:
        model_path = os.path.join(_project_root, "models", "srace_ppo.zip")
    else:
        model_path = args.model

    if not os.path.exists(model_path):
        print(f"{C.RED}  ✗ Model not found: {model_path}{C.END}")
        print(f"  Train first with: python ml/train_ppo.py --timesteps 200000")
        sys.exit(1)

    # ── Load model ──
    print(f"\n{C.BOLD}  Loading model: {model_path}{C.END}")
    model = PPO.load(model_path)
    model_n_appliances = model.action_space.shape[0]

    # ── Resolve config path ──
    config_path = None
    if args.config:
        config_path = os.path.join(_project_root, args.config) if not os.path.isabs(args.config) else args.config
        if not os.path.exists(config_path):
            print(f"{C.RED}  ✗ Config not found: {config_path}{C.END}")
            sys.exit(1)

    # ── Create environment ──
    env = SRACEEnv(config_path=config_path)
    print(f"  Environment: {env.n_zones} zones, {env.n_fans} fans, "
          f"{env.n_lights} lights ({env.n_appliances} appliances)")
    print(f"  Room: {env.cfg.name}")
    print(f"  Scenario: {args.scenario}")

    if model_n_appliances != env.n_appliances:
        print(f"{C.YELLOW}  ⚠ Action space mismatch: model={model_n_appliances}, "
              f"env={env.n_appliances}. Adapting actions.{C.END}")

    # ── Run PPO evaluation ──
    print(f"\n  Running {args.episodes} episodes...\n")
    t0 = time.perf_counter()

    results = run_evaluation(
        model, env, args.episodes, args.scenario, render=args.render,
        model_n_appliances=model_n_appliances
    )

    elapsed = time.perf_counter() - t0

    # ── Run baselines ──
    baselines = {}
    if not args.no_baseline:
        print(f"\n  Running baseline comparisons...")
        for strat in ["all_on", "all_off", "random"]:
            baselines[strat] = run_baseline(env, strat, args.episodes, args.scenario)

    # ── Print report ──
    print_report(results, baselines, args.scenario, args.episodes, model_path)

    print(f"  {C.DIM}Evaluation completed in {elapsed:.1f}s{C.END}\n")


if __name__ == "__main__":
    main()
