"""
visualize_results.py — Generate comparison charts for SRACE PPO evaluation.

Creates publication-ready plots comparing PPO agent vs baselines.
Outputs saved to output/ directory.

Usage:
    python3 visualize_results.py
"""

import os
import sys
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO
from ml.gym_env import SRACEEnv, MAX_STEPS
from ml.test_ppo import scenario_reset, run_baseline

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = {
    "ppo": "#58a6ff",
    "greedy": "#3fb950",
    "all_on": "#f85149",
    "all_off": "#8b949e",
    "random": "#d29922",
}


def run_ppo_episode_detailed(model, env, scenario, seed=0):
    """Run a single episode and collect per-step metrics."""
    obs, _ = scenario_reset(env, scenario, seed=seed)
    powers, temps, co2s, actives, rewards = [], [], [], [], []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        powers.append(info["total_power_w"])
        temps.append(info["avg_temp"])
        co2s.append(info["avg_co2"])
        actives.append(info["n_active"])
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "power": powers, "temp": temps, "co2": co2s,
        "active": actives, "reward": rewards,
    }


def main():
    os.makedirs("output", exist_ok=True)

    # Load model
    model_path = os.path.join(_project_root, "models", "srace_ppo.zip")
    if not os.path.exists(model_path):
        print("✗ No trained model found. Run: python3 ml/train_ppo.py --timesteps 500000")
        return

    print("Loading model...")
    model = PPO.load(model_path)
    env = SRACEEnv()

    # ═══════════════════════════════════════════════════════════
    #  Chart 1: PPO Step-by-Step (Sparse Scenario)
    # ═══════════════════════════════════════════════════════════
    print("Running sparse episode...")
    data = run_ppo_episode_detailed(model, env, "sparse", seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("SRACE PPO Agent — Sparse Occupancy (3/12 zones)",
                 fontsize=16, fontweight="bold", color="#58a6ff")

    steps = range(len(data["power"]))

    # Power
    axes[0, 0].fill_between(steps, data["power"], alpha=0.3, color=COLORS["ppo"])
    axes[0, 0].plot(steps, data["power"], color=COLORS["ppo"], linewidth=2)
    axes[0, 0].axhline(y=1150, color=COLORS["all_on"], linestyle="--", alpha=0.5, label="All ON (1150W)")
    axes[0, 0].set_ylabel("Power (W)")
    axes[0, 0].set_title("Power Consumption")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Temperature
    axes[0, 1].plot(steps, data["temp"], color="#f0883e", linewidth=2)
    axes[0, 1].axhline(y=27, color=COLORS["greedy"], linestyle="--", alpha=0.5, label="Target (27°C)")
    axes[0, 1].set_ylabel("Temperature (°C)")
    axes[0, 1].set_title("Temperature Control")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # CO₂
    axes[1, 0].plot(steps, data["co2"], color="#a371f7", linewidth=2)
    axes[1, 0].axhline(y=1000, color=COLORS["all_on"], linestyle="--", alpha=0.5, label="Max (1000 ppm)")
    axes[1, 0].set_ylabel("CO₂ (ppm)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("CO₂ Management")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Reward
    axes[1, 1].plot(steps, data["reward"], color=COLORS["ppo"], linewidth=2)
    axes[1, 1].axhline(y=0, color="#8b949e", linestyle="-", alpha=0.3)
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_title("Per-Step Reward")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/ppo_sparse_episode.png", dpi=150, bbox_inches="tight")
    print("✓ Saved output/ppo_sparse_episode.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  Chart 2: PPO Step-by-Step (Full Scenario)
    # ═══════════════════════════════════════════════════════════
    print("Running full episode...")
    data_full = run_ppo_episode_detailed(model, env, "full", seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("SRACE PPO Agent — Full Occupancy (12/12 zones, 60 people)",
                 fontsize=16, fontweight="bold", color="#f85149")

    steps = range(len(data_full["power"]))

    axes[0, 0].fill_between(steps, data_full["power"], alpha=0.3, color=COLORS["ppo"])
    axes[0, 0].plot(steps, data_full["power"], color=COLORS["ppo"], linewidth=2)
    axes[0, 0].axhline(y=1150, color=COLORS["all_on"], linestyle="--", alpha=0.5, label="All ON (1150W)")
    axes[0, 0].set_ylabel("Power (W)")
    axes[0, 0].set_title("Power Consumption")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, data_full["temp"], color="#f0883e", linewidth=2)
    axes[0, 1].axhline(y=27, color=COLORS["greedy"], linestyle="--", alpha=0.5, label="Target (27°C)")
    axes[0, 1].set_ylabel("Temperature (°C)")
    axes[0, 1].set_title("Temperature Control")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, data_full["co2"], color="#a371f7", linewidth=2)
    axes[1, 0].axhline(y=1000, color=COLORS["all_on"], linestyle="--", alpha=0.5, label="Max (1000 ppm)")
    axes[1, 0].set_ylabel("CO₂ (ppm)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("CO₂ Management")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, data_full["reward"], color=COLORS["ppo"], linewidth=2)
    axes[1, 1].axhline(y=0, color="#8b949e", linestyle="-", alpha=0.3)
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_title("Per-Step Reward")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/ppo_full_episode.png", dpi=150, bbox_inches="tight")
    print("✓ Saved output/ppo_full_episode.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════
    #  Chart 3: Baseline Comparison Bar Chart
    # ═══════════════════════════════════════════════════════════
    print("Running baseline comparisons (10 episodes each)...")
    n_eps = 10

    # Run PPO
    ppo_rewards = []
    ppo_powers = []
    for ep in range(n_eps):
        obs, _ = scenario_reset(env, "random", seed=ep)
        ep_r, ep_p = 0.0, 0.0
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_p += info["total_power_w"]
            if term or trunc:
                break
        ppo_rewards.append(ep_r)
        ppo_powers.append(ep_p / MAX_STEPS)

    # Run baselines
    baselines = {}
    for strat in ["all_on", "all_off", "random"]:
        baselines[strat] = run_baseline(env, strat, n_eps, "random")

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SRACE — PPO vs Baselines (10 Random Episodes)",
                 fontsize=16, fontweight="bold", color="#58a6ff")

    strategies = ["PPO Agent", "All ON", "All OFF", "Random"]
    colors = [COLORS["ppo"], COLORS["all_on"], COLORS["all_off"], COLORS["random"]]
    rewards = [
        np.mean(ppo_rewards),
        baselines["all_on"]["mean_reward"],
        baselines["all_off"]["mean_reward"],
        baselines["random"]["mean_reward"],
    ]
    powers = [
        np.mean(ppo_powers),
        baselines["all_on"]["mean_power"],
        baselines["all_off"]["mean_power"],
        baselines["random"]["mean_power"],
    ]

    bars1 = ax1.bar(strategies, rewards, color=colors, edgecolor="#30363d", linewidth=1.5)
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title("Reward Comparison")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:+.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(strategies, powers, color=colors, edgecolor="#30363d", linewidth=1.5)
    ax2.set_ylabel("Avg Power (W)")
    ax2.set_title("Power Consumption")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, powers):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{val:.0f}W", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("output/ppo_vs_baselines.png", dpi=150, bbox_inches="tight")
    print("✓ Saved output/ppo_vs_baselines.png")
    plt.close()

    print("\n✓ All charts generated in output/")
    print("  • output/ppo_sparse_episode.png  — step-by-step sparse scenario")
    print("  • output/ppo_full_episode.png    — step-by-step full scenario")
    print("  • output/ppo_vs_baselines.png    — PPO vs baselines bar chart")


if __name__ == "__main__":
    main()
