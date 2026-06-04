"""
train_ppo.py — Train a PPO agent for SRACE room automation.

Usage:
    python ml/train_ppo.py                    # Quick test (1k steps)
    python ml/train_ppo.py --timesteps 2000000  # Full training run

The trained model is saved to models/srace_ppo.zip
"""

import argparse
import os
import sys
import time

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from ml.gym_env import SRACEEnv


def make_env():
    """Factory function for vectorized env creation."""
    return SRACEEnv()


def main():
    parser = argparse.ArgumentParser(description="SRACE PPO Training")
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Total training timesteps (default: 1000 for testing)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing model to resume training from"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50000,
        help="Evaluation frequency (default: 50000)"
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  SRACE v2 — PPO Training")
    print("═" * 60)
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Parallel  : {args.n_envs} environments")
    print(f"  LR        : {args.lr}")
    print(f"  Resume    : {args.resume or 'No (fresh start)'}")
    print("═" * 60 + "\n")

    # ── Create vectorized environments ──
    print("  Creating vectorized environments...")
    vec_env = make_vec_env(make_env, n_envs=args.n_envs)

    # ── Create or load model ──
    model_dir = os.path.join(_project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "srace_ppo")

    if args.resume:
        print(f"  Loading model from {args.resume}...")
        model = PPO.load(args.resume, env=vec_env)
        model.learning_rate = args.lr
    else:
        print("  Initialising PPO agent...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=256,           # steps per rollout per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,         # entropy bonus for exploration
            # tensorboard_log=os.path.join(_project_root, "logs", "ppo_tensorboard"),
        )

    # ── Evaluation callback ──
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(model_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # ── Train ──
    print(f"\n  Starting training for {args.timesteps:,} timesteps...\n")
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=False,
    )

    elapsed = time.perf_counter() - t0
    fps = args.timesteps / elapsed

    # ── Save ──
    model.save(model_path)
    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Time     : {elapsed:.1f}s ({fps:.0f} steps/sec)")
    print(f"  Model    : {model_path}.zip")
    print(f"  Best     : {os.path.join(model_dir, 'best', 'best_model.zip')}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
