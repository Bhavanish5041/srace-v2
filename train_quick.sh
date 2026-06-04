#!/bin/bash
# Quick PPO training script
cd /home/bhavanish/.gemini/antigravity/scratch/srace-v2
source venv/bin/activate
python ml/train_ppo.py --timesteps 2000 --n-envs 1 2>&1 | tail -20
