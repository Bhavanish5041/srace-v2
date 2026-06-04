"""Quick smoke test for the updated gym environment with projectors."""
import sys
sys.path.insert(0, '.')

from ml.gym_env import SRACEEnv

env = SRACEEnv()
print(f"coverage shape: {env.coverage_matrix.shape}")
print(f"n_appliances: {env.n_appliances}")
print(f"n_projectors: {env.n_projectors}")

obs, _ = env.reset(seed=42)
print(f"obs shape: {obs.shape}")

for i in range(10):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)

print(f"10 steps done. last reward: {r:.3f}")
print("PASS")
