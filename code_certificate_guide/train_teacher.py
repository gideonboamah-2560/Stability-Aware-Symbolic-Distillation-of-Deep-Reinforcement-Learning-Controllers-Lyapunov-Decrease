"""Train SAC teachers on Pendulum-v1 using RL-Zoo tuned hyperparameters
(n_timesteps=20000, learning_rate=1e-3, all else SB3 defaults)."""
import sys, time, torch, gymnasium as gym
from stable_baselines3 import SAC
torch.set_num_threads(1)
seed = int(sys.argv[1])
env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, learning_rate=1e-3, seed=seed, verbose=0)
t = time.time(); model.learn(total_timesteps=20000)
model.save(f"models/sac_seed{seed}")
print(f"seed {seed} done in {time.time()-t:.0f}s", flush=True)
