import numpy as np, gymnasium as gym
from sim import *
rng = np.random.default_rng(1)
pol = lambda o: 1.7*np.sin(3*o[:,2]) - 2*o[:,1]       # arbitrary nonlinear test policy
env = gym.make("Pendulum-v1"); maxerr=0; rets=[]
for ep in range(5):
    obs,_ = env.reset(seed=ep); th0, thd0 = env.unwrapped.state.copy()
    R=0
    for t in range(200):
        obs,r,te,tr,_ = env.step(np.array([pol(obs[None])[0]],dtype=np.float32)); R+=r
    out = rollout(pol, np.array([th0]), np.array([thd0]))
    maxerr=max(maxerr, abs(out["ret"][0]-R)); rets.append((R, out["ret"][0]))
print("returns gym vs sim:", [(round(a,3),round(b,3)) for a,b in rets]); print("max abs diff", maxerr)
