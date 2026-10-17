import numpy as np, time
from stable_baselines3 import SAC
from sim import *; from distill import *
def grid_D(n=401):
    th, w = np.meshgrid(np.linspace(-0.2,0.2,n), np.linspace(-1.5,1.5,n)); th, w = th.ravel(), w.ravel()
    k = (V_np(th,w) <= C) & (V_np(th,w) > 1e-6); return th[k], w[k]
gth, gw = grid_D(); obs = np.stack([np.cos(gth), np.sin(gth), gw],1)
T = teacher_fn(SAC.load("models/sac_seed0"))
print("grid pts", len(gth), " teacher viol rate", lyap_violation(gth, gw, T(obs)).mean())
th0, thd0 = init_states(100, seed=777)
for ls in [0.0, 1.0, 10.0, 100.0]:
    t=time.time(); s = dagger(T, "sym", lam_stab=ls, lam1=1e-2, steps=800)
    o = rollout(s, th0, thd0); v = lyap_violation(gth, gw, s(obs)).mean()
    print(f"lam_stab={ls}: ret {o['ret'].mean():.1f} succ {success(o).mean():.2f} nnz {s.nnz()} viol {v:.3f} ({time.time()-t:.0f}s)\n   {s.expr()}")
