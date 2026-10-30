import numpy as np, time
from stable_baselines3 import SAC
from sim import *; from distill import *
T = teacher_fn(SAC.load("models/sac_seed0"))
th0, thd0 = init_states(100, seed=777)            # VALIDATION initial states
o = rollout(T, th0, thd0); print(f"teacher: ret {o['ret'].mean():.1f}  succ {success(o).mean():.2f}")
for lam1 in [1e-3, 1e-2, 3e-2]:
    t=time.time(); s = dagger(T, "sym", lam_stab=0.0, lam1=lam1, steps=800)
    o = rollout(s, th0, thd0)
    print(f"sym lam1={lam1}: ret {o['ret'].mean():.1f} succ {success(o).mean():.2f} nnz {s.nnz()} ({time.time()-t:.0f}s)")
    print("   ", s.expr())
