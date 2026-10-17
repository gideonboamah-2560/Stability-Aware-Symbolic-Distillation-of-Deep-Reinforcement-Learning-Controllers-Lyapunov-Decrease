"""Vectorised closed-loop simulator reproducing Pendulum-v1 exactly (checked in test_sim.py)."""
import numpy as np
from common import *

def rollout(policy, th0, thd0, T=200, g=G, m=M, l=L, umax=UMAX,
            obs_noise=0.0, push=0.0, push_t=100, rng=None, record=False):
    """policy: f(obs (N,3)) -> u (N,). Returns dict with returns, traj."""
    rng = rng or np.random.default_rng(0)
    th, thd = th0.astype(float).copy(), thd0.astype(float).copy()
    N = len(th); ret = np.zeros(N); TH = np.zeros((T, N)); W = np.zeros((T, N)); U = np.zeros((T, N))
    signs = rng.choice([-1.0, 1.0], N)
    for t in range(T):
        if push and t == push_t: thd = np.clip(thd + signs*push, -WMAX, WMAX)
        thm = th + obs_noise*rng.standard_normal(N); wm = thd + obs_noise*rng.standard_normal(N)
        obs = np.stack([np.cos(thm), np.sin(thm), np.clip(wm, -WMAX, WMAX)], 1)
        u = np.clip(policy(obs), -umax, umax)
        ret -= wrap(th)**2 + 0.1*thd**2 + 0.001*u**2
        TH[t], W[t], U[t] = th, thd, u
        th, thd = step_np(th, thd, u, g, m, l, umax)
    return dict(ret=ret, th=TH, thd=W, u=U)

def success(out, last=50, tol=0.1):
    """Upright-stabilised: |wrapped angle| < tol rad over the final `last` steps."""
    return (np.abs(wrap(out["th"][-last:])) < tol).all(0)

def teacher_fn(model):
    return lambda obs: model.predict(obs.astype(np.float32), deterministic=True)[0][:, 0]

def init_states(n, seed):
    r = np.random.default_rng(seed)
    return r.uniform(-np.pi, np.pi, n), r.uniform(-1, 1, n)   # Pendulum-v1 default reset distribution
