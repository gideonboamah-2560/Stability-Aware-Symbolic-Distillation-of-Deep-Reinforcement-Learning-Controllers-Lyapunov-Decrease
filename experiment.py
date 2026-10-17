"""Stage 'select': DAgger students for every teacher seed and hyper-parameter value, scored on
VALIDATION initial states (seed 777). Stage 'test': chosen controllers on TEST states (seed 2026)
under 22 conditions + region-of-attraction grid + Lyapunov violation on D."""
import numpy as np, json, sys, os, pickle, time
from stable_baselines3 import SAC
from sim import *; from distill import *
SEEDS = [0,1,2,3,4]; LAMS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]; LEAVES = [8,16,32,64,128]
os.makedirs("results/students", exist_ok=True)

def grid_D(n=401):
    th, w = np.meshgrid(np.linspace(-0.2,0.2,n), np.linspace(-1.5,1.5,n)); th, w = th.ravel(), w.ravel()
    k = (V_np(th,w) <= C) & (V_np(th,w) > 1e-6); return th[k], w[k]
GTH, GW = grid_D(); GOBS = np.stack([np.cos(GTH), np.sin(GTH), GW], 1)
def viol(pol): return float(lyap_violation(GTH, GW, pol(GOBS)).mean())
def du(o): return float(np.abs(np.diff(o["u"], axis=0)).mean())   # mean |u_t - u_{t-1}| (chattering)

def select():
    vth, vw = init_states(100, 777)
    res = json.load(open('results/select.json')) if os.path.exists('results/select.json') else {}
    for s in SEEDS:
        if f'{s}/tree/{LEAVES[-1]}' in res: continue
        while not os.path.exists(f"models/sac_seed{s}.zip"): time.sleep(20)
        time.sleep(2); T = teacher_fn(SAC.load(f"models/sac_seed{s}"))
        o = rollout(T, vth, vw)
        res[f"{s}/teacher"] = dict(ret=float(o["ret"].mean()), succ=float(success(o).mean()), viol=viol(T))
        for lam in LAMS:
            st = dagger(T, "sym", lam_stab=lam, lam1=1e-2, steps=800, seed=s); o = rollout(st, vth, vw)
            res[f"{s}/sym/{lam}"] = dict(ret=float(o["ret"].mean()), succ=float(success(o).mean()),
                                         viol=viol(st), nnz=st.nnz(), w=st.w.tolist())
            print(s, "sym", lam, res[f"{s}/sym/{lam}"]["ret"], res[f"{s}/sym/{lam}"]["viol"], st.nnz(), flush=True)
        for lv in LEAVES:
            st = dagger(T, "tree", leaves=lv, seed=s); o = rollout(st, vth, vw)
            pickle.dump(st.model, open(f"results/students/tree_s{s}_l{lv}.pkl","wb"))
            res[f"{s}/tree/{lv}"] = dict(ret=float(o["ret"].mean()), succ=float(success(o).mean()), viol=viol(st))
            print(s, "tree", lv, res[f"{s}/tree/{lv}"]["ret"], flush=True)
        json.dump(res, open("results/select.json","w"), indent=1)

if __name__ == "__main__":
    select()
