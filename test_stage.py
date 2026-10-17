import numpy as np, json, pickle, sys
from stable_baselines3 import SAC
from sim import *; from distill import *
from experiment import SEEDS, LAMS, LEAVES, viol, du
sel = json.load(open("results/select.json"))
# ---- apply the pre-registered selection rules (validation data only) ----
tviol = np.mean([sel[f"{s}/teacher"]["viol"] for s in SEEDS])
lam_v = {l: np.mean([sel[f"{s}/sym/{l}"]["viol"] for s in SEEDS]) for l in LAMS}
LAM = next((l for l in LAMS if l > 0 and lam_v[l] <= tviol), LAMS[-1])
tree_r = {lv: np.mean([sel[f"{s}/tree/{lv}"]["ret"] for s in SEEDS]) for lv in LEAVES}
ref = tree_r[LEAVES[-1]]
LEAF = next(lv for lv in LEAVES if tree_r[lv] >= ref - 0.05*abs(ref))
print("teacher viol", tviol, "lam viol", lam_v, "-> LAM", LAM); print("tree ret", tree_r, "-> LEAF", LEAF)

CONDS = [("nominal", {})] + [(f"mass x{k}", dict(m=k)) for k in [0.5,0.75,1.25,1.5,2.0]] \
      + [(f"length x{k}", dict(l=k)) for k in [0.7,0.85,1.15,1.3,1.5]] \
      + [(f"noise {k}", dict(obs_noise=k)) for k in [0.02,0.05,0.1,0.2]] \
      + [(f"push {k}", dict(push=k)) for k in [2.0,4.0,6.0]] \
      + [(f"umax {k}", dict(umax=k)) for k in [1.8,1.6,1.4,1.2]]
tth, tw = init_states(100, 2026)                                  # TEST initial states
rth, rw = np.meshgrid(np.linspace(-np.pi, np.pi, 61), np.linspace(-WMAX, WMAX, 61)); rth, rw = rth.ravel(), rw.ravel()
out = dict(LAM=LAM, LEAF=LEAF, lam_viol=lam_v, teacher_viol=tviol, tree_ret=tree_r, conds=[c for c,_ in CONDS], data={})
for s in SEEDS:
    T = teacher_fn(SAC.load(f"models/sac_seed{s}"))
    tree = pickle.load(open(f"results/students/tree_s{s}_l{LEAF}.pkl","rb"))
    ctrls = {"Teacher (SAC)": T, f"Tree ({LEAF} leaves)": (lambda m: lambda o: m.predict(o))(tree),
             "Symbolic-plain": Symbolic(np.array(sel[f"{s}/sym/0.0"]["w"])),
             "Symbolic-stable": Symbolic(np.array(sel[f"{s}/sym/{LAM}"]["w"]))}
    for name, pol in ctrls.items():
        d = dict(viol=viol(pol), nnz=(pol.nnz() if isinstance(pol, Symbolic) else None),
                 expr=(pol.expr() if isinstance(pol, Symbolic) else None), cond={})
        for cname, kw in CONDS:
            kw = dict(kw); g = dict(m=M*kw.pop("m",1), l=L*kw.pop("l",1))
            o = rollout(pol, tth, tw, rng=np.random.default_rng(99), **g, **kw)
            d["cond"][cname] = dict(ret=o["ret"].tolist(), succ=success(o).astype(int).tolist(), du=du(o))
        o = rollout(pol, rth, rw, T=300); d["roa"] = success(o).astype(int).tolist()
        out["data"][f"{s}/{name}"] = d; print(s, name, d["cond"]["nominal"]["du"], np.mean(d["roa"]), flush=True)
json.dump(out, open("results/test.json","w"))
