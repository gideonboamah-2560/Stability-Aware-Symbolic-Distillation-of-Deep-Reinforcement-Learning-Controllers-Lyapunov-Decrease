"""Applies the pre-registered lambda_r rule on VALIDATION results, then evaluates the selected
certificate-guided students on the TEST protocol (identical to test_stage.py) and appends them."""
import numpy as np, json, sys
from sim import *; from distill import Symbolic
from experiment import viol, du
import certify2 as C2
LAMS = [1, 3, 10, 30, 100]; SEEDS = range(5)
H = {(s, l): json.load(open(f"results/cegis_s{s}_lam{l:g}.json")) for s in SEEDS for l in LAMS}
summ = {}
for l in LAMS:
    fin = [H[(s, l)][-1] for s in SEEDS]
    summ[l] = dict(frac=float(np.mean([f["frac"] for f in fin])), succ=float(np.mean([f["val_succ"] for f in fin])), ret=float(np.mean([f["val_ret"] for f in fin])),
                   nnz=float(np.mean([f["nnz"] for f in fin])), rounds=float(np.mean([len(H[(s, l)]) for s in SEEDS])))
    print(f"lam_r {l:>4}: certified {100*summ[l]['frac']:6.2f}% | val success {100*summ[l]['succ']:5.1f}% | val return {summ[l]['ret']:8.1f} | terms {summ[l]['nnz']:.1f} | rounds {summ[l]['rounds']:.1f}")
ok = [l for l in LAMS if summ[l]["succ"] >= 0.95]
LAM = max(ok, key=lambda l: summ[l]["frac"]) if ok else None
print("selected lambda_r =", LAM)
if LAM is None: sys.exit("no lambda_r meets the success constraint")
R = json.load(open("results/test.json")); CONDS = [("nominal", {})] + [(f"mass x{k}", dict(m=k)) for k in [0.5,0.75,1.25,1.5,2.0]] \
  + [(f"length x{k}", dict(l=k)) for k in [0.7,0.85,1.15,1.3,1.5]] + [(f"noise {k}", dict(obs_noise=k)) for k in [0.02,0.05,0.1,0.2]] \
  + [(f"push {k}", dict(push=k)) for k in [2.0,4.0,6.0]] + [(f"umax {k}", dict(umax=k)) for k in [1.8,1.6,1.4,1.2]]
assert [c for c, _ in CONDS] == R["conds"]
tth, tw = init_states(100, 2026); rth, rw = np.meshgrid(np.linspace(-np.pi, np.pi, 61), np.linspace(-WMAX, WMAX, 61)); rth, rw = rth.ravel(), rw.ravel()
for s in SEEDS:
    w = np.array(H[(s, LAM)][-1]["w"]); pol = Symbolic(w); cert = C2.certify(w)
    d = dict(viol=viol(pol), nnz=pol.nnz(), expr=pol.expr(), cert_frac=cert["frac"], cert_status=cert["status"], cond={})
    for cname, kw in CONDS:
        kw = dict(kw); g = dict(m=M*kw.pop("m", 1), l=L*kw.pop("l", 1))
        o = rollout(pol, tth, tw, rng=np.random.default_rng(99), **g, **kw); d["cond"][cname] = dict(ret=o["ret"].tolist(), succ=success(o).astype(int).tolist(), du=du(o))
    o = rollout(pol, rth, rw, T=300); d["roa"] = success(o).astype(int).tolist()
    R["data"][f"{s}/Symbolic-certified"] = d; print(s, "certified", round(100*cert["frac"], 2), "% | nominal succ", np.mean(d["cond"]["nominal"]["succ"]), flush=True)
R["LAM_R"] = LAM; R["lamr_summary"] = {str(k): v for k, v in summ.items()}
json.dump(R, open("results/test2.json", "w")); print("saved results/test2.json")
