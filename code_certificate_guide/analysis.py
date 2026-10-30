import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from experiment import SEEDS, LAMS
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200})
sel, R = json.load(open("results/select.json")), json.load(open("results/test.json"))
LAM, LEAF, CONDS = R["LAM"], R["LEAF"], R["conds"]
NAMES = ["Teacher (SAC)", f"Tree ({LEAF} leaves)", "Symbolic-plain", "Symbolic-stable"]
COL = dict(zip(NAMES, ["k", "tab:orange", "tab:blue", "tab:green"]))
get = lambda s, n: R["data"][f"{s}/{n}"]
rep = {}
# ---------- Fig 1: lambda sweep (validation) ----------
vr = np.array([[sel[f"{s}/sym/{l}"]["viol"] for l in LAMS] for s in SEEDS])
rr = np.array([[sel[f"{s}/sym/{l}"]["ret"] for l in LAMS] for s in SEEDS])
nz = np.array([[sel[f"{s}/sym/{l}"]["nnz"] for l in LAMS] for s in SEEDS])
tv = np.array([sel[f"{s}/teacher"]["viol"] for s in SEEDS])
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6)); x = np.arange(len(LAMS))
ax[0].errorbar(x, 100*vr.mean(0), 100*vr.std(0), marker="o", ms=3, capsize=2, c="tab:green")
ax[0].axhline(100*tv.mean(), ls="--", c="k", lw=.8); ax[0].text(0.1, 100*tv.mean()+2, "teacher", fontsize=6)
ax[0].set_ylabel("violations in $\\mathcal{D}$ (%)"); ax[1].errorbar(x, rr.mean(0), rr.std(0), marker="s", ms=3, capsize=2, c="tab:blue")
ax[1].set_ylabel("validation return")
for a in ax: a.set_xticks(x); a.set_xticklabels([f"{l:g}" for l in LAMS], fontsize=6); a.set_xlabel("$\\lambda_{stab}$")
fig.tight_layout(); fig.savefig("figs/lambda_sweep.pdf")
rep["sweep"] = dict(viol=vr.mean(0).tolist(), viol_sd=vr.std(0).tolist(), ret=rr.mean(0).tolist(), ret_sd=rr.std(0).tolist(),
                    nnz=nz.mean(0).tolist(), teacher_viol=tv.mean())
# ---------- Table: nominal ----------
tab = {}
for n in NAMES:
    ret = [np.mean(get(s,n)["cond"]["nominal"]["ret"]) for s in SEEDS]
    suc = [np.mean(get(s,n)["cond"]["nominal"]["succ"]) for s in SEEDS]
    vio = [get(s,n)["viol"] for s in SEEDS]; dU = [get(s,n)["cond"]["nominal"]["du"] for s in SEEDS]
    roa = [np.mean(get(s,n)["roa"]) for s in SEEDS]; nnz = [get(s,n)["nnz"] for s in SEEDS]
    allsuc = [np.mean([np.mean(get(s,n)["cond"][c]["succ"]) for c in CONDS]) for s in SEEDS]
    tab[n] = {k:(float(np.mean(v)), float(np.std(v))) for k,v in dict(ret=ret, succ=suc, viol=vio, du=dU, roa=roa, allsucc=allsuc).items()}
    tab[n]["nnz"] = None if nnz[0] is None else (float(np.mean(nnz)), float(np.std(nnz)))
rep["table"] = tab
# ---------- Fig 2: robustness ----------
groups = [("mass x", "mass scale", [1.0]), ("length x", "length scale", [1.0]), ("noise ", "obs. noise std", [0.0]),
          ("umax ", "torque limit", [2.0]), ("push ", "push (rad/s)", [0.0])]
fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.55), sharey=True)
for a, (pre, lab, nomx) in zip(axs, groups):
    cs = [c for c in CONDS if c.startswith(pre)]; xs = nomx + [float(c[len(pre):]) for c in cs]; cs = ["nominal"] + cs
    o = np.argsort(xs); xs = np.array(xs)[o]; cs = np.array(cs)[o]
    for n in NAMES:
        m = np.array([[np.mean(get(s,n)["cond"][c]["succ"]) for c in cs] for s in SEEDS])
        a.errorbar(xs, 100*m.mean(0), 100*m.std(0), c=COL[n], marker="o", ms=2, lw=.9, capsize=1.5, label=n)
    a.set_xlabel(lab, fontsize=7)
axs[0].set_ylabel("success (%)"); axs[0].legend(fontsize=5, loc="lower left"); fig.tight_layout(); fig.savefig("figs/robustness.pdf")
# ---------- Fig 3: ROA maps seed 0 ----------
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=True)
for a, n in zip(axs, NAMES):
    g = np.array(get(0,n)["roa"]).reshape(61, 61)
    a.imshow(g, origin="lower", extent=[-np.pi, np.pi, -8, 8], aspect="auto", cmap="Greens", vmin=0, vmax=1.3)
    a.set_title(f"{n}: {100*g.mean():.0f}%", fontsize=7); a.set_xlabel("$\\theta_0$ (rad)")
axs[0].set_ylabel("$\\dot\\theta_0$ (rad/s)"); fig.tight_layout(); fig.savefig("figs/roa.pdf")
# ---------- statistics ----------
def per(n, key):
    return np.array([np.mean(get(s,n)["cond"][c][key]) for s in SEEDS for c in CONDS])
st = {}
for a_, b_ in [("Symbolic-stable","Symbolic-plain"), ("Symbolic-stable", NAMES[1]), ("Symbolic-stable","Teacher (SAC)")]:
    for key in ["ret","succ"]:
        x, y = per(a_, key), per(b_, key); d = x - y
        p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        st[f"{a_} vs {b_} [{key}]"] = dict(mean_diff=float(d.mean()), wins=int((d>0).sum()), losses=int((d<0).sum()), n=len(d), p=float(p))
rep["stats"] = st
rep["exprs"] = {s: {n: get(s,n)["expr"] for n in ["Symbolic-plain","Symbolic-stable"]} for s in SEEDS}
rep["percond"] = {n: {c: (float(np.mean([np.mean(get(s,n)["cond"][c]["succ"]) for s in SEEDS])),
                         float(np.mean([np.mean(get(s,n)["cond"][c]["ret"]) for s in SEEDS]))) for c in CONDS} for n in NAMES}
json.dump(rep, open("results/report.json","w"), indent=1)
print(json.dumps({k: rep[k] for k in ["table","stats"]}, indent=1))
