"""Regenerate the three figures as PGF (vector LaTeX code) from the same result files."""
import numpy as np, json, matplotlib; matplotlib.use("pgf"); import matplotlib.pyplot as plt
from experiment import SEEDS, LAMS
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pgf.rcfonts": False, "pgf.texsystem": "pdflatex"})
sel, R = json.load(open("results/select.json")), json.load(open("results/test.json"))
LEAF, CONDS = R["LEAF"], R["conds"]
NAMES = ["Teacher (SAC)", f"Tree ({LEAF} leaves)", "Symbolic-plain", "Symbolic-stable"]
COL = dict(zip(NAMES, ["k", "tab:orange", "tab:blue", "tab:green"]))
get = lambda s, n: R["data"][f"{s}/{n}"]
# Fig 1
vr = np.array([[sel[f"{s}/sym/{l}"]["viol"] for l in LAMS] for s in SEEDS]); rr = np.array([[sel[f"{s}/sym/{l}"]["ret"] for l in LAMS] for s in SEEDS])
tv = np.array([sel[f"{s}/teacher"]["viol"] for s in SEEDS])
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6)); x = np.arange(len(LAMS))
ax[0].errorbar(x, 100*vr.mean(0), 100*vr.std(0), marker="o", ms=3, capsize=2, c="tab:green")
ax[0].axhline(100*tv.mean(), ls="--", c="k", lw=.8); ax[0].text(0.1, 100*tv.mean()+2, "teacher", fontsize=6)
ax[0].set_ylabel(r"violations in $\mathcal{D}$ (\%)")
ax[1].errorbar(x, rr.mean(0), rr.std(0), marker="s", ms=3, capsize=2, c="tab:blue"); ax[1].set_ylabel("validation return")
for a in ax: a.set_xticks(x); a.set_xticklabels([f"{l:g}" for l in LAMS], fontsize=6); a.set_xlabel(r"$\lambda_{stab}$")
fig.tight_layout(); fig.savefig("figs/lambda_sweep.pgf")
# Fig 2
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
axs[0].set_ylabel(r"success (\%)"); axs[0].legend(fontsize=5, loc="lower left"); fig.tight_layout(); fig.savefig("figs/robustness.pgf")
# Fig 3: binary RoA maps drawn as filled vector regions (contourf at the 0/1 boundary)
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=True)
th, w = np.linspace(-np.pi, np.pi, 61), np.linspace(-8, 8, 61)
for a, n in zip(axs, NAMES):
    g = np.array(get(0,n)["roa"]).reshape(61, 61)
    a.contourf(th, w, g, levels=[-0.5, 0.5, 1.5], colors=["white", "#3a9a50"])
    a.set_xlim(-np.pi, np.pi); a.set_ylim(-8, 8)
    a.set_title(f"{n}: {100*g.mean():.0f}\\%", fontsize=7); a.set_xlabel(r"$\theta_0$ (rad)")
axs[0].set_ylabel(r"$\dot\theta_0$ (rad/s)"); fig.tight_layout(); fig.savefig("figs/roa.pgf")
