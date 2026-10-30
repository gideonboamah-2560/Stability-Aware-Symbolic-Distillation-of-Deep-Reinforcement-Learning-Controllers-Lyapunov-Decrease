import numpy as np, json, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from cegis import lin_gains; from certify import lmi_ok
plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 200})
R = json.load(open("results/test2.json")); CB = json.load(open("results/cert_baselines.json")); S = range(5); CONDS = R["conds"]
NM = ["Teacher (SAC)", "Tree (128 leaves)", "Symbolic-plain", "Symbolic-stable", "Symbolic-certified"]
COL = dict(zip(NM, ["k", "tab:orange", "tab:blue", "tab:green", "tab:purple"]))
g = lambda s, n: R["data"][f"{s}/{n}"]; rep = {}
cert = {n: [CB[f"{s}/{n}"]["frac"] for s in S] for n in ["Symbolic-plain", "Symbolic-stable"]}; cert["Symbolic-certified"] = [g(s, "Symbolic-certified")["cert_frac"] for s in S]
lmi = {}
for n in ["Symbolic-plain", "Symbolic-stable", "Symbolic-certified"]:
    ks = [[float(v) for v in lin_gains(torch.tensor(np.array([float(x.split("*")[0]) for x in []]) if False else torch.tensor(0.)))] for _ in []]
tab = {}
for n in NM:
    f = lambda key, c="nominal": [np.mean(g(s, n)["cond"][c][key]) for s in S]
    row = dict(ret=f("ret"), succ=f("succ"), viol=[g(s, n)["viol"] for s in S], du=[g(s, n)["cond"]["nominal"]["du"] for s in S],
               roa=[np.mean(g(s, n)["roa"]) for s in S], allsucc=[np.mean([np.mean(g(s, n)["cond"][c]["succ"]) for c in CONDS]) for s in S])
    tab[n] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in row.items()}
    tab[n]["nnz"] = None if g(0, n)["nnz"] is None else (float(np.mean([g(s, n)["nnz"] for s in S])), float(np.std([g(s, n)["nnz"] for s in S])))
    tab[n]["cert"] = (float(np.mean(cert[n])), float(np.std(cert[n]))) if n in cert else None
rep["table"] = tab; rep["cert_per_seed"] = cert; rep["lqr_cert"] = CB["LQR"]["frac"]
# gains + LMI from weights (expr strings carry weights; use stored weights)
H = {s: json.load(open(f"results/cegis_s{s}_lam{R['LAM_R']:g}.json")) for s in S}; sel = json.load(open("results/select.json"))
W = {"Symbolic-plain": [sel[f"{s}/sym/0.0"]["w"] for s in S], "Symbolic-stable": [sel[f"{s}/sym/10.0"]["w"] for s in S], "Symbolic-certified": [H[s][-1]["w"] for s in S]}
rep["gains"] = {}; rep["lmi"] = {}
for n, ws in W.items():
    kk = np.array([[float(v) for v in lin_gains(torch.tensor(np.array(w)))] for w in ws]); rep["gains"][n] = [kk.mean(0).tolist(), kk.std(0).tolist()]
    rep["lmi"][n] = [lmi_ok(*k)[1] for k in kk]
rep["rounds"] = [len(H[s]) for s in S]; rep["cegis_hist"] = {s: [dict(round=h["round"], frac=h["frac"], val_succ=h["val_succ"], val_ret=h["val_ret"]) for h in H[s]] for s in S}
rep["lamr"] = R["lamr_summary"]; rep["LAM_R"] = R["LAM_R"]
# per-condition
rep["percond"] = {n: {c: (float(np.mean([np.mean(g(s, n)["cond"][c]["succ"]) for s in S])), float(np.mean([np.mean(g(s, n)["cond"][c]["ret"]) for s in S]))) for c in CONDS} for n in NM}
# stats
def per(n, key): return np.array([np.mean(g(s, n)["cond"][c][key]) for s in S for c in CONDS])
st = {}
for a, b in [("Symbolic-certified", "Symbolic-stable"), ("Symbolic-certified", "Symbolic-plain"), ("Symbolic-certified", "Teacher (SAC)"), ("Symbolic-certified", "Tree (128 leaves)"), ("Symbolic-stable", "Symbolic-plain")]:
    for key in ["ret", "succ"]:
        x, y = per(a, key), per(b, key); d = x - y; p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        st[f"{a} vs {b} [{key}]"] = dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=float(p))
rep["stats"] = st; rep["exprs0"] = {n: g(0, n)["expr"] for n in ["Symbolic-plain", "Symbolic-stable", "Symbolic-certified"]}
json.dump(rep, open("results/report2.json", "w"), indent=1)
# ---- figures ----
L_ = [1, 3, 10, 30, 100]; ls = rep["lamr"]
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.6)); x = np.arange(5)
ax[0].plot(x, [100*ls[str(l)]["frac"] for l in L_], "o-", c="tab:purple", ms=3); ax[0].set_ylabel("certified share of $\\mathcal{D}$ (%)")
ax[1].plot(x, [ls[str(l)]["ret"] for l in L_], "s-", c="tab:blue", ms=3); ax[1].set_ylabel("validation return")
for a in ax: a.set_xticks(x); a.set_xticklabels([str(l) for l in L_], fontsize=6); a.set_xlabel("$\\lambda_r$")
fig.tight_layout(); fig.savefig("figs2/lamr_sweep.pdf")
groups = [("mass x", "mass scale", [1.0]), ("length x", "length scale", [1.0]), ("noise ", "obs. noise std", [0.0]), ("umax ", "torque limit", [2.0]), ("push ", "push (rad/s)", [0.0])]
fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.6), sharey=True)
for a, (pre, lab, nx) in zip(axs, groups):
    cs = [c for c in CONDS if c.startswith(pre)]; xs = nx + [float(c[len(pre):]) for c in cs]; cs = ["nominal"] + cs; o = np.argsort(xs); xs = np.array(xs)[o]; cs = np.array(cs)[o]
    for n in NM:
        m = np.array([[np.mean(g(s, n)["cond"][c]["succ"]) for c in cs] for s in S]); a.errorbar(xs, 100*m.mean(0), 100*m.std(0), c=COL[n], marker="o", ms=2, lw=.9, capsize=1.5, label=n)
    a.set_xlabel(lab, fontsize=7)
axs[0].set_ylabel("success (%)"); axs[0].legend(fontsize=4.5, loc="lower left"); fig.tight_layout(); fig.savefig("figs2/robustness.pdf")
fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.6), sharey=True)
for a, n in zip(axs, NM):
    gg = np.array(g(0, n)["roa"]).reshape(61, 61); a.imshow(gg, origin="lower", extent=[-np.pi, np.pi, -8, 8], aspect="auto", cmap="Greens", vmin=0, vmax=1.3)
    a.set_title(f"{n.split(' (')[0]}: {100*gg.mean():.0f}%", fontsize=6.5); a.set_xlabel("$\\theta_0$ (rad)", fontsize=7)
axs[0].set_ylabel("$\\dot\\theta_0$ (rad/s)"); fig.tight_layout(); fig.savefig("figs2/roa.pdf")
print(json.dumps({n: {k: (None if v is None else [round(x, 4) for x in v]) for k, v in d.items()} for n, d in tab.items()}, indent=0)[:3000])
print("gains", rep["gains"]); print("lmi", {k: [round(x, 5) for x in v] for k, v in rep["lmi"].items()}); print("rounds", rep["rounds"])
for k, v in st.items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
