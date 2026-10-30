import json, numpy as np
r = json.load(open("results/report.json")); t = r["table"]; R = json.load(open("results/test.json"))
f = lambda m, s, k=1: f"{m:.{k}f}$\\pm${s:.{k}f}".replace("-", "$-$")
L = ["\\begin{table*}[t]\\centering\\caption{Test-set results (mean$\\pm$std over five teacher seeds; 100 test episodes per seed). Violations: fraction of the 65{,}010 grid points in $\\mathcal{D}$ violating \\eqref{eq:dec}. RoA: fraction of a $61\\times61$ grid of initial states stabilised within 300 steps. $|\\Delta u|$: mean absolute change of torque per step. Robust success: success averaged over all 22 conditions.}\\label{tab:main}",
     "\\begin{tabular}{lccccccc}\\toprule",
     "Controller & Size & Return & Success (\\%) & Violations (\\%) & RoA (\\%) & $|\\Delta u|$ & Robust success (\\%)\\\\\\midrule"]
for n, d in t.items():
    size = "MLP 2$\\times$256" if n.startswith("Teacher") else (f"{R['LEAF']} leaves" if n.startswith("Tree") else f"{d['nnz'][0]:.1f} terms")
    L.append(f"{n} & {size} & {f(*d['ret'])} & {f(100*d['succ'][0],100*d['succ'][1])} & {f(100*d['viol'][0],100*d['viol'][1])} & {f(100*d['roa'][0],100*d['roa'][1])} & {f(*d['du'],2)} & {f(100*d['allsucc'][0],100*d['allsucc'][1])}\\\\")
L += ["\\bottomrule\\end{tabular}\\end{table*}"]
open("paper/table_main.tex","w").write("\n".join(L))
def ptex(p):
    m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$" if p < 1e-3 else f"${p:.3f}$"
S = ["\\begin{table}[t]\\centering\\caption{Paired Wilcoxon signed-rank tests over all (seed, condition) pairs ($n=110$). $\\Delta$: mean difference (first minus second); W/L: pairs won/lost by the first.}\\label{tab:stats}",
     "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llccc}\\toprule Comparison & Metric & $\\Delta$ & W/L & $p$\\\\\\midrule"]
for k, v in r["stats"].items():
    comp, met = k.split(" [")[0].replace(" (128 leaves)","").replace(" (SAC)","").replace("Symbolic-","Sym-").replace(" vs ", " vs.\\ "), k.split("[")[1][:-1]
    d = v["mean_diff"]*(100 if met=="succ" else 1)
    S.append(f"{comp} & {'success (pp)' if met=='succ' else 'return'} & ${d:+.2f}$ & {v['wins']}/{v['losses']} & {ptex(v['p'])}\\\\")
S += ["\\bottomrule\\end{tabular}\\end{table}"]
open("paper/table_stats.tex","w").write("\n".join(S)); print("tables written")
