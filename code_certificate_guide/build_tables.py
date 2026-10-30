import json, numpy as np
r = json.load(open("/home/claude/proj/results/report2.json")); t = r["table"]; R = json.load(open("/home/claude/proj/results/test2.json"))
pm = lambda m, s, k=1: (f"{m:.{k}f}$\\pm${s:.{k}f}").replace("-", "$-$")
def pct(v, k=1): return pm(100*v[0], 100*v[1], k)
NM = ["Teacher (SAC)", "Tree (128 leaves)", "Symbolic-plain", "Symbolic-stable", "Symbolic-certified"]
size = {"Teacher (SAC)": "MLP 2$\\times$256", "Tree (128 leaves)": "128 leaves"}
T1 = ["\\begin{table}[H]\\centering\\caption{Test-set results (mean$\\pm$std over five teacher seeds; 100 test episodes per seed). Grid viol.: share of 65{,}010 grid points in $\\mathcal{D}$ violating \\eqref{eq:dec}. Certified: share of $\\mathcal{D}$ (by level) on which \\eqref{eq:dec} is formally proved (Section~\\ref{sec:cert}). RoA: share of a $61\\times61$ grid of initial states stabilised within 300 steps. Robust: success averaged over all 22 conditions.}\\label{tab:main}",
      "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccccc}\\hline", "Controller & Size & Return & Success (\\%) & Grid viol. (\\%) & Certified (\\%) & RoA (\\%) & $|\\Delta u|$ & Robust (\\%)\\\\\\hline"]
for n in NM:
    d = t[n]; sz = size[n] if n in size else f"{d['nnz'][0]:.1f} terms"; ce = "n/a" if d["cert"] is None else ("$\\approx$0" if n == "Symbolic-plain" else pct(d["cert"]))
    T1.append(f"{n} & {sz} & {pm(*d['ret'])} & {pct(d['succ'])} & {pct(d['viol'], 2 if n=='Symbolic-certified' else 1)} & {ce} & {pct(d['roa'])} & {pm(*d['du'], 2)} & {pct(d['allsucc'])}\\\\")
T1 += ["\\hline\\end{tabular}}\\end{table}"]
cb = json.load(open("/home/claude/proj/results/cert_baselines.json")); cs = r["cert_per_seed"]; lm = r["lmi"]
T2 = ["\\begin{table}[H]\\centering\\caption{Formal certification per teacher seed. $\\mu$: largest eigenvalue of $A_{cl}^\\top P A_{cl}-(1-\\alpha)P$ at the linearisation (negative = linearised decrease holds). Cert.: certified share of $\\mathcal{D}$; Rounds: verifier calls (initial fit plus counterexample rounds) until the stopping rule. The saturated LQR law is certified on " + f"{100*r['lqr_cert']:.1f}" + "\\% of $\\mathcal{D}$.}\\label{tab:cert}",
      "\\begin{tabular}{c|cc|cc|ccc}\\hline", " & \\multicolumn{2}{c|}{Symbolic-plain} & \\multicolumn{2}{c|}{Symbolic-stable} & \\multicolumn{3}{c}{Symbolic-certified}\\\\", "Seed & $\\mu$ & Cert. (\\%) & $\\mu$ & Cert. (\\%) & $\\mu$ & Cert. (\\%) & Rounds\\\\\\hline"]
f = lambda x: f"${x:+.5f}$"
for s in range(5):
    T2.append(f"{s} & {f(lm['Symbolic-plain'][s])} & $\\approx$0 & {f(lm['Symbolic-stable'][s])} & {100*cs['Symbolic-stable'][s]:.1f} & {f(lm['Symbolic-certified'][s])} & {100*cs['Symbolic-certified'][s]:.1f} & {r['rounds'][s]}\\\\")
T2.append(f"Mean & & $\\approx$0 & & {100*np.mean(cs['Symbolic-stable']):.1f} & & {100*np.mean(cs['Symbolic-certified']):.1f} & {np.mean(r['rounds']):.1f}\\\\\\hline\\end{{tabular}}\\end{{table}}")
def ptex(p):
    if p < 1e-3: m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
    return f"${p:.3f}$"
T3 = ["\\begin{table}[H]\\centering\\caption{Paired Wilcoxon signed-rank tests over all (seed, condition) pairs ($n=110$). $\\Delta$: mean difference (first minus second); W/L: pairs won/lost by the first.}\\label{tab:stats}",
      "\\begin{tabular}{llccc}\\hline Comparison & Metric & $\\Delta$ & W/L & $p$\\\\\\hline"]
for k, v in r["stats"].items():
    comp, met = k.split(" [")[0].replace("Symbolic-", "Sym-").replace(" (128 leaves)", "").replace(" (SAC)", "").replace(" vs ", " vs.\\ "), k.split("[")[1][:-1]
    d = v["mean_diff"]*(100 if met == "succ" else 1); T3.append(f"{comp} & {'success (pp)' if met=='succ' else 'return'} & ${d:+.1f}$ & {v['wins']}/{v['losses']} & {ptex(v['p'])}\\\\")
T3 += ["\\hline\\end{tabular}\\end{table}"]
ls = r["lamr"]
T4 = ["\\begin{table}[H]\\centering\\caption{Selection of $\\lambda_r$ on validation data (means over five teachers, final counterexample round). Rule: largest certified share subject to mean validation success $\\ge95\\%$.}\\label{tab:lamr}",
      "\\begin{tabular}{cccccc}\\hline $\\lambda_r$ & Certified (\\%) & Val. success (\\%) & Val. return & Terms & Rounds\\\\\\hline"]
for l in [1, 3, 10, 30, 100]:
    d = ls[str(l)]; T4.append(f"{l}{' (selected)' if l == r['LAM_R'] else ''} & {100*d['frac']:.1f} & {100*d['succ']:.1f} & ${d['ret']:.1f}$ & {d['nnz']:.1f} & {d['rounds']:.1f}\\\\".replace("$-", "$-"))
T4 += ["\\hline\\end{tabular}\\end{table}"]
json.dump(dict(T1="\n".join(T1), T2="\n".join(T2), T3="\n".join(T3), T4="\n".join(T4)), open("tables.json", "w"))
print("\n".join(T2)); print("\n".join(T4))
