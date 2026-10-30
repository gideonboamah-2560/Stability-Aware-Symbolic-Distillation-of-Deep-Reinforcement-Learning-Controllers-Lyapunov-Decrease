"""For each certified student: re-certifies, times it, and bounds V(x+) over the excluded ball {V < c_min}
(forward invariance of the certified region). Adds the results to results/report2.json."""
import numpy as np, json, time
import certify2 as C2; from distill import C
R = json.load(open("results/test2.json")); rep = json.load(open("results/report2.json"))
H = {s: json.load(open(f"results/cegis_s{s}_lam{R['LAM_R']:g}.json")) for s in range(5)}
det = C2.p00*C2.p11 - C2.p01**2; cmin = 1e-4*C; tm, wm = np.sqrt(cmin*C2.p11/det)*1.001, np.sqrt(cmin*C2.p00/det)*1.001; n = 400
e1, e2 = np.linspace(-tm, tm, n+1), np.linspace(-wm, wm, n+1); T0, W0 = np.meshgrid(e1[:-1], e2[:-1]); T1, W1 = np.meshgrid(e1[1:], e2[1:]); ball = {}
for s in range(5):
    w = np.array(H[s][-1]["w"]); t = time.time(); r = C2.certify(w); dt = time.time() - t
    D_, V1 = C2.closed_loop_delta(w, C2.IA(T0.ravel(), T1.ravel()), C2.IA(W0.ravel(), W1.ravel())); sup = float((D_ + V1*(1-C2.ALPHA)).hi[V1.lo < cmin].max())
    ball[str(s)] = dict(c_cert=r["c_cert"], frac=r["frac"], status=r["status"], boxes=r["boxes"], secs=dt, sup_V_next_ball=sup, invariant=bool(sup <= r["c_cert"]))
rep["ball"] = ball; rep["c_min"] = cmin; json.dump(rep, open("results/report2.json", "w"), indent=1); print("ball bounds added to results/report2.json")
