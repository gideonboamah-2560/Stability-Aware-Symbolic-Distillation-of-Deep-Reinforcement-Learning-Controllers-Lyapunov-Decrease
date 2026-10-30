"""Certifies the saturated LQR law and the plain / stability-aware students of every seed (Table 2)."""
import numpy as np, json, torch
import certify2 as C2; from certify import lmi_ok; from cegis import lin_gains
sel = json.load(open("results/select.json")); out = {}
r = C2.certify("lqr"); out["LQR"] = dict(frac=r["frac"], status=r["status"], cex=len(r["cex"]))
for s in range(5):
    for lab, key in [("Symbolic-plain", "0.0"), ("Symbolic-stable", "10.0")]:
        w = np.array(sel[f"{s}/sym/{key}"]["w"]); r = C2.certify(w); ok, lm = lmi_ok(*[float(v) for v in lin_gains(torch.tensor(w))])
        out[f"{s}/{lab}"] = dict(frac=r["frac"], status=r["status"], cex=len(r["cex"]), lmi=lm, lmi_ok=ok, boxes=r["boxes"])
json.dump(out, open("results/cert_baselines.json", "w"), indent=1); print("saved results/cert_baselines.json")
