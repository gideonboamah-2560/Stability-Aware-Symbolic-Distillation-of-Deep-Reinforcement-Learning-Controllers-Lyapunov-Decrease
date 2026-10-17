import numpy as np, json
from common import *
th, thd = np.meshgrid(np.linspace(-np.pi, np.pi, 801), np.linspace(-WMAX, WMAX, 801))
th, thd = th.ravel(), thd.ravel()
u = -(K[0,0]*wrap(th) + K[0,1]*thd)
th2, thd2 = step_np(th, thd, u)
V0, V1 = V_np(th, thd), V_np(th2, thd2)
ok = (V1 < V0) | (V0 < 1e-9)
bad = np.sort(V0[~ok])
c_star = bad[0] if len(bad) else V0.max()
c = 0.99*c_star                       # largest sublevel set free of violations on the grid
inD = (V0 <= c) & (V0 > 1e-6)
rho = np.max(V1[inD]/V0[inD])
alpha = 0.5*(1-rho)
print(f"c*={c_star:.4f}, c={c:.4f}, grid pts in D={inD.sum()}, max ratio rho={rho:.4f}, alpha={alpha:.4f}")
print("theta extent of D (rad):", np.abs(wrap(th[inD])).max(), " thd extent:", np.abs(thd[inD]).max())
json.dump(dict(c=float(c), rho=float(rho), alpha=float(alpha)), open("results/region.json","w"))
