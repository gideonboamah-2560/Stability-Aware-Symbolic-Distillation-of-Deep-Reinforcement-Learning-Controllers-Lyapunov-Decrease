"""Sound certification of the Lyapunov decrease condition V(f(x,pi(x))) <= (1-alpha) V(x)
for symbolic controllers, by branch-and-bound interval arithmetic with outward rounding."""
import numpy as np, json
from common import P, K, EXPS, DT, UMAX, WMAX, G, M, L
from distill import C, ALPHA
A_, B_ = 3*G/(2*L), 3.0/(M*L**2)
INF = np.inf
def out(lo, hi):                       # outward rounding: widen by one ulp each side
    return np.nextafter(lo, -INF), np.nextafter(hi, INF)
def iadd(a, b): return out(a[0]+b[0], a[1]+b[1])
def isub(a, b): return out(a[0]-b[1], a[1]-b[0])
def iscale(k, a): return out(*( (k*a[0], k*a[1]) if k >= 0 else (k*a[1], k*a[0]) ))
def imul(a, b):
    p = np.stack([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]); return out(p.min(0), p.max(0))
def ipow(a, n):                        # exact range of x^n on an interval
    if n == 0: return np.ones_like(a[0]), np.ones_like(a[1])
    lo, hi = a[0]**n, a[1]**n
    if n % 2: return out(lo, hi)
    m = np.minimum(np.abs(a[0]), np.abs(a[1])) ; m = np.where((a[0] <= 0) & (a[1] >= 0), 0.0, m**n)
    return out(m, np.maximum(lo, hi))
EPS = 4e-16                            # libm sin/cos error allowance (well above the < 1 ulp of glibc)
def isin(a):                           # sin monotone on [-pi/2, pi/2]; states here satisfy |theta| < 0.6
    assert np.all(np.abs(a[0]) < 1.5) and np.all(np.abs(a[1]) < 1.5)
    return np.sin(a[0]) - EPS, np.sin(a[1]) + EPS
def icos(a):
    assert np.all(np.abs(a[0]) < 1.5) and np.all(np.abs(a[1]) < 1.5)
    c0, c1 = np.cos(a[0]), np.cos(a[1]); hi = np.where((a[0] <= 0) & (a[1] >= 0), 1.0, np.maximum(c0, c1))
    return np.minimum(c0, c1) - EPS, hi + EPS
def iclip(a, lo, hi): return np.clip(a[0], lo, hi), np.clip(a[1], lo, hi)
def isq(a): return ipow(a, 2)
p00, p01, p11 = P[0,0], P[0,1], P[1,1]; R1 = p01/p00; Q1 = p11 - p01**2/p00
def iV(th, w):                         # V = p00 (th + R1 w)^2 + Q1 w^2  (each variable once per square)
    return iadd(iscale(p00, isq(iadd(th, iscale(R1, w)))), iscale(Q1, isq(w)))
def iu(w_coef, th, om):                # symbolic controller on intervals
    c, s, z = icos(th), isin(th), iscale(1/WMAX, om); acc = (np.zeros_like(th[0]), np.zeros_like(th[0]))
    for wi, e in zip(w_coef, EXPS):
        if wi == 0: continue
        term = imul(imul(ipow(c, e[0]), ipow(s, e[1])), ipow(z, e[2])); acc = iadd(acc, iscale(wi, term))
    return iclip(acc, -UMAX, UMAX)
def delta_upper(ctrl, th, om):
    """Upper bound of V(x+) - (1-alpha) V(x) over boxes; ctrl = weights or ('lqr',)."""
    u = iclip(isub(iscale(-K[0,0], th), iscale(K[0,1], om)), -UMAX, UMAX) if isinstance(ctrl, str) else iu(ctrl, th, om)
    om2 = iclip(iadd(om, iscale(DT, iadd(iscale(A_, isin(th)), iscale(B_, u)))), -WMAX, WMAX)
    # V(x+) with th2 = th + DT*om2:  p00 (th + (DT+R1) om2)^2 + Q1 om2^2
    V2 = iadd(iscale(p00, isq(iadd(th, iscale(DT + R1, om2)))), iscale(Q1, isq(om2)))
    V1 = iV(th, om)
    return isub(V2, iscale(1-ALPHA, V1))[1], V1
# ---- point (float) evaluation, used to confirm counterexamples ----
def u_pt(ctrl, th, om):
    if isinstance(ctrl, str): return np.clip(-(K[0,0]*th + K[0,1]*om), -UMAX, UMAX)
    c, s, z = np.cos(th), np.sin(th), om/WMAX
    return np.clip(sum(wi*c**e[0]*s**e[1]*z**e[2] for wi, e in zip(ctrl, EXPS)), -UMAX, UMAX)
def delta_pt(ctrl, th, om):
    om2 = np.clip(om + DT*(A_*np.sin(th) + B_*u_pt(ctrl, th, om)), -WMAX, WMAX); th2 = th + DT*om2
    V = lambda a, b: p00*a*a + 2*p01*a*b + p11*b*b
    return V(th2, om2) - (1-ALPHA)*V(th, om), V(th, om)

def certify(ctrl, c_top=C, c_min_frac=1e-4, min_w=1e-7, max_boxes=4_000_000, init=48):
    """Largest c* <= c_top such that the decrease condition provably holds on c_min <= V <= c*."""
    c_min = c_min_frac*c_top; det = p00*p11 - p01**2
    tmax, wmax = np.sqrt(c_top*p11/det)*1.001, np.sqrt(c_top*p00/det)*1.001
    e1, e2 = np.linspace(-tmax, tmax, init+1), np.linspace(-wmax, wmax, init+1)
    T0, W0 = np.meshgrid(e1[:-1], e2[:-1]); T1, W1 = np.meshgrid(e1[1:], e2[1:])
    q = [np.stack([T0.ravel(), T1.ravel(), W0.ravel(), W1.ravel()], 1)]
    best, n_done, n_ver, n_undec, cex = c_top, 0, 0, 0, []
    while q:
        Bx = np.concatenate(q); q = []
        if len(Bx) == 0: break
        d_hi, V1 = delta_upper(ctrl, (Bx[:,0], Bx[:,1]), (Bx[:,2], Bx[:,3])); n_done += len(Bx)
        keep = (V1[0] < best) & (V1[1] >= c_min)            # intersects the annulus below current best
        ver = keep & (d_hi < 0); n_ver += ver.sum(); Bx, d_hi, V1 = Bx[keep & ~ver], d_hi[keep & ~ver], (V1[0][keep & ~ver], V1[1][keep & ~ver])
        if len(Bx) == 0: break
        tc, wc = Bx[:,[0,1]].mean(1), Bx[:,[2,3]].mean(1); dp, Vp = delta_pt(ctrl, tc, wc)
        bad = (dp > 0) & (Vp >= c_min) & (Vp <= c_top)
        if bad.any():
            best = min(best, Vp[bad].min()); cex.append(np.stack([tc[bad], wc[bad], Vp[bad]], 1))
        rest = ~bad & (V1[0] < best)
        Bx, V1lo = Bx[rest], V1[0][rest]
        small = np.maximum((Bx[:,1]-Bx[:,0])/tmax, (Bx[:,3]-Bx[:,2])/wmax) < min_w
        if small.any(): best = min(best, V1lo[small].min()); n_undec += small.sum(); Bx = Bx[~small]
        if n_done > max_boxes: best = min(best, V1lo.min() if len(V1lo) else best); break
        wt, ww = (Bx[:,1]-Bx[:,0])/tmax, (Bx[:,3]-Bx[:,2])/wmax; sp = wt >= ww
        mt, mw = Bx[:,[0,1]].mean(1), Bx[:,[2,3]].mean(1)
        a = Bx[sp].copy(); b = Bx[sp].copy(); a[:,1] = mt[sp]; b[:,0] = mt[sp]
        c_ = Bx[~sp].copy(); d_ = Bx[~sp].copy(); c_[:,3] = mw[~sp]; d_[:,2] = mw[~sp]
        q = [a, b, c_, d_]
    cex = np.concatenate(cex) if cex else np.zeros((0, 3))
    return dict(c_cert=float(best), frac=float(best/c_top), c_min=float(c_min), boxes=int(n_done), verified=int(n_ver), undecided=int(n_undec), cex=cex)

def lmi_ok(kth, kw, alpha=ALPHA):
    """Linearised test at the origin: A_cl^T P A_cl - (1-alpha) P negative definite?"""
    from common import A, B
    Acl = A + B @ np.array([[kth, kw]]); Mx = Acl.T @ P @ Acl - (1-alpha)*P
    return bool(np.max(np.linalg.eigvalsh((Mx+Mx.T)/2)) < 0), float(np.max(np.linalg.eigvalsh((Mx+Mx.T)/2)))
