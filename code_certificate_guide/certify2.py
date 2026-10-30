"""Sound Lyapunov-decrease certification, v2: natural interval extension AND mean-value form
(interval forward-mode AD), outward rounding, explicit 'inconclusive' status on budget exhaustion."""
import numpy as np
from common import P, K, EXPS, DT, UMAX, WMAX, G, M, L
from distill import C, ALPHA
A_, B_ = 3*G/(2*L), 3.0/(M*L**2); INF = np.inf; EPS = 4e-16
p00, p01, p11 = P[0,0], P[0,1], P[1,1]
class IA:
    __slots__ = ("lo", "hi")
    def __init__(s, lo, hi=None): s.lo = np.asarray(lo, float); s.hi = s.lo if hi is None else np.asarray(hi, float)
    def r(s): return IA(np.nextafter(s.lo, -INF), np.nextafter(s.hi, INF))
    def __add__(s, o): o = o if isinstance(o, IA) else IA(o); return IA(s.lo+o.lo, s.hi+o.hi).r()
    __radd__ = __add__
    def __neg__(s): return IA(-s.hi, -s.lo)
    def __sub__(s, o): o = o if isinstance(o, IA) else IA(o); return IA(s.lo-o.hi, s.hi-o.lo).r()
    def __mul__(s, o):
        if not isinstance(o, IA):
            k = float(o); return IA(k*s.lo, k*s.hi).r() if k >= 0 else IA(k*s.hi, k*s.lo).r()
        p = np.stack([s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi]); return IA(p.min(0), p.max(0)).r()
    __rmul__ = __mul__
    def pw(s, n):
        if n == 0: return IA(np.ones_like(s.lo))
        a, b = s.lo**n, s.hi**n
        if n % 2: return IA(a, b).r()
        m = np.where((s.lo <= 0) & (s.hi >= 0), 0.0, np.minimum(np.abs(s.lo), np.abs(s.hi))**n); return IA(m, np.maximum(a, b)).r()
    def sin(s): assert np.all(np.abs(s.lo) < 1.5) and np.all(np.abs(s.hi) < 1.5); return IA(np.sin(s.lo)-EPS, np.sin(s.hi)+EPS)
    def cos(s):
        assert np.all(np.abs(s.lo) < 1.5) and np.all(np.abs(s.hi) < 1.5); a, b = np.cos(s.lo), np.cos(s.hi)
        return IA(np.minimum(a, b)-EPS, np.where((s.lo <= 0) & (s.hi >= 0), 1.0, np.maximum(a, b))+EPS)
    def hull0(s): return IA(np.minimum(s.lo, 0), np.maximum(s.hi, 0))
class AD:  # value interval + gradient intervals w.r.t. (theta, omega)
    __slots__ = ("v", "g")
    def __init__(s, v, g): s.v, s.g = v, g
    def __add__(s, o):
        if isinstance(o, AD): return AD(s.v+o.v, [a+b for a, b in zip(s.g, o.g)])
        return AD(s.v+o, s.g)
    __radd__ = __add__
    def __sub__(s, o): return AD(s.v-o.v, [a-b for a, b in zip(s.g, o.g)]) if isinstance(o, AD) else AD(s.v-o, s.g)
    def __mul__(s, o):
        if isinstance(o, AD): return AD(s.v*o.v, [s.v*b + o.v*a for a, b in zip(s.g, o.g)])
        return AD(s.v*o, [a*o for a in s.g])
    __rmul__ = __mul__
    def pw(s, n):
        if n == 0: return AD(IA(np.ones_like(s.v.lo)), [IA(np.zeros_like(s.v.lo))]*2)
        d = s.v.pw(n-1)*n; return AD(s.v.pw(n), [d*a for a in s.g])
    def sin(s): c = s.v.cos(); return AD(s.v.sin(), [c*a for a in s.g])
    def cos(s): sn = -s.v.sin(); return AD(s.v.cos(), [sn*a for a in s.g])
    def clip(s, lo, hi):
        v = IA(np.clip(s.v.lo, lo, hi), np.clip(s.v.hi, lo, hi)); inside = (s.v.lo >= lo) & (s.v.hi <= hi); out_ = (s.v.hi <= lo) | (s.v.lo >= hi)
        def gfix(a):
            h = a.hull0(); return IA(np.where(inside, a.lo, np.where(out_, 0.0, h.lo)), np.where(inside, a.hi, np.where(out_, 0.0, h.hi)))
        return AD(v, [gfix(a) for a in s.g])
def closed_loop_delta(ctrl, th, om):
    """Works for IA (natural extension) or AD (gradients). Returns (Delta, V1)."""
    if isinstance(ctrl, str): u = (th*(-K[0,0]) + om*(-K[0,1]))
    else:
        c, s, z = th.cos(), th.sin(), om*(1/WMAX); u = None
        for wi, e in zip(ctrl, EXPS):
            if wi == 0: continue
            t = c.pw(e[0])*s.pw(e[1])*z.pw(e[2])*float(wi); u = t if u is None else u + t
    u = clipX(u, -UMAX, UMAX)
    om2 = clipX(om + (th.sin()*A_ + u*B_)*DT, -WMAX, WMAX); th2 = th + om2*DT
    V2 = th2.pw(2)*p00 + th2*om2*(2*p01) + om2.pw(2)*p11; V1 = th.pw(2)*p00 + th*om*(2*p01) + om.pw(2)*p11
    return V2 - V1*(1-ALPHA), V1
def clipX(x, lo, hi): return x.clip(lo, hi) if isinstance(x, AD) else IA(np.clip(x.lo, lo, hi), np.clip(x.hi, lo, hi))
def delta_upper(ctrl, t0, t1, w0, w1):
    # natural extension
    dN, V1 = closed_loop_delta(ctrl, IA(t0, t1), IA(w0, w1))
    # mean-value form: D(X) in D(c) + sum_i G_i (X_i - c_i)
    tc, wc = (t0+t1)/2, (w0+w1)/2; rt, rw = np.nextafter(np.maximum(t1-tc, tc-t0), INF), np.nextafter(np.maximum(w1-wc, wc-w0), INF)
    dc, _ = closed_loop_delta(ctrl, IA(tc), IA(wc))
    one, zero = IA(np.ones_like(t0)), IA(np.zeros_like(t0))
    dA, _ = closed_loop_delta(ctrl, AD(IA(t0, t1), [one, zero]), AD(IA(w0, w1), [zero, one]))
    gt, gw = dA.g; mv = dc.hi + np.maximum(np.abs(gt.lo), np.abs(gt.hi))*rt + np.maximum(np.abs(gw.lo), np.abs(gw.hi))*rw
    return np.minimum(dN.hi, np.nextafter(mv, INF)), V1
def delta_pt(ctrl, th, om):
    from certify import delta_pt as dp; return dp(ctrl, th, om)
def certify(ctrl, c_top=C, c_min_frac=1e-4, min_w=1e-9, max_boxes=6_000_000, init=48):
    c_min = c_min_frac*c_top; det = p00*p11 - p01**2
    tmax, wmax = np.sqrt(c_top*p11/det)*1.001, np.sqrt(c_top*p00/det)*1.001
    e1, e2 = np.linspace(-tmax, tmax, init+1), np.linspace(-wmax, wmax, init+1)
    T0, W0 = np.meshgrid(e1[:-1], e2[:-1]); T1, W1 = np.meshgrid(e1[1:], e2[1:])
    Bx = np.stack([T0.ravel(), T1.ravel(), W0.ravel(), W1.ravel()], 1)
    best, n, nver, cex, status = c_top, 0, 0, [], "complete"
    while len(Bx):
        if n + len(Bx) > max_boxes: status = "inconclusive (box budget)"; best = min(best, float(closed_loop_delta(ctrl, IA(Bx[:,0], Bx[:,1]), IA(Bx[:,2], Bx[:,3]))[1].lo.min())); break
        dh, V1 = delta_upper(ctrl, Bx[:,0], Bx[:,1], Bx[:,2], Bx[:,3]); n += len(Bx)
        keep = (V1.lo < best) & (V1.hi >= c_min); ver = keep & (dh < 0); nver += int(ver.sum()); m = keep & ~ver
        Bx, Vlo = Bx[m], V1.lo[m]
        if not len(Bx): break
        tc, wc = Bx[:,:2].mean(1), Bx[:,2:].mean(1); dp, Vp = delta_pt(ctrl, tc, wc); bad = (dp > 0) & (Vp >= c_min) & (Vp <= c_top)
        if bad.any(): best = min(best, float(Vp[bad].min())); cex.append(np.stack([tc[bad], wc[bad], Vp[bad]], 1))
        m = ~bad & (Vlo < best); Bx, Vlo = Bx[m], Vlo[m]
        sm = np.maximum((Bx[:,1]-Bx[:,0])/tmax, (Bx[:,3]-Bx[:,2])/wmax) < min_w
        if sm.any(): status = "inconclusive (min box size)"; best = min(best, float(Vlo[sm].min())); Bx = Bx[~sm]
        sp = (Bx[:,1]-Bx[:,0])/tmax >= (Bx[:,3]-Bx[:,2])/wmax; mt, mw = Bx[:,:2].mean(1), Bx[:,2:].mean(1)
        a = Bx[sp].copy(); b = Bx[sp].copy(); a[:,1] = mt[sp]; b[:,0] = mt[sp]; c_ = Bx[~sp].copy(); d_ = Bx[~sp].copy(); c_[:,3] = mw[~sp]; d_[:,2] = mw[~sp]
        Bx = np.concatenate([a, b, c_, d_])
    cex = np.concatenate(cex) if cex else np.zeros((0, 3))
    if best <= c_min: best = 0.0
    return dict(c_cert=best, frac=best/c_top, status=status, boxes=n, verified=nver, cex=cex)
