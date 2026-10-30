"""Certificate-guided symbolic distillation: DAgger data from the stability-aware student, then refits with a
scale-invariant ratio penalty + exact linearised (LMI) term, adding verifier counterexamples each round."""
import numpy as np, torch, json, sys, time
from stable_baselines3 import SAC
from common import *; from common import A as AM, B as BM
from sim import rollout, init_states, teacher_fn, success
import distill as D
from certify import lmi_ok
from certify2 import certify
torch.set_num_threads(1)
IDX = {n: i for i, n in enumerate(NAMES)}
At, Bt, Pt_ = (torch.tensor(m, dtype=torch.float64) for m in (AM, BM, P))
def lin_gains(w):  # exact linearisation at upright (c=1, s=theta, w=thd/8)
    return w[IDX['s']] + w[IDX['c*s']] + w[IDX['c^2*s']], (w[IDX['w']] + w[IDX['c*w']] + w[IDX['c^2*w']]) / WMAX
def lmi_max(w):
    kt, kw = lin_gains(w); Kr = torch.stack([kt, kw]).reshape(1, 2)
    Acl = At + Bt @ Kr; Mx = Acl.T @ Pt_ @ Acl - (1 - D.ALPHA) * Pt_
    tr, det = Mx[0,0] + Mx[1,1], Mx[0,0]*Mx[1,1] - Mx[0,1]*Mx[1,0]
    return tr/2 + torch.sqrt(torch.clamp((tr/2)**2 - det, min=0))
def fit_cert(S, A, Dst, lam_r=10.0, lam_lmi=0.0, lam1=1e-2, steps=800, prune=0.02, seed=0, margin=1e-3):
    torch.manual_seed(seed); f64 = torch.float64
    th, thd = torch.tensor(S[:,0], dtype=f64), torch.tensor(S[:,1], dtype=f64); Ft, Aa = feats_torch(th, thd), torch.tensor(A, dtype=f64)
    dth, dw = torch.tensor(Dst[:,0], dtype=f64), torch.tensor(Dst[:,1], dtype=f64); FD = feats_torch(dth, dw)
    VD = D.V_torch(dth, dw).double()
    def loss(w, l1):
        imit = ((torch.clamp(Ft@w, -UMAX, UMAX) - Aa)**2).mean()
        u = torch.clamp(FD@w, -UMAX, UMAX); w2 = torch.clamp(dw + (3*G/(2*L)*torch.sin(dth) + 3.0/(M*L**2)*u)*DT, -WMAX, WMAX); t2 = dth + w2*DT
        V2 = Pt_[0,0]*t2*t2 + 2*Pt_[0,1]*t2*w2 + Pt_[1,1]*w2*w2
        ratio = torch.relu(V2/VD - (1 - D.ALPHA)).mean()                 # scale-invariant violation
        return imit + lam_r*ratio + lam_lmi*torch.relu(lmi_max(w) + margin) + l1*w.abs().sum()
    w = torch.zeros(len(EXPS), dtype=f64, requires_grad=True); opt = torch.optim.Adam([w], lr=0.02)
    for _ in range(steps): opt.zero_grad(); loss(w, lam1).backward(); opt.step()
    mask = (w.detach().abs() > prune).double(); w2 = (w.detach()*mask).clone().requires_grad_(True); opt = torch.optim.Adam([w2], lr=0.01)
    for _ in range(steps): opt.zero_grad(); loss(w2*mask, 0.0).backward(); opt.step()
    return (w2.detach()*mask).numpy()
def dagger_data(T, seed):   # reproduces experiment.py's stability-aware DAgger run and keeps its data
    rng = np.random.default_rng(1000+seed); Dst = D.sample_D(4000, rng); S, A, st = [], [], None
    for it in range(10):
        th0, thd0 = rng.uniform(-np.pi, np.pi, 20), rng.uniform(-1, 1, 20); out = rollout(T if it == 0 else st, th0, thd0, rng=rng)
        th, thd = out["th"].ravel(), out["thd"].ravel(); S.append(np.stack([th, thd], 1)); A.append(T(np.stack([np.cos(th), np.sin(th), thd], 1)))
        st = D.Symbolic(D.fit_symbolic(np.concatenate(S), np.concatenate(A), 10.0, 1e-2, Dst, seed=seed, steps=800))
    return np.concatenate(S), np.concatenate(A), Dst, st.w
def run(seed, rounds=5, log=print, lam_r=10.0, target=0.95):
    T = teacher_fn(SAC.load(f"models/sac_seed{seed}")); sel = json.load(open("results/select.json"))
    import os
    cache = f"results/dagger_s{seed}.npz"
    if os.path.exists(cache): z = np.load(cache); S, A, Dst, w_st = z["S"], z["A"], z["D"], z["w"]
    else: S, A, Dst, w_st = dagger_data(T, seed); np.savez(cache, S=S, A=A, D=Dst, w=w_st)
    log(f"seed {seed}: DAgger reproduces stored stable student: {np.allclose(w_st, sel[f'{seed}/sym/10.0']['w'], atol=1e-6)}")
    rng = np.random.default_rng(7+seed); hist = []
    for r in range(rounds):
        w = fit_cert(S, A, Dst, lam_r=lam_r, seed=seed); cert = certify(w); ok, lm = lmi_ok(*[float(v) for v in lin_gains(torch.tensor(w))])
        vth, vw = init_states(100, 777); o = rollout(D.Symbolic(w), vth, vw)
        hist.append(dict(round=r, frac=cert["frac"], cex=len(cert["cex"]), lmi=lm, nnz=int((w != 0).sum()), val_ret=float(o["ret"].mean()), val_succ=float(success(o).mean()), w=w.tolist()))
        log(f"  round {r}: certified {100*cert['frac']:.2f}% of D | LMI max-eig {lm:+.5f} | cex {len(cert['cex'])} | terms {hist[-1]['nnz']} | val ret {hist[-1]['val_ret']:.1f} succ {hist[-1]['val_succ']:.2f}")
        if cert["frac"] >= target or len(cert["cex"]) == 0: break
        cx = cert["cex"][:, :2]; jit = np.concatenate([cx + rng.normal(0, 1, cx.shape) * np.array([2e-3, 2e-2]) for _ in range(20)])
        jit = jit[(V_np(jit[:,0], jit[:,1]) <= D.C) & (V_np(jit[:,0], jit[:,1]) > 1e-9)]
        Dst = np.concatenate([Dst, cx, jit])
    return hist
if __name__ == "__main__":
    s, lam = int(sys.argv[1]), float(sys.argv[2]); t = time.time(); print(f"== seed {s}, lam_r {lam}")
    h = run(s, lam_r=lam); json.dump(h, open(f"results/cegis_s{s}_lam{lam:g}.json", "w")); print(f"done in {time.time()-t:.0f}s", flush=True)
