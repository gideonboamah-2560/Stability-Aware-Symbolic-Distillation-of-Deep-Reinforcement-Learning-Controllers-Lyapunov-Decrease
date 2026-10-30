"""DAgger distillation (Ross et al., 2011) into (i) CART trees and (ii) sparse symbolic policies
trained with an optional Lyapunov-decrease penalty on region D = {V <= c}."""
import numpy as np, torch, json
from sklearn.tree import DecisionTreeRegressor
from common import *
from sim import rollout, init_states
torch.set_num_threads(1)
REG = json.load(open("results/region.json")); C, ALPHA = REG["c"], REG["alpha"]

def sample_D(n, rng):
    """Uniform samples in the sublevel set {V<=c} by rejection from its bounding box."""
    out = []
    while sum(len(o) for o in out) < n:
        th = rng.uniform(-0.2, 0.2, 4*n); w = rng.uniform(-1.5, 1.5, 4*n)
        keep = V_np(th, w) <= C; out.append(np.stack([th[keep], w[keep]], 1))
    return np.concatenate(out)[:n]

def lyap_violation(th, thd, u):
    th2, thd2 = step_np(th, thd, u)
    return V_np(th2, thd2) > (1-ALPHA)*V_np(th, thd)

class Symbolic:
    def __init__(self, w): self.w = w
    def __call__(self, obs): return np.clip(feats_np(obs) @ self.w, -UMAX, UMAX)
    def nnz(self): return int((self.w != 0).sum())
    def expr(self, tol=0):
        return " + ".join(f"{v:+.3f}*{n}" for v, n in zip(self.w, NAMES) if abs(v) > tol)

def fit_symbolic(S, A, lam_stab, lam1, D_states, w0=None, steps=1500, prune=0.02, seed=0):
    """S: states (N,2) [th, thd]; A: teacher actions. Two stages: L1 fit, prune, masked refit."""
    torch.manual_seed(seed)
    th, thd = torch.tensor(S[:,0], dtype=torch.float32), torch.tensor(S[:,1], dtype=torch.float32)
    Ft, At = feats_torch(th, thd), torch.tensor(A, dtype=torch.float32)
    dth, dw = torch.tensor(D_states[:,0], dtype=torch.float32), torch.tensor(D_states[:,1], dtype=torch.float32)
    FD, VD = feats_torch(dth, dw), V_torch(dth, dw)
    w = torch.zeros(len(EXPS), requires_grad=True) if w0 is None else torch.tensor(w0, requires_grad=True, dtype=torch.float32)
    def loss_fn(w, l1):
        imit = ((torch.clamp(Ft@w, -UMAX, UMAX) - At)**2).mean()
        th2, w2 = step_torch(dth, dw, FD@w)
        lyap = torch.relu(V_torch(th2, w2) - (1-ALPHA)*VD).mean()/C
        return imit + lam_stab*lyap + l1*w.abs().sum(), imit.item(), lyap.item()
    opt = torch.optim.Adam([w], lr=0.02)
    for _ in range(steps):
        opt.zero_grad(); l,_,_ = loss_fn(w, lam1); l.backward(); opt.step()
    mask = (w.detach().abs() > prune).float()
    w2 = (w.detach()*mask).clone().requires_grad_(True); opt = torch.optim.Adam([w2], lr=0.01)
    for _ in range(steps):
        opt.zero_grad(); l,_,_ = loss_fn(w2*mask, 0.0); l.backward(); opt.step()
    wf = (w2.detach()*mask).numpy().astype(float)
    return wf

def dagger(teacher, kind, n_iter=10, n_eps=20, seed=0, **kw):
    rng = np.random.default_rng(1000+seed)
    D_states = sample_D(4000, rng)
    S, A, student, w = [], [], None, None
    for it in range(n_iter):
        th0, thd0 = rng.uniform(-np.pi, np.pi, n_eps), rng.uniform(-1, 1, n_eps)
        pol = teacher if it == 0 else student
        out = rollout(pol, th0, thd0, rng=rng)
        th, thd = out["th"].ravel(), out["thd"].ravel()
        obs = np.stack([np.cos(th), np.sin(th), thd], 1)
        S.append(np.stack([th, thd], 1)); A.append(teacher(obs))
        Sa, Aa = np.concatenate(S), np.concatenate(A)
        if kind == "tree":
            m = DecisionTreeRegressor(max_leaf_nodes=kw["leaves"], random_state=seed)
            m.fit(np.stack([np.cos(Sa[:,0]), np.sin(Sa[:,0]), Sa[:,1]], 1), Aa)
            student = (lambda m: (lambda o: m.predict(o)))(m); student.model = m
        else:
            w = fit_symbolic(Sa, Aa, kw["lam_stab"], kw["lam1"], D_states, seed=seed,
                             steps=kw.get("steps", 1500))
            student = Symbolic(w)
    return student
