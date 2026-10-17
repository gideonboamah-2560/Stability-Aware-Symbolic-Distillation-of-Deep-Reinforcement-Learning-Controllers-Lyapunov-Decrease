"""Shared definitions: exact Pendulum-v1 dynamics (verified against Gymnasium 1.2.x source),
LQR-based Lyapunov candidate, student policy class."""
import numpy as np, torch
from scipy.linalg import solve_discrete_are

G, M, L, DT, UMAX, WMAX = 10.0, 1.0, 1.0, 0.05, 2.0, 8.0

def wrap(th):  # identical to gymnasium angle_normalize
    return ((th + np.pi) % (2*np.pi)) - np.pi

def step_np(th, thd, u, g=G, m=M, l=L, umax=UMAX):
    u = np.clip(u, -umax, umax)
    thd2 = np.clip(thd + (3*g/(2*l)*np.sin(th) + 3.0/(m*l**2)*u)*DT, -WMAX, WMAX)
    return th + thd2*DT, thd2

def step_torch(th, thd, u):
    u = torch.clamp(u, -UMAX, UMAX)
    thd2 = torch.clamp(thd + (3*G/(2*L)*torch.sin(th) + 3.0/(M*L**2)*u)*DT, -WMAX, WMAX)
    return th + thd2*DT, thd2

# Linearization of the semi-implicit Euler map about the upright equilibrium (th=0, thd=0)
a, b = 3*G/(2*L), 3.0/(M*L**2)
A = np.array([[1 + DT*DT*a, DT], [DT*a, 1.0]])
B = np.array([[DT*DT*b], [DT*b]])
Q = np.diag([1.0, 0.1]); R = np.array([[0.001]])   # same weights as the env cost
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T@P@B, B.T@P@A)          # u = -K x

def V_np(th, thd):
    x0, x1 = wrap(th), thd
    return P[0,0]*x0*x0 + 2*P[0,1]*x0*x1 + P[1,1]*x1*x1

Pt = torch.tensor(P, dtype=torch.float32)
def V_torch(th, thd):
    x0 = torch.remainder(th + np.pi, 2*np.pi) - np.pi
    return Pt[0,0]*x0*x0 + 2*Pt[0,1]*x0*thd + Pt[1,1]*thd*thd

# ---- symbolic student: sparse combination of monomials (deg<=3) of (cos th, sin th, thd/8) ----
import itertools
EXPS = [e for d in range(1,4) for e in itertools.product(range(4), repeat=3) if sum(e)==d]
EXPS = sorted(set(EXPS), key=lambda e:(sum(e), e[::-1]))
# keep only odd features (odd total degree in sin, thd): the dynamics and cost are invariant under
# (th,thd,u)->(-th,-thd,-u), so the optimal policy is odd; even terms cannot belong to it.
EXPS = [e for e in EXPS if (e[1]+e[2]) % 2 == 1]
NAMES = []
for e in EXPS:
    parts=[]
    for v,p in zip(["c","s","w"], e):
        if p==1: parts.append(v)
        elif p>1: parts.append(f"{v}^{p}")
    NAMES.append("*".join(parts))

def feats_np(obs):  # obs: (N,3) = cos, sin, thdot
    z = np.stack([obs[:,0], obs[:,1], obs[:,2]/WMAX], 1)
    return np.stack([np.prod(z**np.array(e),1) for e in EXPS], 1)

def feats_torch(th, thd):
    z = [torch.cos(th), torch.sin(th), thd/WMAX]
    return torch.stack([z[0]**e[0]*z[1]**e[1]*z[2]**e[2] for e in EXPS], 1)
