import numpy as np
from scipy import stats
import json

K, M, LAMBDA = 9, 3, 2.5

def A(theta):
    return np.exp(theta) - 1

def F_SP(i):
    if i < 0: return 0.0
    return 1 - (1 - A(LAMBDA*(i+1)/(K+1))/A(LAMBDA))**M

def F_PS(i):
    if i < 0: return 0.0
    return (1 - A(LAMBDA*(K-i)/(K+1))/A(LAMBDA))**M

rows = []
for i in range(-1, K+1):
    FSP = F_SP(i)
    FPS = F_PS(i)
    RSP = 1 - FSP
    RPS = 1 - FPS
    rows.append((i, FSP, FPS, RSP, RPS))

print(f"{'i':>3} {'F_SP':>8} {'F_PS':>8} {'R_SP':>8} {'R_PS':>8}")
for r in rows:
    print(f"{r[0]:>3} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>8.4f} {r[4]:>8.4f}")

# pmf and hazard
p_SP = np.array([F_SP(i)-F_SP(i-1) for i in range(K+1)])
p_PS = np.array([F_PS(i)-F_PS(i-1) for i in range(K+1)])
p_SP = p_SP/p_SP.sum()
p_PS = p_PS/p_PS.sum()

def hazard(p):
    Fc = np.cumsum(p)
    Fprev = np.concatenate(([0], Fc[:-1]))
    return p/(1-Fprev)

h_SP = hazard(p_SP)
h_PS = hazard(p_PS)
print("\ni, p_SP, p_PS, h_SP, h_PS")
for i in range(K+1):
    print(f"{i}: {p_SP[i]:.4f} {p_PS[i]:.4f} {h_SP[i]:.4f} {h_PS[i]:.4f}")

def moments(p, support):
    mean = np.sum(support*p)
    var = np.sum((support-mean)**2*p)
    sd = np.sqrt(var)
    skew = np.sum((support-mean)**3*p)/sd**3
    kurt = np.sum((support-mean)**4*p)/sd**4 - 3
    ent = -np.sum(p[p>0]*np.log2(p[p>0]))
    return mean, var, skew, kurt, ent

support = np.arange(K+1)
mSP = moments(p_SP, support)
mPS = moments(p_PS, support)
print("\nS-P moments (mean,var,skew,kurt,entropy):", [round(x,4) for x in mSP])
print("P-S moments:", [round(x,4) for x in mPS])

# n_min via CLT
z = stats.norm.ppf(0.98)  # 96% two-sided CI -> alpha/2=0.02
eps = 0.01
sigma_SP = np.sqrt(mSP[1])
sigma_PS = np.sqrt(mPS[1])
n_SP = (z*sigma_SP/eps)**2
n_PS = (z*sigma_PS/eps)**2
n_min = max(n_SP, n_PS)
print(f"\nz(0.98)={z:.4f}, sigma_SP={sigma_SP:.4f}, sigma_PS={sigma_PS:.4f}, eps={eps}")
print(f"n_SP={n_SP:.1f} -> ceil {np.ceil(n_SP):.0f}")
print(f"n_PS={n_PS:.1f} -> ceil {np.ceil(n_PS):.0f}")
print(f"n_min={n_min:.1f} -> ceil {np.ceil(n_min):.0f}")

json.dump({
    "F_SP": [F_SP(i) for i in range(K+1)],
    "F_PS": [F_PS(i) for i in range(K+1)],
    "p_SP": p_SP.tolist(), "p_PS": p_PS.tolist(),
    "h_SP": h_SP.tolist(), "h_PS": h_PS.tolist(),
    "moments_SP": mSP, "moments_PS": mPS,
    "n_SP": float(np.ceil(n_SP)), "n_PS": float(np.ceil(n_PS)), "n_min": float(np.ceil(n_min)),
    "z": float(z), "eps": eps
}, open("final_values.json","w"), indent=2)