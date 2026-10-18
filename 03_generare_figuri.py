"""
Generarea figurilor reprezentative pentru articol.
Rulați DUPĂ 02_simulare_si_crqi_final.py (are nevoie de sim_results.json
NU e generat de acel script direct -- vezi nota de mai jos).

Acest script produce:
  Figura 1: PMF teoretică vs. empirică (U și V), pentru cele 3 surse PRNG
  Figura 2: CDF teoretică vs. empirică (U și V)
  Figura 3: Rata hazardului teoretică vs. empirică (U și V)

Necesită: numpy, matplotlib
    pip install numpy matplotlib

NOTĂ: acest script re-simulează un singur set (n=5000) per sursă PRNG,
doar pentru a avea date empirice de afișat grafic -- graficele NU sunt
identice cu rezultatele numerice din Tabelul 2 (care provin din media pe
100 de seturi de câte 1000), ci sunt ilustrative pentru forma generală a
distribuțiilor.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('/Users/astafivalentina/PycharmProjects/Poisson_trunc/figure', exist_ok=True)

K, M, LAMBDA = 9, 3, 2.5
N_ILLUSTRATION = 5000  # sample size just for plotting (not the paper's official n)



SOURCES = {
    "Java_ThreadLocalRandom": [
        "data/java_threadlocal/threadlocal_data.csv",
        "data/java_threadlocal/threadlocal_data1.csv",
    ],
    "Python_secrets": [
        "data/python_secrets/secrets_data.csv",
        "data/python_secrets/secrets_data1.csv",
    ],
    "Pi_digits": [
        "data/pi/pi_digits_part.csv",
        "data/pi/pi_digits_part1.csv",
    ],
}

def load_digits(files):
    digits = []
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    digits.append(int(line))
    return np.array(digits, dtype=np.int8)

class DigitStream:
    def __init__(self, digits):
        self.d = digits; self.i = 0
    def next_digit(self):
        v = self.d[self.i]; self.i += 1; return v
    def next_uniform(self, ndigits=5):
        val = 0
        for _ in range(ndigits):
            val = val * 10 + int(self.next_digit())
        return val / (10 ** ndigits)

def knuth_poisson(lam, stream):
    L = np.exp(-lam); k = 0; p = 1.0
    while True:
        k += 1; p *= stream.next_uniform()
        if p <= L: return k - 1

def ztp(lam, stream):
    while True:
        n = knuth_poisson(lam, stream)
        if n > 0: return n

def simulate(stream, n):
    U = np.empty(n, dtype=np.int64); V = np.empty(n, dtype=np.int64)
    for r in range(n):
        sp = np.empty(M); ps = np.empty(M)
        for k in range(M):
            nk = ztp(LAMBDA, stream)
            life = np.array([stream.next_digit() for _ in range(nk)])
            sp[k] = life.max(); ps[k] = life.min()
        U[r] = sp.min(); V[r] = ps.max()
    return U, V

def A(theta): return np.exp(theta) - 1
def F_SP(i):
    if i < 0: return 0.0
    return 1 - (1 - A(LAMBDA*(i+1)/(K+1))/A(LAMBDA))**M
def F_PS(i):
    if i < 0: return 0.0
    return (1 - A(LAMBDA*(K-i)/(K+1))/A(LAMBDA))**M

support = np.arange(K+1)
p_SP = np.clip(np.array([F_SP(i)-F_SP(i-1) for i in range(K+1)]),0,None); p_SP/=p_SP.sum()
p_PS = np.clip(np.array([F_PS(i)-F_PS(i-1) for i in range(K+1)]),0,None); p_PS/=p_PS.sum()
F_SP_v = np.cumsum(p_SP); F_PS_v = np.cumsum(p_PS)

def hazard(p):
    Fc = np.cumsum(p); Fprev = np.concatenate(([0], Fc[:-1]))
    return p/(1-Fprev)
h_SP_v = hazard(p_SP); h_PS_v = hazard(p_PS)

# ---------- Run illustrative simulation for one source (fastest: Python_secrets) ----------
# You can loop over all three sources if you want all three on one figure.
results = {}
for name, files in SOURCES.items():
    digits = load_digits(files)
    stream = DigitStream(digits)
    U, V = simulate(stream, N_ILLUSTRATION)
    results[name] = (U, V)

# ---------- Figure 1: PMF theoretical vs empirical (both networks) ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
for col, (name, (U, V)) in enumerate(results.items()):
    p_U_emp = np.bincount(U, minlength=K+1)[:K+1] / len(U)
    p_V_emp = np.bincount(V, minlength=K+1)[:K+1] / len(V)
    width = 0.35
    ax = axes[0, col]
    ax.bar(support - width/2, p_SP, width, label='Teoretic', color='steelblue', alpha=0.8)
    ax.bar(support + width/2, p_U_emp, width, label='Empiric', color='darkorange', alpha=0.8)
    ax.set_title(name.replace('_', ' ') + ' (S-P, U)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('p(i)')
        ax.legend()
    ax = axes[1, col]
    ax.bar(support - width/2, p_PS, width, label='Teoretic', color='steelblue', alpha=0.8)
    ax.bar(support + width/2, p_V_emp, width, label='Empiric', color='darkorange', alpha=0.8)
    ax.set_title(name.replace('_', ' ') + ' (P-S, V)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('p(i)')
plt.suptitle('Figura 1. PMF teoretică vs. empirică pentru U (S-P, sus) şi V (P-S, jos)')
plt.tight_layout()
plt.savefig('/Users/astafivalentina/PycharmProjects/Poisson_trunc/figure/figura1_pmf.png', dpi=150)
plt.close()

# ---------- Figure 2: CDF theoretical vs empirical (both networks) ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
for col, (name, (U, V)) in enumerate(results.items()):
    ecdf_U = np.array([np.mean(U <= i) for i in range(K+1)])
    ecdf_V = np.array([np.mean(V <= i) for i in range(K+1)])
    ax = axes[0, col]
    ax.step(support, F_SP_v, where='post', label='Teoretic', color='steelblue', linewidth=2)
    ax.step(support, ecdf_U, where='post', label='Empiric', color='darkorange', linestyle='--', linewidth=2)
    ax.set_title(name.replace('_', ' ') + ' (S-P, U)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('F(i)')
        ax.legend()
    ax = axes[1, col]
    ax.step(support, F_PS_v, where='post', label='Teoretic', color='steelblue', linewidth=2)
    ax.step(support, ecdf_V, where='post', label='Empiric', color='darkorange', linestyle='--', linewidth=2)
    ax.set_title(name.replace('_', ' ') + ' (P-S, V)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('F(i)')
plt.suptitle('Figura 2. CDF teoretică vs. empirică pentru U (S-P, sus) şi V (P-S, jos)')
plt.tight_layout()
plt.savefig('/Users/astafivalentina/PycharmProjects/Poisson_trunc/figure/figura2_cdf.png', dpi=150)
plt.close()

# ---------- Figure 3: Hazard rate theoretical vs empirical (both networks) ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
for col, (name, (U, V)) in enumerate(results.items()):
    p_U_emp = np.bincount(U, minlength=K+1)[:K+1] / len(U)
    p_V_emp = np.bincount(V, minlength=K+1)[:K+1] / len(V)
    h_U_emp = hazard(p_U_emp)
    h_V_emp = hazard(p_V_emp)
    ax = axes[0, col]
    ax.plot(support, h_SP_v, 'o-', label='Teoretic', color='steelblue')
    ax.plot(support, h_U_emp, 's--', label='Empiric', color='darkorange')
    ax.set_title(name.replace('_', ' ') + ' (S-P, U)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('h(i)')
        ax.legend()
    ax = axes[1, col]
    ax.plot(support, h_PS_v, 'o-', label='Teoretic', color='steelblue')
    ax.plot(support, h_V_emp, 's--', label='Empiric', color='darkorange')
    ax.set_title(name.replace('_', ' ') + ' (P-S, V)')
    ax.set_xlabel('i')
    if col == 0:
        ax.set_ylabel('h(i)')
plt.suptitle('Figura 3. Rata hazardului teoretică vs. empirică pentru U (S-P, sus) şi V (P-S, jos)')
plt.tight_layout()
plt.savefig('/Users/astafivalentina/PycharmProjects/Poisson_trunc/figure/figura3_hazard.png', dpi=150)
plt.close()

print("Salvate: figura1_pmf.png, figura2_cdf.png, figura3_hazard.png")