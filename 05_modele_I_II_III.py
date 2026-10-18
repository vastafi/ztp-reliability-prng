"""
05_modele_I_II_III.py

Simulare Monte Carlo și calculul CRQI pentru cele trei modele de rețele S-P / P-S:
    Modelul I  : M = 3 fix,  N_k ~ ZTP(lambda)          (B = 236 seturi)
    Modelul II : N = 3 fix,  M ~ ZTP(mu)                 (B = 236 seturi)
    Modelul III: M ~ ZTP(mu), N_k ~ ZTP(lambda)          (B = 199 seturi, eps = 0.0121)
cu K = 9, lambda = mu = 2.5, n = 1000 replici / set.

Metoda de generare este identică cu 02_simulare_si_crqi_final.py: Poisson prin algoritmul
lui Knuth cu uniforme din 5 cifre consecutive, ZTP prin respingerea lui 0, duratele de viață
ale unităților = cifrele sursei. Pentru Modelul I scriptul reproduce exact valorile din
crqi_repeated_trials.json. Pentru fiecare model, simularea pornește de la începutul șirului
de cifre al fiecărei surse.

Utilizare (din directorul proiectului):
    python 05_modele_I_II_III.py            # toate cele trei modele
    python 05_modele_I_II_III.py II         # un singur model
Rezultate: results/rezultate_model_I.json, results/rezultate_model_II.json, results/rezultate_model_III.json
Figurile se generează apoi cu: python 06_figuri_modele_I_II_III.py
"""
import sys, glob, math, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

K = 9; LAM = 2.5; MU = 2.5; MFIX = 3; NFIX = 3
B_BY_MODEL = {'A': 236, 'B': 236, 'C': 199}
B_SETS = 236
NSET = 1000
W = np.array([9/70, 1/30, 1/30, 1/30, 9/70, 9/70, 9/70, 9/70, 9/70, 9/70])

def A(t): return math.expm1(t) if np.isscalar(t) else np.expm1(t)

def Fu(i): return (np.asarray(i) + 1) / (K + 1)

def cdfs(model):
    i = np.arange(K + 1); F = Fu(i)
    if model == 'A':
        Fsp = 1 - (1 - A(LAM*F)/A(LAM))**MFIX
        Fps = (1 - A(LAM*(1-F))/A(LAM))**MFIX
    elif model == 'B':
        Fsp = 1 - A(MU*(1 - F**NFIX))/A(MU)
        Fps = A(MU*(1 - (1-F)**NFIX))/A(MU)
    elif model == 'C':
        Fsp = 1 - A(MU*(1 - A(LAM*F)/A(LAM)))/A(MU)
        Fps = A(MU*(1 - A(LAM*(1-F))/A(LAM)))/A(MU)
    return Fsp, Fps

def theory(Fc):
    i = np.arange(K + 1)
    p = np.diff(np.r_[0.0, Fc]); m = (i*p).sum(); v = ((i-m)**2*p).sum()
    sk = ((i-m)**3*p).sum()/v**1.5; ex = ((i-m)**4*p).sum()/v**2 - 3
    H = -(p[p > 0]*np.log2(p[p > 0])).sum()
    R = 1 - Fc; Rprev = np.r_[1.0, R[:-1]]; h = p/Rprev
    return dict(p=p, F=Fc, mean=m, var=v, skew=sk, exc=ex, H=H, h=h)

def load(pattern):
    files = sorted(glob.glob(pattern))
    return np.concatenate([pd.read_csv(f, header=None).iloc[:, 0].values for f in files]).astype(int).tolist()

class Stream:
    def __init__(s, d): s.d = d; s.pos = 0; s.n = len(d)
    def take(s, k):
        if s.pos + k > s.n: raise StopIteration
        out = s.d[s.pos:s.pos+k]; s.pos += k; return out
    def unif(s):
        b = s.take(5); return (b[0]*10000 + b[1]*1000 + b[2]*100 + b[3]*10 + b[4]) / 100000

def ztp(s, lam):
    L = math.exp(-lam)
    while True:
        k = 0; p = 1.0
        while True:
            k += 1; p *= s.unif()
            if p <= L: break
        if k - 1 > 0: return k - 1

def replica(s, model, order):
    if model == 'A': M = MFIX
    else: M = ztp(s, MU)
    tmax = []; tmin = []
    if order == 'interleaved' or model == 'B':
        for _ in range(M):
            N = NFIX if model == 'B' else ztp(s, LAM)
            x = s.take(N); tmax.append(max(x)); tmin.append(min(x))
    else:  # all N_k first, then unit lifetimes
        Ns = [ztp(s, LAM) for _ in range(M)]
        for N in Ns:
            x = s.take(N); tmax.append(max(x)); tmin.append(min(x))
    return min(tmax), max(tmin)

def criteria(x, T):
    n = len(x); i = np.arange(K + 1)
    m = x.mean(); d = x - m; m2 = (d**2).mean()
    S2 = (d**2).sum()/(n-1)
    sk = (d**3).mean()/m2**1.5; ex = (d**4).mean()/m2**2 - 3
    O = np.bincount(x, minlength=K+1); ph = O/n
    H = -(ph[ph > 0]*np.log2(ph[ph > 0])).sum()
    rho = (d[:-1]*d[1:]).sum()/(d**2).sum()
    E = n*T['p']; chi = ((O-E)**2/E).sum(); pcon = stats.chi2.sf(chi, K)
    Fh = np.cumsum(ph); ks = np.abs(Fh - T['F']).max()
    tab = np.zeros((K+1, K+1)); np.add.at(tab, (x[:-1], x[1:]), 1)
    Er = np.outer(tab.sum(1), tab.sum(0))/(n-1); msk = Er > 0
    chii = ((tab[msk]-Er[msk])**2/Er[msk]).sum(); pind = stats.chi2.sf(chii, K*K)
    den = 1 - np.r_[0.0, Fh[:-1]]; ok = den > 1e-12
    hh = np.where(ok, ph/np.where(ok, den, 1), np.nan)
    dh = np.nanmax(np.abs(hh - T['h']))
    return np.array([abs(m-T['mean'])/K, min(abs(S2-T['var'])/T['var'], 1), min(abs(sk-T['skew']), 1),
                     min(abs(ex-T['exc'])/(abs(T['exc'])+1e-6), 1), min(abs(H-T['H'])/T['H'], 1), abs(rho),
                     1-pcon, ks, 1-pind, dh])

SOURCES = {'Java ThreadLocalRandom': 'java_threadlocal/*.csv', 'Python secrets': 'python_secrets/*.csv', 'Cifrele lui pi': 'pi/*.csv'}

def run(ddir, model):
    global B_SETS
    B_SETS = B_BY_MODEL[model]; order = 'interleaved'
    Fsp, Fps = cdfs(model); Tu, Tv = theory(Fsp), theory(Fps)
    out = {'model': ROMAN[model], 'theory_U': {k: v for k, v in Tu.items() if np.isscalar(v)},
           'theory_V': {k: v for k, v in Tv.items() if np.isscalar(v)}, 'sources': {}}
    for name, pat in SOURCES.items():
        s = Stream(load(f'{ddir}/{pat}')); U = []; V = []
        try:
            while len(U) < B_SETS*NSET:
                u, v = replica(s, model, order); U.append(u); V.append(v)
        except StopIteration: pass
        U = np.array(U); V = np.array(V); nb = len(U)//NSET
        res = {'replicas': len(U), 'digits_used': s.pos, 'digits_total': s.n, 'sets': nb}
        for lab, X, T in (('SP', U, Tu), ('PS', V, Tv)):
            D = np.array([criteria(X[b*NSET:(b+1)*NSET], T) for b in range(nb)])
            crqi_b = 1 - D @ W
            res[lab] = {'dbar': D.mean(0).tolist(), 'CRQI_simplu': float(1 - D.mean(0).mean()),
                        'CRQI_ponderat': float(1 - D.mean(0) @ W), 'sd_ponderat': float(crqi_b.std(ddof=0)),
                        'emp_mean': float(X.mean()), 'crqi_b': crqi_b.tolist(), 'emp_pmf': (np.bincount(X, minlength=K+1)/len(X)).tolist()}
        out['sources'][name] = res
        print(f"Modelul {ROMAN[model]} | {name}: {res['replicas']} replici, cifre {res['digits_used']:,}/{res['digits_total']:,} | "
              f"CRQI ponderat S-P {res['SP']['CRQI_ponderat']:.6f}, P-S {res['PS']['CRQI_ponderat']:.6f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f'rezultate_model_{ROMAN[model]}.json'
    json.dump(out, open(path, 'w'), indent=1, default=float)
    print(f'Salvat: {path}')

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / 'results'
ROMAN = {'A': 'I', 'B': 'II', 'C': 'III'}
if __name__ == '__main__':
    pick = {'I': 'A', 'II': 'B', 'III': 'C'}
    models = [pick[sys.argv[1]]] if len(sys.argv) > 1 else ['A', 'B', 'C']
    for m in models: run(str(BASE_DIR / 'data'), m)
