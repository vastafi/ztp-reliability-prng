import numpy as np
from scipy import stats
import json

K, M, LAMBDA = 9, 3, 2.5
N_TRIALS = 236
N_PER_TRIAL = 1000

SOURCES = {
    "Java_ThreadLocalRandom": [
        "data/java_threadlocal/threadlocal_data.csv",
        "data/java_threadlocal/threadlocal_data1.csv",
        "data/java_threadlocal/threadlocal_data2.csv",
        "data/java_threadlocal/threadlocal_data3.csv",
        "data/java_threadlocal/threadlocal_data4.csv",
        "data/java_threadlocal/threadlocal_data5.csv",
    ],
    "Python_secrets": [
        "data/python_secrets/secrets_data.csv",
        "data/python_secrets/secrets_data1.csv",
        "data/python_secrets/secrets_data2.csv",
        "data/python_secrets/secrets_data3.csv",
        "data/python_secrets/secrets_data4.csv",
        "data/python_secrets/secrets_data5.csv",
    ],
    "Pi_digits": [
        "data/pi/pi_digits_part.csv",
        "data/pi/pi_digits_part1.csv",
        "data/pi/pi_digits_part2.csv",
        "data/pi/pi_digits_part3.csv",
        "data/pi/pi_digits_part4.csv",
        "data/pi/pi_digits_part5.csv",
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
        self.d = digits; self.i = 0; self.n = len(digits)
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

def ztp_via_rejection(lam, stream):
    while True:
        n = knuth_poisson(lam, stream)
        if n > 0: return n

def simulate_trial(stream, n_replicates):
    U = np.empty(n_replicates, dtype=np.int64)
    V = np.empty(n_replicates, dtype=np.int64)
    for r in range(n_replicates):
        sp_life = np.empty(M); ps_life = np.empty(M)
        for k in range(M):
            nk = ztp_via_rejection(LAMBDA, stream)
            lifetimes = np.array([stream.next_digit() for _ in range(nk)])
            sp_life[k] = lifetimes.max()
            ps_life[k] = lifetimes.min()
        U[r] = sp_life.min()
        V[r] = ps_life.max()
    return U, V

# ---------- theoretical values ----------
def A(theta): return np.exp(theta) - 1
def F_SP(i):
    if i < 0: return 0.0
    return 1 - (1 - A(LAMBDA*(i+1)/(K+1))/A(LAMBDA))**M
def F_PS(i):
    if i < 0: return 0.0
    return (1 - A(LAMBDA*(K-i)/(K+1))/A(LAMBDA))**M

support = np.arange(K+1)
p_SP = np.clip(np.array([F_SP(i)-F_SP(i-1) for i in range(K+1)]), 0, None); p_SP /= p_SP.sum()
p_PS = np.clip(np.array([F_PS(i)-F_PS(i-1) for i in range(K+1)]), 0, None); p_PS /= p_PS.sum()
F_SP_vals = np.cumsum(p_SP); F_PS_vals = np.cumsum(p_PS)

def hazard(p):
    Fc = np.cumsum(p); Fprev = np.concatenate(([0], Fc[:-1]))
    return p / (1-Fprev)
h_SP_theory = hazard(p_SP); h_PS_theory = hazard(p_PS)

def moments(p, supp):
    mean = np.sum(supp*p); var = np.sum((supp-mean)**2*p); sd = np.sqrt(var)
    As = np.sum((supp-mean)**3*p)/sd**3
    Exc = np.sum((supp-mean)**4*p)/sd**4 - 3
    H = -np.sum(p[p>0]*np.log2(p[p>0]))
    return dict(mean=mean, var=var, As=As, Exc=Exc, H=H)

theory_SP = moments(p_SP, support); theory_PS = moments(p_PS, support)

def lag1_autocorr(x):
    x = x.astype(float); xm = x - x.mean()
    return np.sum(xm[:-1]*xm[1:]) / np.sum(xm**2)

def chi2_independence(x, K):
    n_cat = K+1; obs = np.zeros((n_cat, n_cat))
    for a, b in zip(x[:-1], x[1:]): obs[a, b] += 1
    rs = obs.sum(axis=1, keepdims=True); cs = obs.sum(axis=0, keepdims=True); N = obs.sum()
    exp = rs @ cs / N
    mask = exp > 0
    stat = np.sum((obs[mask]-exp[mask])**2/exp[mask])
    dof = (n_cat-1)**2
    return 1 - stats.chi2.cdf(stat, dof)

def trial_deviations(U, V, n):
    out = {}
    for net, X, theory, p_theory, F_theory, h_theory in [
        ("U", U, theory_SP, p_SP, F_SP_vals, h_SP_theory),
        ("V", V, theory_PS, p_PS, F_PS_vals, h_PS_theory),
    ]:
        m = moments(np.array([np.mean(X==i) for i in range(K+1)]) if False else None, support) if False else None
        mean_e = X.mean(); var_e = X.var(ddof=1)
        As_e = stats.skew(X); Exc_e = stats.kurtosis(X)
        counts = np.bincount(X, minlength=K+1)[:K+1]
        p_e = counts / counts.sum()
        H_e = -np.sum(p_e[p_e>0]*np.log2(p_e[p_e>0]))
        h_e = hazard(p_e)
        valid = ~np.isnan(h_e) & ~np.isnan(h_theory)
        d_hazard = np.max(np.abs(h_e[valid]-h_theory[valid]))
        d_autocorr = abs(lag1_autocorr(X))
        chi2_stat, chi2_p = stats.chisquare(f_obs=counts, f_exp=p_theory*n)
        ecdf = np.array([np.mean(X<=i) for i in range(K+1)])
        d_ks = np.max(np.abs(ecdf - F_theory))
        p_ind = chi2_independence(X, K)

        out[net] = dict(
            mean=abs(mean_e-theory["mean"])/K,
            var=min(abs(var_e-theory["var"])/theory["var"],1.0),
            As=min(abs(As_e-theory["As"]),1.0),
            Exc=min(abs(Exc_e-theory["Exc"])/(abs(theory["Exc"])+1e-6),1.0),
            entropy=min(abs(H_e-theory["H"])/theory["H"],1.0),
            hazard=min(d_hazard,1.0),
            autocorr=min(d_autocorr,1.0),
            chi2=1-chi2_p,
            ks=d_ks,
            independence=1-p_ind,
        )
    return out

GROUP_7 = ['mean','entropy','autocorr','chi2','independence','ks','hazard']
GROUP_3 = ['var','As','Exc']
WEIGHTS = {c: 9/70 for c in GROUP_7}; WEIGHTS.update({c: 1/30 for c in GROUP_3})

results_all = {}
for name, files in SOURCES.items():
    digits = load_digits(files)
    stream = DigitStream(digits)
    trial_devs_U = []; trial_devs_V = []
    for t in range(N_TRIALS):
        U, V = simulate_trial(stream, N_PER_TRIAL)
        d = trial_deviations(U, V, N_PER_TRIAL)
        trial_devs_U.append(d["U"]); trial_devs_V.append(d["V"])
    digits_used = stream.i
    print(f"{name}: {N_TRIALS} trials x {N_PER_TRIAL} = digits used {digits_used:,} / {len(digits):,} ({100*digits_used/len(digits):.1f}%)")

    avg_U = {k: np.mean([d[k] for d in trial_devs_U]) for k in trial_devs_U[0]}
    std_U = {k: np.std([d[k] for d in trial_devs_U]) for k in trial_devs_U[0]}
    avg_V = {k: np.mean([d[k] for d in trial_devs_V]) for k in trial_devs_V[0]}
    std_V = {k: np.std([d[k] for d in trial_devs_V]) for k in trial_devs_V[0]}

    crqi_s_U = 1 - np.mean(list(avg_U.values())); crqi_w_U = 1 - sum(WEIGHTS[k]*v for k,v in avg_U.items())
    crqi_s_V = 1 - np.mean(list(avg_V.values())); crqi_w_V = 1 - sum(WEIGHTS[k]*v for k,v in avg_V.items())

    # trial-level CRQI std (for error bars)
    crqi_w_trials_U = [1 - sum(WEIGHTS[k]*d[k] for k in d) for d in trial_devs_U]
    crqi_w_trials_V = [1 - sum(WEIGHTS[k]*d[k] for k in d) for d in trial_devs_V]
    crqi_w_std_U = np.std(crqi_w_trials_U); crqi_w_std_V = np.std(crqi_w_trials_V)

    results_all[name] = dict(
        avg_U=avg_U, std_U=std_U, avg_V=avg_V, std_V=std_V,
        crqi_simple_U=crqi_s_U, crqi_weighted_U=crqi_w_U,
        crqi_simple_V=crqi_s_V, crqi_weighted_V=crqi_w_V,
        crqi_weighted_std_U=crqi_w_std_U, crqi_weighted_std_V=crqi_w_std_V,
        crqi_simple_overall=(crqi_s_U+crqi_s_V)/2,
        crqi_weighted_overall=(crqi_w_U+crqi_w_V)/2,
    )
    print(f"  CRQI_simple: U={crqi_s_U:.4f} V={crqi_s_V:.4f} | CRQI_weighted: U={crqi_w_U:.4f}±{crqi_w_std_U:.4f} V={crqi_w_V:.4f}±{crqi_w_std_V:.4f}")

print("\n=== RANKING (overall CRQI weighted, descending) ===")
ranking = sorted(results_all.items(), key=lambda kv: kv[1]["crqi_weighted_overall"], reverse=True)
for rank, (name, v) in enumerate(ranking, 1):
    print(f"{rank}. {name}: simple={v['crqi_simple_overall']:.4f}, weighted={v['crqi_weighted_overall']:.4f}")

with open("crqi_repeated_trials.json", "w") as f:
    json.dump(results_all, f, indent=2, default=float)
print("\nSaved crqi_repeated_trials.json")
