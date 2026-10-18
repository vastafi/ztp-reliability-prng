import glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "/data"
M = 3
LAMBDA_TRUE = 2.5

def load_digits(pattern):
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, header=None).iloc[:,0].values for f in files]
    return np.concatenate(frames).astype(int)

def ztp_cdf_table(lam, k_max=60):
    ks = np.arange(1, k_max+1)
    unnorm = np.array([lam**k * math.exp(-lam) / math.factorial(k) for k in ks])
    unnorm /= (1 - math.exp(-lam))
    return ks, np.cumsum(unnorm)

ZTP_KS, ZTP_CDF = ztp_cdf_table(LAMBDA_TRUE)

def digits_to_uniform(block):
    return int("".join(str(d) for d in block)) / 1e9

def sample_ztp(u):
    idx = np.searchsorted(ZTP_CDF, u)
    return int(ZTP_KS[min(idx, len(ZTP_KS)-1)])

class Stream:
    def __init__(self, digits):
        self.d = digits; self.pos = 0
    def take(self, n):
        if self.pos+n > len(self.d): raise StopIteration
        out = self.d[self.pos:self.pos+n]; self.pos += n
        return out

def gen_realization(stream, M_=M):
    Tmax, Tmin = [], []
    for _ in range(M_):
        u = digits_to_uniform(stream.take(9))
        Nk = sample_ztp(u)
        xs = stream.take(Nk)
        Tmax.append(xs.max()); Tmin.append(xs.min())
    return min(Tmax), max(Tmin)

def gen_all(digits):
    s = Stream(digits)
    Us, Vs = [], []
    while True:
        try:
            u, v = gen_realization(s)
            Us.append(u); Vs.append(v)
        except StopIteration:
            break
    return np.array(Us), np.array(Vs)

def F_base(i): return (i+1)/10
def F_SP_theo(lam): 
    F = np.array([F_base(i) for i in range(10)])
    return 1 - (1 - (np.exp(lam*F)-1)/(np.exp(lam)-1))**M
def F_PS_theo(lam):
    F = np.array([F_base(i) for i in range(10)])
    return (1 - (np.exp(lam*(1-F))-1)/(np.exp(lam)-1))**M

def emp_cdf(sample):
    n = len(sample)
    return np.array([np.sum(sample<=k)/n for k in range(10)])

SOURCES = {
    "Java ThreadLocalRandom": (f"{DATA_DIR}/java_threadlocal/*.csv", 2.497045, '#2a78d6', 'o'),
    "Python secrets": (f"{DATA_DIR}/python_secrets/*.csv", 2.499783, '#1baf7a', 's'),
    "Cifrele lui π": (f"{DATA_DIR}/pi/*.csv", 2.501764, '#eda100', '^'),
}

results = {}
for name, (pattern, lam, color, marker) in SOURCES.items():
    digits = load_digits(pattern)
    Us, Vs = gen_all(digits)
    results[name] = dict(Us=Us, Vs=Vs, lam=lam, color=color, marker=marker)
    print(name, len(Us), "realizations")

k = np.arange(10)
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), gridspec_kw={'height_ratios':[3,1.3]}, sharex='col')

for col, (net_name, theo_func, sample_key) in enumerate([("S-P", F_SP_theo, "Us"), ("P-S", F_PS_theo, "Vs")]):
    ax_top, ax_bot = axes[0, col], axes[1, col]
    lam_ref = np.mean([v['lam'] for v in results.values()])
    F_theo = theo_func(lam_ref)
    ax_top.step(k, F_theo, where='post', color='black', linewidth=2, label='F teoretic', zorder=5)
    for name, r in results.items():
        F_emp = emp_cdf(r[sample_key])
        ax_top.plot(k, F_emp, r['marker'], color=r['color'], markersize=5,
                    markerfacecolor='none', markeredgewidth=1.5, label=name)
        dF = F_emp - F_theo
        ax_bot.plot(k, dF, r['marker']+'-', color=r['color'], markersize=4, linewidth=1)
    ax_top.set_title(f"Rețeaua {net_name}")
    ax_top.set_ylabel('F(k)')
    ax_top.spines[['top','right']].set_visible(False)
    ax_bot.axhline(0, color='#c3c2b7', linewidth=1)
    ax_bot.set_xlabel('k')
    ax_bot.set_ylabel(r'$\Delta F(k)$')
    ax_bot.spines[['top','right']].set_visible(False)
    ax_bot.set_ylim(-0.0015, 0.0015)

axes[0,0].legend(frameon=False, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('fig3_empirical_vs_theoretical.png', dpi=200, bbox_inches='tight')
print("saved fig3")
