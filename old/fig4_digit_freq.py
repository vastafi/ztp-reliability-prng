import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "/data"
SOURCES = {
    "Java ThreadLocalRandom": (f"{DATA_DIR}/java_threadlocal/*.csv", '#2a78d6'),
    "Python secrets": (f"{DATA_DIR}/python_secrets/*.csv", '#1baf7a'),
    "Cifrele lui π": (f"{DATA_DIR}/pi/*.csv", '#eda100'),
}
chi2_vals = {"Java ThreadLocalRandom": 13.7268, "Python secrets": 2.3697, "Cifrele lui π": 4.9008}

freqs = {}
for name, (pattern, color) in SOURCES.items():
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, header=None).iloc[:,0].values for f in files]
    digits = np.concatenate(frames)
    counts = np.bincount(digits, minlength=10)
    freqs[name] = counts / counts.sum()

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(10)
width = 0.26
for i, (name, (pattern, color)) in enumerate(SOURCES.items()):
    ax.bar(x + (i-1)*width, freqs[name], width, color=color,
           label=f"{name} (χ²={chi2_vals[name]})")
ax.axhline(0.1, color='#e34948', linestyle='--', linewidth=1.2, label='Teoretic (0,1)')
ax.set_xlabel('Cifra')
ax.set_ylabel('Frecvența relativă')
ax.set_title('Frecvența relativă a cifrelor 0-9 pentru cele trei surse PRNG')
ax.set_xticks(x)
ax.set_ylim(0.0985, 0.1018)
ax.legend(frameon=False, loc='upper right', fontsize=8, ncol=2)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('fig4_digit_freq.png', dpi=200, bbox_inches='tight')
print("saved fig4")
