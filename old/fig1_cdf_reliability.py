import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

lam = 2.499530666667
M = 3

def F_base(i): return (i+1)/10
def F_SP(i): return 1 - (1 - (np.exp(lam*F_base(i))-1)/(np.exp(lam)-1))**M
def F_PS(i): return (1 - (np.exp(lam*(1-F_base(i)))-1)/(np.exp(lam)-1))**M

k = np.arange(0,10)
Fsp = np.array([F_SP(i) for i in k])
Fps = np.array([F_PS(i) for i in k])
Rsp = 1 - Fsp
Rps = 1 - Fps

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

ax = axes[0]
ax.step(k, Fsp, where='post', color='#2a78d6', linewidth=2, label='F$_{S-P}$(k)')
ax.step(k, Fps, where='post', color='#e34948', linewidth=2, linestyle='--', label='F$_{P-S}$(k)')
ax.set_xlabel('k')
ax.set_ylabel('F(k)')
ax.set_title('(a) Funcția de repartiție cumulativă')
ax.legend(frameon=False, loc='lower right')
ax.set_xlim(-0.3, 9.3)
ax.set_ylim(0, 1.05)
ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.step(k, Rsp, where='post', color='#2a78d6', linewidth=2, label='R$_{S-P}$(k)')
ax.step(k, Rps, where='post', color='#e34948', linewidth=2, linestyle='--', label='R$_{P-S}$(k)')
ax.set_xlabel('k')
ax.set_ylabel('R(k)')
ax.set_title('(b) Funcția de fiabilitate')
ax.legend(frameon=False, loc='upper right')
ax.set_xlim(-0.3, 9.3)
ax.set_ylim(0, 1.05)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('fig1_cdf_reliability.png', dpi=200, bbox_inches='tight')
print("saved fig1")
