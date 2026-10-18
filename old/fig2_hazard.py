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
psp = np.diff(np.concatenate([[0],Fsp]))
pps = np.diff(np.concatenate([[0],Fps]))
Rsp = 1-Fsp
Rps = 1-Fps

hsp = np.empty(10); hps = np.empty(10)
Rsp_prev, Rps_prev = 1.0, 1.0
for i in range(10):
    hsp[i] = psp[i]/Rsp_prev
    hps[i] = pps[i]/Rps_prev
    Rsp_prev, Rps_prev = Rsp[i], Rps[i]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(k, hsp, 'o-', color='#2a78d6', linewidth=2, markersize=5, label='h$_{S-P}$(k)')
ax.plot(k, hps, 's--', color='#e34948', linewidth=2, markersize=5, label='h$_{P-S}$(k)')
ax.set_xlabel('k')
ax.set_ylabel('h(k)')
ax.set_title('Rata de hazard pentru rețelele S-P și P-S')
ax.legend(frameon=False, loc='upper left')
ax.set_xlim(-0.3, 9.3)
ax.set_ylim(0, 1.05)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', color='#e1e0d9', linewidth=0.7)
plt.tight_layout()
plt.savefig('fig2_hazard.png', dpi=200, bbox_inches='tight')
print("saved fig2")
