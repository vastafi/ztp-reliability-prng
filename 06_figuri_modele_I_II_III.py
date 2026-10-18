"""
06_figuri_modele_I_II_III.py

Generează figurile noi ale articolului pentru Modelele I, II și III:
    Figura 4 - funcțiile de fiabilitate teoretice R(i) ale rețelelor S-P și P-S
    Figura 5 - funcțiile de masă teoretice și empirice (cele 3 surse)
    Figura 6 - CRQI ponderat global cu intervale de încredere de 95%

Citește rezultatele din results/rezultate_model_I.json, ..._II.json, ..._III.json,
produse de 05_modele_I_II_III.py, și salvează figurile în figure/.

Utilizare (din directorul proiectului):
    python 05_modele_I_II_III.py        # simularea (o singură dată)
    python 06_figuri_modele_I_II_III.py # figurile
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figure"

K, LAMBDA, MU, M_FIX, N_FIX = 9, 2.5, 2.5, 3, 3
MODELS = ["I", "II", "III"]
TITLES = {"I": "Modelul I: M = 3, N ~ ZTP(2,5)",
          "II": "Modelul II: N = 3, M ~ ZTP(2,5)",
          "III": "Modelul III: M, N ~ ZTP(2,5)"}
SOURCES = [("Java ThreadLocalRandom", "Java ThreadLocalRandom"),
           ("Python secrets", "Python secrets"),
           ("Cifrele lui pi", "Cifrele lui π")]

# culori (paletă validată pentru daltonism) + marcaje diferite pentru fiecare serie
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SRC_STYLE = {"Java ThreadLocalRandom": (BLUE, "o"),
             "Python secrets": (ORANGE, "s"),
             "Cifrele lui pi": (AQUA, "^")}
THEORY_BAR = "#3b6ea8"

plt.rcParams.update({
    "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
    "xtick.color": "#444444", "ytick.color": "#444444",
})


# ------------------------------------------------------------------
# Funcțiile de repartiție teoretice, relațiile (11)-(12) și (14e)-(14h)
# ------------------------------------------------------------------
def A(theta):
    """Funcția generatoare a distribuției ZTP: A(theta) = e^theta - 1."""
    return np.expm1(theta)


def cdf_theoretical(model):
    """Returnează (F_SP, F_PS) pe suportul i = 0, 1, ..., K."""
    i = np.arange(K + 1)
    F = (i + 1) / (K + 1)
    Q = 1 - F
    if model == "I":       # M fix, N_k ~ ZTP(lambda)
        G_sp = A(LAMBDA * F) / A(LAMBDA)
        G_ps = 1 - A(LAMBDA * Q) / A(LAMBDA)
        return 1 - (1 - G_sp) ** M_FIX, G_ps ** M_FIX
    if model == "II":      # N fix, M ~ ZTP(mu)
        G_sp = F ** N_FIX
        G_ps = 1 - Q ** N_FIX
    else:                  # Modelul III: M ~ ZTP(mu), N_k ~ ZTP(lambda)
        G_sp = A(LAMBDA * F) / A(LAMBDA)
        G_ps = 1 - A(LAMBDA * Q) / A(LAMBDA)
    return 1 - A(MU * (1 - G_sp)) / A(MU), A(MU * G_ps) / A(MU)


def pmf_from_cdf(F):
    return np.diff(np.concatenate([[0.0], F]))


def load_results():
    res = {}
    for m in MODELS:
        path = RESULTS / f"rezultate_model_{m}.json"
        if not path.exists():
            raise FileNotFoundError(f"Lipsește {path}. Rulați întâi 05_modele_I_II_III.py.")
        res[m] = json.load(open(path, encoding="utf-8"))
    return res


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvat: {out}")


def comma(v, digits=3):
    return f"{v:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------
# Figura 4 - funcțiile de fiabilitate R(i) = P(T > i)
# ------------------------------------------------------------------
def figura4_fiabilitate():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    x = np.arange(-1, K + 1)
    for ax, m in zip(axes, MODELS):
        F_sp, F_ps = cdf_theoretical(m)
        R_sp = np.concatenate([[1.0], 1 - F_sp])
        R_ps = np.concatenate([[1.0], 1 - F_ps])
        ax.step(x, R_sp, where="post", color=BLUE, lw=2)
        ax.plot(x[1:], R_sp[1:], "o", color=BLUE, ms=6, mec="white", mew=1.2)
        ax.step(x, R_ps, where="post", color=ORANGE, lw=2)
        ax.plot(x[1:], R_ps[1:], "s", color=ORANGE, ms=6, mec="white", mew=1.2)
        EU, EV = (1 - F_sp[:-1]).sum(), (1 - F_ps[:-1]).sum()   # E(T) = sum_i P(T > i)
        ax.text(0.97, 0.95, f"E(U) = {comma(EU)}\nE(V) = {comma(EV)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#333333")
        ax.set_title(TITLES[m], fontsize=11)
        ax.set_xticks(range(K + 1)); ax.set_xlim(-0.5, K + 0.5); ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("i")
    axes[0].set_ylabel("R(i) = P(T > i)")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", lw=2, mec="white", label="Rețeaua S-P (U)"),
               plt.Line2D([], [], color=ORANGE, marker="s", lw=2, mec="white", label="Rețeaua P-S (V)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "figura4_fiabilitate_modele_I_II_III.png")


# ------------------------------------------------------------------
# Figura 5 - PMF teoretice și empirice
# ------------------------------------------------------------------
def figura5_pmf(res):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    i = np.arange(K + 1)
    for c, m in enumerate(MODELS):
        F_sp, F_ps = cdf_theoretical(m)
        for r, (net, F, lab) in enumerate((("SP", F_sp, "S-P, U"), ("PS", F_ps, "P-S, V"))):
            ax = axes[r, c]
            ax.bar(i, pmf_from_cdf(F), 0.7, color=THEORY_BAR, label="Teoretic", zorder=2)
            for key, name in SOURCES:
                col, mk = SRC_STYLE[key]
                ax.plot(i, res[m]["sources"][key][net]["emp_pmf"], mk, ms=5, mfc="none", mew=1.2,
                        color=col, label=f"Empiric – {name}", zorder=3)
            ax.set_title(f"{TITLES[m]} ({lab})", fontsize=10.5)
            ax.set_xticks(i); ax.grid(axis="y", alpha=0.3, zorder=0)
            if c == 0: ax.set_ylabel("p(i)")
            if r == 1: ax.set_xlabel("i")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "figura5_pmf_modele_I_II_III.png")


# ------------------------------------------------------------------
# Figura 6 - CRQI ponderat global, IC 95% = medie ± 1,96·s/√B
# ------------------------------------------------------------------
def figura6_crqi(res):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(MODELS)); offsets = [-0.18, 0.0, 0.18]
    for j, (key, name) in enumerate(SOURCES):
        col, mk = SRC_STYLE[key]
        means, half = [], []
        for m in MODELS:
            s = res[m]["sources"][key]
            if "crqi_b" not in s["SP"]:
                raise KeyError("Fișierele JSON nu conțin 'crqi_b'. Rulați din nou 05_modele_I_II_III.py (versiunea actualizată).")
            g = (np.array(s["SP"]["crqi_b"]) + np.array(s["PS"]["crqi_b"])) / 2   # scorul global pe fiecare set
            means.append(g.mean()); half.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
        ax.errorbar(xs + offsets[j], means, yerr=half, fmt=mk, color=col, ms=8, mec="white", mew=1.2,
                    elinewidth=2, capsize=0, label=name)
    ax.set_xticks(xs); ax.set_xticklabels([f"Modelul {m}" for m in MODELS])
    ax.set_ylabel("CRQI ponderat global")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: comma(v)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "figura6_crqi_ic95_modele_I_II_III.png")


def main():
    res = load_results()
    figura4_fiabilitate()
    figura5_pmf(res)
    figura6_crqi(res)


if __name__ == "__main__":
    main()
