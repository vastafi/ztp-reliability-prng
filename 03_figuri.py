"""
03_figuri.py — toate figurile articolului, generate din rezultatele salvate în results/ (fără resimulare).

    Figura 1 - PMF teoretică vs. empirică, Modelul I (3 surse x S-P/P-S)
    Figura 2 - funcția de repartiție teoretică vs. empirică, Modelul I
    Figura 3 - rata hazardului teoretică vs. empirică, Modelul I
    Figura 4 - funcțiile de fiabilitate R(i), Modelele I, II, III
    Figura 5 - PMF teoretice și empirice, Modelele I, II, III
    Figura 6 - CRQI ponderat global cu IC 95%, Modelele I, II, III

Rulați după 02_simulare_crqi.py. Ieșire: figure/figura1_pmf.png ... figure/figura6_crqi_ic95.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from ztp.config import RESULTS_DIR, FIGURE_DIR, MODELS, MODEL_TITLES, SOURCE_LABELS, K
from ztp.models import cdf, theory

SOURCES = list(SOURCE_LABELS.items())
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # paletă validată pentru daltonism
SRC_STYLE = {"Java ThreadLocalRandom": (BLUE, "o"), "Python secrets": (ORANGE, "s"), "Cifrele lui pi": (AQUA, "^")}
THEORY_BAR = "#3b6ea8"
SUPPORT = np.arange(K + 1)
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                     "xtick.color": "#444444", "ytick.color": "#444444"})


def load_results():
    res = {}
    for m in MODELS:
        path = RESULTS_DIR / f"rezultate_model_{m}.json"
        if not path.exists():
            raise FileNotFoundError(f"Lipsește {path}. Rulați întâi 02_simulare_crqi.py.")
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


def pmf_from_cdf(F):
    return np.diff(np.concatenate([[0.0], F]))


def empirical(res_model, source, net):
    """PMF, CDF și hazard empirice din toate replicile unei surse."""
    p = np.array(res_model["sources"][source][net]["emp_pmf"])
    F = np.cumsum(p)
    den = 1 - np.concatenate([[0.0], F[:-1]])
    h = np.where(den > 1e-12, p / np.where(den > 1e-12, den, 1), np.nan)
    return p, F, h


def six_panels(res, kind, ylabel, name):
    """Figurile 1-3: Modelul I, 3 surse (coloane) x S-P/P-S (rânduri)."""
    T = theory("I")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for c, (src, label) in enumerate(SOURCES):
        for r, (net, lab) in enumerate((("SP", "S-P, U"), ("PS", "P-S, V"))):
            ax = axes[r, c]
            p, F, h = empirical(res["I"], src, net)
            if kind == "pmf":
                ax.bar(SUPPORT - 0.18, T[net]["p"], 0.36, color=BLUE, label="Teoretic")
                ax.bar(SUPPORT + 0.18, p, 0.36, color=ORANGE, label="Empiric")
            elif kind == "cdf":
                ax.step(SUPPORT, T[net]["F"], where="post", color=BLUE, lw=2, marker="o", label="Teoretic")
                ax.step(SUPPORT, F, where="post", color=ORANGE, lw=2, ls="--", marker="s", label="Empiric")
            else:
                ax.plot(SUPPORT, T[net]["h"], color=BLUE, lw=2, marker="o", label="Teoretic")
                ax.plot(SUPPORT, h, color=ORANGE, lw=2, ls="--", marker="s", label="Empiric")
            ax.set_title(f"{label} ({lab})"); ax.set_xticks(SUPPORT); ax.grid(axis="y", alpha=0.3)
            if r == 1: ax.set_xlabel("i")
            if c == 0: ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------------
# Figura 4 - funcțiile de fiabilitate R(i) = P(T > i)
# ------------------------------------------------------------------
def figura4_fiabilitate():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    x = np.arange(-1, K + 1)
    for ax, m in zip(axes, MODELS):
        F_sp, F_ps = cdf(m)
        R_sp = np.concatenate([[1.0], 1 - F_sp])
        R_ps = np.concatenate([[1.0], 1 - F_ps])
        ax.step(x, R_sp, where="post", color=BLUE, lw=2)
        ax.plot(x[1:], R_sp[1:], "o", color=BLUE, ms=6, mec="white", mew=1.2)
        ax.step(x, R_ps, where="post", color=ORANGE, lw=2)
        ax.plot(x[1:], R_ps[1:], "s", color=ORANGE, ms=6, mec="white", mew=1.2)
        EU, EV = (1 - F_sp[:-1]).sum(), (1 - F_ps[:-1]).sum()   # E(T) = sum_i P(T > i)
        ax.text(0.97, 0.95, f"E(U) = {comma(EU)}\nE(V) = {comma(EV)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#333333")
        ax.set_title(MODEL_TITLES[m], fontsize=11)
        ax.set_xticks(range(K + 1)); ax.set_xlim(-0.5, K + 0.5); ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("i")
    axes[0].set_ylabel("R(i) = P(T > i)")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", lw=2, mec="white", label="Rețeaua S-P (U)"),
               plt.Line2D([], [], color=ORANGE, marker="s", lw=2, mec="white", label="Rețeaua P-S (V)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "figura4_fiabilitate.png")


# ------------------------------------------------------------------
# Figura 5 - PMF teoretice și empirice
# ------------------------------------------------------------------
def figura5_pmf(res):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    i = np.arange(K + 1)
    for c, m in enumerate(MODELS):
        F_sp, F_ps = cdf(m)
        for r, (net, F, lab) in enumerate((("SP", F_sp, "S-P, U"), ("PS", F_ps, "P-S, V"))):
            ax = axes[r, c]
            ax.bar(i, pmf_from_cdf(F), 0.7, color=THEORY_BAR, label="Teoretic", zorder=2)
            for key, name in SOURCES:
                col, mk = SRC_STYLE[key]
                ax.plot(i, res[m]["sources"][key][net]["emp_pmf"], mk, ms=5, mfc="none", mew=1.2,
                        color=col, label=f"Empiric – {name}", zorder=3)
            ax.set_title(f"{MODEL_TITLES[m]} ({lab})", fontsize=10.5)
            ax.set_xticks(i); ax.grid(axis="y", alpha=0.3, zorder=0)
            if c == 0: ax.set_ylabel("p(i)")
            if r == 1: ax.set_xlabel("i")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "figura5_pmf_modele.png")


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
                raise KeyError("Fișierele JSON nu conțin 'crqi_b'. Rulați din nou 02_simulare_crqi.py.")
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
    save(fig, "figura6_crqi_ic95.png")


def main():
    res = load_results()
    six_panels(res, "pmf", "p(i)", "figura1_pmf.png")
    six_panels(res, "cdf", "F(i)", "figura2_cdf.png")
    six_panels(res, "hazard", "h(i)", "figura3_hazard.png")
    figura4_fiabilitate()
    figura5_pmf(res)
    figura6_crqi(res)


if __name__ == "__main__":
    main()
