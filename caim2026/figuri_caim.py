#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAIM 2026 — generates the figures and formula images for the two new slides.

    pip install numpy matplotlib
    python figuri_caim.py

Writes into ./fg/

The formula images are transparent PNGs at 320 dpi, so they can be dropped
straight onto a slide and scaled by width; keep the aspect ratio.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
NAVY = "#1F3864"      # titles, curves for S-P
TEAL = "#2E7D8C"
GOLD = "#C9A227"      # curves for P-S
GREY = "#9AA7B0"      # annotations
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 220,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fg")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- model
K, M, LAM = 9, 3, 2.5          # parameters of the paper

i = np.arange(K + 1)           # 0,1,...,9
F = (i + 1) / (K + 1)          # CDF of one unit, U{0,...,K}
A = lambda t: np.exp(t) - 1    # generating function of the ZTP family

# equations (11) and (12) of the paper
F_SP = 1 - (1 - A(LAM * F) / A(LAM)) ** M
F_PS = (1 - A(LAM * (1 - F)) / A(LAM)) ** M

p_SP = np.diff(np.r_[0.0, F_SP])
p_PS = np.diff(np.r_[0.0, F_PS])

# sanity checks — these must hold, and they are what the talk claims
assert abs(p_SP.sum() - 1) < 1e-12 and abs(p_PS.sum() - 1) < 1e-12
assert np.allclose(p_SP, p_PS[::-1]), "duality p_SP(i) = p_PS(K-i) violated"
mean_SP = (i * p_SP).sum()
mean_PS = (i * p_PS).sum()
print(f"means: S-P {mean_SP:.6f}   P-S {mean_PS:.6f}   (paper: 4.175996 / 4.824004)")


# ------------------------------------------------- figure: CDF + reliability
def fig_cdf_reliability():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.15))

    a = ax[0]
    a.step(np.r_[-0.5, i], np.r_[0, F_SP], where="post", color=TEAL, lw=2.2,
           label=r"$F_{S\!-\!P}(i)$")
    a.step(np.r_[-0.5, i], np.r_[0, F_PS], where="post", color=GOLD, lw=2.2,
           label=r"$F_{P\!-\!S}(i)$")
    a.plot(i, F_SP, "o", color=TEAL, ms=4.5)
    a.plot(i, F_PS, "s", color=GOLD, ms=4.5)
    a.set_xticks(i)
    a.set_xlabel("lifetime  $i$")
    a.set_ylabel("cumulative probability")
    a.set_ylim(0, 1.04)
    a.set_title("Distribution functions", fontsize=12, color=NAVY, pad=8)
    a.legend(loc="upper left", fontsize=10.5)
    a.text(5.6, 0.10, "S-P curve lies above:\nit fails earlier",
           fontsize=9.5, color=GREY, ha="center")

    b = ax[1]
    b.step(np.r_[-0.5, i], np.r_[1, 1 - F_SP], where="post", color=TEAL, lw=2.2,
           label=r"$R_{S\!-\!P}(i)$")
    b.step(np.r_[-0.5, i], np.r_[1, 1 - F_PS], where="post", color=GOLD, lw=2.2,
           label=r"$R_{P\!-\!S}(i)$")
    b.plot(i, 1 - F_SP, "o", color=TEAL, ms=4.5)
    b.plot(i, 1 - F_PS, "s", color=GOLD, ms=4.5)
    b.fill_between(i, 1 - F_SP, 1 - F_PS, step="post", color=GOLD, alpha=0.12)
    b.set_xticks(i)
    b.set_xlabel("lifetime  $i$")
    b.set_ylabel("reliability  $R(i)$")
    b.set_ylim(0, 1.04)
    b.set_title("Reliability functions", fontsize=12, color=NAVY, pad=8)
    b.legend(loc="upper right", fontsize=10.5)
    b.text(6.4, 0.62, "P-S is uniformly\nmore reliable",
           fontsize=9.5, color=GREY, ha="center")

    fig.tight_layout(pad=1.0)
    fig.savefig(f"{OUT}/cdf_rel.png")
    plt.close(fig)


# ------------------------------------------------- figure: pmf (spare slide)
def fig_pmf():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w = 0.4
    ax.bar(i - w / 2, p_SP, w, color=TEAL, label=r"$p_{S\!-\!P}(i)$   (mean 4.176)")
    ax.bar(i + w / 2, p_PS, w, color=GOLD, label=r"$p_{P\!-\!S}(i)$   (mean 4.824)")
    ax.set_xticks(i)
    ax.set_xlabel("lifetime  $i$")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 0.175)
    ax.legend(loc="upper center", fontsize=10, ncol=2, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pmf.png")
    plt.close(fig)


# ------------------------------------------------- formulas as images
def eq(name, tex, fs=25, color=NAVY):
    """Render a LaTeX expression to a tight transparent PNG."""
    f = plt.figure(figsize=(0.01, 0.01))
    f.text(0, 0, tex, fontsize=fs, color=color, math_fontfamily="dejavuserif")
    f.savefig(f"{OUT}/{name}.png", dpi=320, bbox_inches="tight",
              pad_inches=0.10, transparent=True)
    plt.close(f)


if __name__ == "__main__":
    fig_cdf_reliability()
    fig_pmf()
    print("written to", os.path.abspath(OUT))
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
