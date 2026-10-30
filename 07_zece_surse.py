"""
07_zece_surse.py — extinderea analizei la cele 10 surse din setul public de cifre
PRNG-Random-Digits-0-9-Dataset (Secțiunea 3.5, Tabelele 15-16, Figura 7).

Sursele sunt grupate în patru clase (ztp/config.py: SOURCE_CLASSES):
  A - PRNG algoritmic, implicit într-un mediu de calcul răspândit;  B - CSPRNG;
  C - secvență matematică deterministă (π);  D - TRNG (sursă fizică, Random.org).
Pentru fiecare model și sursă: CRQI ponderat (S-P, P-S, global) pe seturi, testul pe două niveluri
pentru D7 și D9 (valori p calibrate, ztp/two_level.py), ANOVA și Kruskal-Wallis între cele 10 surse.

Setul de date se descarcă separat (~450 MB):
    git clone https://github.com/vastafi/PRNG-Random-Digits-0-9-Dataset.git ../PRNG-Random-Digits-0-9-Dataset
sau se indică prin variabila de mediu ZTP_DATASET.

Ieșire: results/zece_surse.json, results/tabel_zece_surse.csv, figure/figura7_crqi_zece_surse.png
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from ztp.config import (RESULTS_DIR, FIGURE_DIR, MODELS, N_SETS, SET_SIZE, WEIGHTS,
                        DATASET_DIR, SOURCES_10, SOURCE_CLASSES, source_files_10)
from ztp.models import theory
from ztp.simulation import load_digits, simulate
from ztp.criteria import deviations
from ztp.two_level import chi_con, chi_ind, null_stats, p_calibrated

SEED = 2026


def effect_sizes(groups):
    allv = np.concatenate(groups); n = len(allv)
    ss_b = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups)
    eta2 = ss_b / ((allv - allv.mean()) ** 2).sum()
    H, pK = stats.kruskal(*groups)
    F, pF = stats.f_oneway(*groups)
    return dict(F=float(F), p_anova=float(pF), eta2=float(eta2), H=float(H), p_kruskal=float(pK), epsilon2=float(H / (n - 1)))


def main():
    if not DATASET_DIR.exists():
        raise SystemExit(f"Setul de date nu a fost găsit în {DATASET_DIR}.\n"
                         "git clone https://github.com/vastafi/PRNG-Random-Digits-0-9-Dataset.git "
                         f"{DATASET_DIR}\n(sau setați ZTP_DATASET)")
    digits = {}
    for name in SOURCES_10:
        d = load_digits(source_files_10(name))
        c = np.bincount(np.array(d), minlength=10)
        digits[name] = d
        print(f"{name:24s} {len(d):,} cifre, χ² uniformitate cifre p = {stats.chisquare(c).pvalue:.3f}", flush=True)
    rng = np.random.default_rng(SEED)
    w = np.array(WEIGHTS)
    out = {"clase": SOURCE_CLASSES, "surse": {k: v[0] for k, v in SOURCES_10.items()}, "modele": {}}
    rows = []
    for m in MODELS:
        T = theory(m)
        nulls = {net: null_stats(T[net]["p"], rng) for net in ("SP", "PS")}
        res, glob = {}, {}
        for name, d in digits.items():
            U, V, used = simulate(d, m, N_SETS[m] * SET_SIZE)
            B = len(U) // SET_SIZE
            r = {"sets": B, "digits_used": used}
            per = {}
            for net, X in (("SP", U), ("PS", V)):
                sets = [X[b * SET_SIZE:(b + 1) * SET_SIZE] for b in range(B)]
                D = np.array([deviations(s, T[net]) for s in sets])
                per[net] = 1 - D @ w
                p7 = p_calibrated(np.array([chi_con(s, T[net]["p"]) for s in sets]), nulls[net][0])
                p9 = p_calibrated(np.array([chi_ind(s) for s in sets]), nulls[net][1])
                r[net] = dict(CRQI_ponderat=float(per[net].mean()), dbar=D.mean(0).tolist(),
                              ks_D7=float(stats.kstest(p7, "uniform").pvalue),
                              ks_D9=float(stats.kstest(p9, "uniform").pvalue))
            g = (per["SP"] + per["PS"]) / 2
            glob[name] = g
            r["CRQI_global"] = float(g.mean())
            r["ic95"] = float(1.96 * g.std(ddof=1) / np.sqrt(B))
            r["crqi_global_b"] = g.tolist()
            r["ks_sub_005"] = int(sum(r[n][c] < 0.05 for n in ("SP", "PS") for c in ("ks_D7", "ks_D9")))
            res[name] = r
            rows.append({"Model": m, "Sursa": name, "Clasa": SOURCES_10[name][0], "CRQI global": r["CRQI_global"],
                         "IC95 ±": r["ic95"], "CRQI S-P": r["SP"]["CRQI_ponderat"], "CRQI P-S": r["PS"]["CRQI_ponderat"],
                         **{f"KS2 {n}-{c[3:]}": r[n][c] for n in ("SP", "PS") for c in ("ks_D7", "ks_D9")}})
        ranks = sorted(res, key=lambda k: -res[k]["CRQI_global"])
        for i, k in enumerate(ranks):
            res[k]["loc"] = i + 1
        test = effect_sizes(list(glob.values()))
        out["modele"][m] = {"surse": res, "teste": test}
        print(f"\nModelul {m}: ANOVA p = {test['p_anova']:.3f} (η² = {test['eta2']:.4f}), "
              f"Kruskal-Wallis p = {test['p_kruskal']:.3f}")
        for k in ranks:
            r = res[k]
            print(f"  {r['loc']:2d}. {k:24s} [{SOURCES_10[k][0]}] CRQI {r['CRQI_global']:.4f} ± {r['ic95']:.4f} | "
                  f"KS2<0,05: {r['ks_sub_005']}/4", flush=True)
    ks_all = [out["modele"][m]["surse"][k][n][c] for m in MODELS for k in SOURCES_10 for n in ("SP", "PS") for c in ("ks_D7", "ks_D9")]
    out["ks_total"] = dict(n=len(ks_all), sub_005=int(sum(p < 0.05 for p in ks_all)), asteptat=0.05 * len(ks_all))
    print(f"\nTestul pe două niveluri: {out['ks_total']['sub_005']} din {len(ks_all)} valori p < 0,05 "
          f"(așteptat sub H0: {0.05 * len(ks_all):.1f})")
    RESULTS_DIR.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS_DIR / "zece_surse.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tabel_zece_surse.csv", index=False, float_format="%.6f")
    print(f"Salvat: {RESULTS_DIR / 'zece_surse.json'}, {RESULTS_DIR / 'tabel_zece_surse.csv'}")
    figura7(out)


def figura7(out):
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#8a5cc2"}
    MK = {"A": "o", "B": "s", "C": "^", "D": "D"}
    names = sorted(SOURCES_10, key=lambda k: (SOURCES_10[k][0], k))
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        res = out["modele"][m]["surse"]
        pooled = np.mean([res[k]["CRQI_global"] for k in names])
        ax.axvline(pooled, color="#999999", lw=1, ls="--", zorder=1)
        for yi, k in zip(y, names):
            c = SOURCES_10[k][0]
            ax.errorbar(res[k]["CRQI_global"], yi, xerr=res[k]["ic95"], fmt=MK[c], color=COL[c], ms=7,
                        mec="white", mew=1, elinewidth=1.8, capsize=0, zorder=3)
        t = out["modele"][m]["teste"]
        ax.set_title(f"Modelul {m}  (ANOVA p = {t['p_anova']:.2f})".replace(".", ","), fontsize=11)
        ax.grid(axis="x", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.3f}".replace(".", ",")))
        ax.set_xlabel("CRQI ponderat global (IC 95%)")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names)
    handles = [plt.Line2D([], [], color=COL[c], marker=MK[c], ls="none", ms=7, label=f"{c}: {SOURCE_CLASSES[c]}")
               for c in SOURCE_CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "figura7_crqi_zece_surse.png"
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Salvat: {path}")


if __name__ == "__main__":
    main()
