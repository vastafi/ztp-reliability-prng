"""
06_referinte_doua_niveluri.py — analize de referință (Secțiunile 3.2 și 3.4 ale articolului):

  1. Cazul clasic cu numere fixe (M = N = 3) și Modelele I/II la aceeași medie a numărului aleator
     (E[N_k] = 3, respectiv E[M] = 3): mediile lui U și V și ordinea stocastică dintre ele.
  2. Un generator de referință (numpy PCG64, sămânță fixă), simulat exact ca sursele reale:
     valoarea CRQI pe care o obține un generator bun (nivelul de bază al indicelui).
  3. Testul pe două niveluri pentru criteriile D7 și D9: pentru fiecare set se calculează valoarea p
     (a) asimptotică (χ²) și (b) calibrată prin simulare sub ipoteza nulă; apoi se testează
     uniformitatea celor B valori p (Kolmogorov–Smirnov).

Rulați după 02_simulare_crqi.py. Ieșire: results/referinte_doua_niveluri.json
"""
import json
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from ztp.config import RESULTS_DIR, MODELS, SOURCES, N_SETS, SET_SIZE, K, WEIGHTS, M_FIX, N_FIX, LAMBDA
from ztp.models import cdf, characteristics, theory
from ztp.simulation import load_digits, simulate
from ztp.controls import CONTROLS, N_CONTROL_DIGITS
from ztp.criteria import deviations
from ztp.two_level import chi_con, chi_ind, null_stats, p_calibrated

REF_NAME = "Referință PCG64 (numpy)"
REF_SEED = 2026


# ---------------------------------------------------------------- 1. cazul fix și aceeași medie
def fixed_cdf(m, n):
    i = np.arange(K + 1); F = (i + 1) / (K + 1); Q = 1 - F
    return 1 - (1 - F ** n) ** m, (1 - Q ** n) ** m


def order(Fu, Fv):
    d = (1 - Fu)[:-1] - (1 - Fv)[:-1]
    return "U ≥st V" if (d >= -1e-15).all() else ("V ≥st U" if (d <= 1e-15).all() else "fără ordine")


def baseline():
    theta3 = brentq(lambda t: t / (1 - np.exp(-t)) - 3, 0.1, 10)      # ZTP cu media 3
    rows = {}
    cases = {"Fix: M = N = 3": fixed_cdf(M_FIX, N_FIX),
             f"Modelul I, λ = {LAMBDA}": cdf("I"),
             f"Modelul I, λ = {theta3:.4f} (E[N] = 3)": cdf("I", lam=theta3),
             f"Modelul II, μ = {LAMBDA}": cdf("II"),
             f"Modelul II, μ = {theta3:.4f} (E[M] = 3)": cdf("II", mu=theta3),
             f"Modelul III, λ = μ = {LAMBDA}": cdf("III"),
             f"Modelul III, λ = μ = {theta3:.4f}": cdf("III", lam=theta3, mu=theta3)}
    for name, (Fu, Fv) in cases.items():
        cu, cv = characteristics(Fu), characteristics(Fv)
        rows[name] = dict(EU=cu["mean"], EV=cv["mean"], var=cu["var"], ordine=order(Fu, Fv))
    return dict(theta_media_3=theta3, cazuri=rows)


def main():
    rng = np.random.default_rng(REF_SEED)
    out = {"baseline": baseline()}
    for name, r in out["baseline"]["cazuri"].items():
        print(f"{name:38s} E(U)={r['EU']:.4f} E(V)={r['EV']:.4f}  {r['ordine']}")

    ref_digits = np.random.default_rng(REF_SEED).integers(0, 10, size=N_CONTROL_DIGITS).tolist()
    digits = {k: load_digits(v) for k, v in SOURCES.items()}
    digits.update({k: g(N_CONTROL_DIGITS) for k, g in CONTROLS.items()})
    digits[REF_NAME] = ref_digits
    w = np.array(WEIGHTS)

    out["modele"] = {}
    for m in MODELS:
        T = theory(m)
        nulls = {net: null_stats(T[net]["p"], rng) for net in ("SP", "PS")}
        res = {}
        for name, d in digits.items():
            U, V, _ = simulate(d, m, N_SETS[m] * SET_SIZE)
            B = len(U) // SET_SIZE
            g = {}
            per_set = []
            for net, X in (("SP", U), ("PS", V)):
                sets = [X[b * SET_SIZE:(b + 1) * SET_SIZE] for b in range(B)]
                D = np.array([deviations(s, T[net]) for s in sets])
                per_set.append(1 - D @ w)
                c_obs = np.array([chi_con(s, T[net]["p"]) for s in sets])
                i_obs = np.array([chi_ind(s) for s in sets])
                p_as = {"D7": 1 - D[:, 6], "D9": 1 - D[:, 8]}
                p_cal = {"D7": p_calibrated(c_obs, nulls[net][0]), "D9": p_calibrated(i_obs, nulls[net][1])}
                g[net] = {}
                for crit in ("D7", "D9"):
                    g[net][crit] = dict(
                        media_1_minus_p_asimptotic=float(1 - p_as[crit].mean()),
                        media_1_minus_p_calibrat=float(1 - p_cal[crit].mean()),
                        ks_p_asimptotic=float(stats.kstest(p_as[crit], "uniform").pvalue),
                        ks_p_calibrat=float(stats.kstest(p_cal[crit], "uniform").pvalue))
            g["CRQI_global"] = float(((per_set[0] + per_set[1]) / 2).mean())
            g["sets"] = B
            res[name] = g
            print(f"Modelul {m} | {name:26s} CRQI global {g['CRQI_global']:.4f} | KS calibrat "
                  + " ".join(f"{net}-{c}:{g[net][c]['ks_p_calibrat']:.3f}" for net in ("SP", "PS") for c in ("D7", "D9")),
                  flush=True)
        out["modele"][m] = res
    path = RESULTS_DIR / "referinte_doua_niveluri.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"Salvat: {path}")


if __name__ == "__main__":
    main()
