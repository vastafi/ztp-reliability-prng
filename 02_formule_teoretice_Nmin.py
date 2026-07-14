"""
Script 2 (CORECTAT): Caracteristici teoretice + N_min din informatia Fisher
=============================================================================

IMPORTANT -- versiune corectata fata de scriptul anterior.

Ținta acestui articol este estimarea parametrului LAMBDA (al distributiei
ZTP) prin MLE -- vezi Sectiunea 2.4-2.5. Prin urmare N_min (volumul minim
de esantion) TREBUIE calculat din formula (22) a articolului, bazata pe
INFORMATIA FISHER I(lambda) pentru precizia lui lambda_hat insusi:

    N_min = ceil( 1 / (eps^2 * I(lambda)) ),  eps = eps_rel * lambda

NU din formula gresita folosita anterior (CLT aplicat direct pe media
duratei de viata a retelei E[U]/E[V]) -- aceea raspunde la o intrebare
diferita (precizia lui E[U], nu a lui lambda) si NU e formula pe care
articolul o defineste in Sectiunea 2.5.

O consecinta importanta: N_min din informatia Fisher NU depinde de
topologia retelei (S-P vs P-S) -- Fisher info se refera la observatii
Nk ~ ZTP(lambda), independent de cum sunt combinate ulterior in U/V.
Deci exista UN SINGUR N_min per sursa PRNG, nu cate unul separat
pentru S-P si P-S.
"""

import math
import numpy as np
import pandas as pd

M = 3
Z_0975 = 1.9600
EPS_REL = 0.001  # eroare relativa 0.1%, consistenta cu restul articolului


def F_base(i):
    return (i + 1) / 10


def A(theta):
    return np.exp(theta) - 1


def F_SP(i, lam, M_=M):
    return 1 - (1 - A(lam * F_base(i)) / A(lam)) ** M_


def F_PS(i, lam, M_=M):
    return (1 - A(lam * (1 - F_base(i))) / A(lam)) ** M_


def pmf_from_cdf(F_vals):
    p = np.empty(10)
    p[0] = F_vals[0]
    p[1:] = np.diff(F_vals)
    return p


def caracteristici(lam, M_=M):
    """Tabelul 4: caracteristici teoretice F, p, R, h + E[U], D[U] etc.
       (neschimbat fata de scriptul anterior -- nu are legatura cu N_min)."""
    idx = np.arange(10)
    Fsp = np.array([F_SP(i, lam, M_) for i in idx])
    Fps = np.array([F_PS(i, lam, M_) for i in idx])
    psp = pmf_from_cdf(Fsp)
    pps = pmf_from_cdf(Fps)
    Rsp, Rps = 1 - Fsp, 1 - Fps
    hsp, hps = np.empty(10), np.empty(10)
    Rsp_prev, Rps_prev = 1.0, 1.0
    for i in idx:
        hsp[i] = psp[i] / Rsp_prev
        hps[i] = pps[i] / Rps_prev
        Rsp_prev, Rps_prev = Rsp[i], Rps[i]
    EU, EV = np.sum(idx * psp), np.sum(idx * pps)
    DU = np.sum((idx - EU) ** 2 * psp)
    DV = np.sum((idx - EV) ** 2 * pps)
    return {
        "F": F_base(idx), "F_SP": Fsp, "p_SP": psp, "R_SP": Rsp, "h_SP": hsp,
        "F_PS": Fps, "p_PS": pps, "R_PS": Rps, "h_PS": hps,
        "E_U_SP": EU, "D_U_SP": DU, "sigma_SP": math.sqrt(DU),
        "E_V_PS": EV, "D_V_PS": DV, "sigma_PS": math.sqrt(DV),
    }


def fisher_info_ZTP(lam: float) -> float:
    """Informatia Fisher I(lambda) pentru o SINGURA observatie Nk ~ ZTP(lambda).
       Formula (21) din articol."""
    return (1 / lam) * (1 / (1 - math.exp(-lam))) - math.exp(-lam) / (1 - math.exp(-lam)) ** 2


def n_min_fisher(lam: float, eps_rel: float = EPS_REL) -> int:
    """N_min CORECT -- formula (22) din articol, bazata pe informatia Fisher."""
    I = fisher_info_ZTP(lam)
    eps = eps_rel * lam
    return math.ceil(1 / (eps ** 2 * I))


def tabel_4(lam, M_=M, label=""):
    c = caracteristici(lam, M_)
    df = pd.DataFrame({
        "k": range(10), "F(k)": c["F"], "F_SP(k)": c["F_SP"], "p_SP(k)": c["p_SP"],
        "R_SP(k)": c["R_SP"], "h_SP(k)": c["h_SP"], "F_PS(k)": c["F_PS"], "p_PS(k)": c["p_PS"],
        "R_PS(k)": c["R_PS"], "h_PS(k)": c["h_PS"],
    }).round(6)
    print(f"\n=== Tabelul 4 -- {label} (lambda={lam}, M={M_}) ===")
    print(df.to_string(index=False))
    print(f"E[U_SP]={c['E_U_SP']:.6f}  sigma_SP={c['sigma_SP']:.6f}  "
          f"E[V_PS]={c['E_V_PS']:.6f}  sigma_PS={c['sigma_PS']:.6f}  "
          f"E[U]+E[V]={c['E_U_SP']+c['E_V_PS']:.6f}")
    return df, c


def tabel_3_CORECTAT(lambda_estimates: dict, eps_rel: float = EPS_REL):
    """
    Tabelul 3 CORECTAT -- N_min din informatia Fisher (ec. 22), UN SINGUR
    N_min per sursa (nu separat S-P / P-S, pentru ca nu depinde de topologie).

    lambda_estimates: dict {"Nume sursa": lambda_hat}
    """
    rows = []
    for nume, lam in lambda_estimates.items():
        I = fisher_info_ZTP(lam)
        eps = eps_rel * lam
        Nmin = n_min_fisher(lam, eps_rel)
        rows.append({
            "Sursa PRNG": nume, "lambda_hat": round(lam, 6),
            "I(lambda_hat)": round(I, 6), "eps": round(eps, 6), "N_min": Nmin,
        })
    df = pd.DataFrame(rows)
    print("\n=== Tabelul 3 CORECTAT -- N_min din informatia Fisher (ec. 22) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    tabel_4(2.4995, 3, label="lambda combinat (toate sursele)")

    lambda_estimates = {
        "Java ThreadLocalRandom": 2.497045,
        "Python secrets":         2.499783,
        "Cifrele lui pi":         2.501764,
    }
    tabel_3_CORECTAT(lambda_estimates)

    print("\n--- Comparatie cu volumul de date deja disponibil ---")
    n_realizari = {
        "Java ThreadLocalRandom": 446145,
        "Python secrets":         446054,
        "Cifrele lui pi":         446070,
    }
    for nume, lam in lambda_estimates.items():
        Nmin = n_min_fisher(lam)
        n_disp = n_realizari[nume]
        pct = 100 * n_disp / Nmin
        print(f"{nume:30s}: N_min={Nmin:>7d}  disponibil={n_disp:>7d}  ({pct:.1f}%)")

    for lam_test in (1.0, 2.4995, 5.0):
        c = caracteristici(lam_test, 3)
        print(f"\nlambda={lam_test}: E[U]+E[V]={c['E_U_SP']+c['E_V_PS']:.10f} (trebuie sa fie exact 9)")
