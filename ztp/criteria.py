"""Cele zece criterii de conformitate D1-D10 (relațiile (27)-(43)) și indicele CRQI (46)-(48)."""
import numpy as np
from scipy import stats
from .config import K, WEIGHTS


def deviations(x, T):
    """Vectorul (d1..d10) pentru un set x de durate de viață, față de caracteristicile teoretice T."""
    n = len(x)
    m = x.mean(); d = x - m; m2 = (d ** 2).mean()
    S2 = (d ** 2).sum() / (n - 1)
    skew = (d ** 3).mean() / m2 ** 1.5
    exc = (d ** 4).mean() / m2 ** 2 - 3
    O = np.bincount(x, minlength=K + 1); ph = O / n
    H = -(ph[ph > 0] * np.log2(ph[ph > 0])).sum()
    rho = (d[:-1] * d[1:]).sum() / (d ** 2).sum()
    E = n * T["p"]
    p_con = stats.chi2.sf(((O - E) ** 2 / E).sum(), K)                     # (37), nu = K
    Fh = np.cumsum(ph)
    ks = np.abs(Fh - T["F"]).max()                                          # (39)
    tab = np.zeros((K + 1, K + 1)); np.add.at(tab, (x[:-1], x[1:]), 1)
    Er = np.outer(tab.sum(1), tab.sum(0)) / (n - 1); mask = Er > 0
    p_ind = stats.chi2.sf(((tab[mask] - Er[mask]) ** 2 / Er[mask]).sum(), K * K)   # (40), nu = K^2
    den = 1 - np.concatenate([[0.0], Fh[:-1]]); ok = den > 1e-12
    h_emp = np.where(ok, ph / np.where(ok, den, 1), np.nan)
    d_h = np.nanmax(np.abs(h_emp - T["h"]))                                 # (43)
    return np.array([abs(m - T["mean"]) / K,
                     min(abs(S2 - T["var"]) / T["var"], 1),
                     min(abs(skew - T["skew"]), 1),
                     min(abs(exc - T["exc"]) / (abs(T["exc"]) + 1e-6), 1),
                     min(abs(H - T["H"]) / T["H"], 1),
                     abs(rho), 1 - p_con, ks, 1 - p_ind, d_h])


def crqi(D):
    """D: matrice (B, 10) cu deviațiile pe seturi. Returnează CRQI simplu, ponderat și pe seturi."""
    w = np.array(WEIGHTS)
    dbar = D.mean(0)
    per_set = 1 - D @ w
    return dict(dbar=dbar.tolist(), CRQI_simplu=float(1 - dbar.mean()), CRQI_ponderat=float(1 - dbar @ w),
                sd_ponderat=float(per_set.std(ddof=0)), crqi_b=per_set.tolist())
