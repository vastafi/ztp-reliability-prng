"""Cele opt criterii de mărime a abaterii D1,D2,D3,D4,D5,D6,D8,D10 (relațiile (27)-(43), fără (37)/(40))
și indicele CRQI (46)-(48). D7 (concordanță) și D9 (independență) sunt teste de semnificație, calculate
separat în ztp/two_level.py (chi_con, chi_ind, p_calibrated, holm_significant) și raportate ca atare,
nu agregate în CRQI."""
import numpy as np
from .config import K, WEIGHTS


def deviations(x, T):
    """Vectorul (d1,d2,d3,d4,d5,d6,d8,d10) pentru un set x de durate de viață, față de T."""
    n = len(x)
    m = x.mean(); d = x - m; m2 = (d ** 2).mean()
    S2 = (d ** 2).sum() / (n - 1)
    skew = (d ** 3).mean() / m2 ** 1.5
    exc = (d ** 4).mean() / m2 ** 2 - 3
    O = np.bincount(x, minlength=K + 1); ph = O / n
    H = -(ph[ph > 0] * np.log2(ph[ph > 0])).sum()
    rho = (d[:-1] * d[1:]).sum() / (d ** 2).sum()
    Fh = np.cumsum(ph)
    ks = np.abs(Fh - T["F"]).max()                                          # (39)
    den = 1 - np.concatenate([[0.0], Fh[:-1]]); ok = den > 1e-12
    h_emp = np.where(ok, ph / np.where(ok, den, 1), np.nan)
    d_h = np.nanmax(np.abs(h_emp - T["h"]))                                 # (43)
    return np.array([abs(m - T["mean"]) / K,
                     min(abs(S2 - T["var"]) / T["var"], 1),
                     min(abs(skew - T["skew"]), 1),
                     min(abs(exc - T["exc"]) / (abs(T["exc"]) + 1e-6), 1),
                     min(abs(H - T["H"]) / T["H"], 1),
                     abs(rho), ks, d_h])


def crqi(D):
    """D: matrice (B, 10) cu deviațiile pe seturi. Returnează CRQI simplu, ponderat și pe seturi."""
    w = np.array(WEIGHTS)
    dbar = D.mean(0)
    per_set = 1 - D @ w
    return dict(dbar=dbar.tolist(), CRQI_simplu=float(1 - dbar.mean()), CRQI_ponderat=float(1 - dbar @ w),
                sd_ponderat=float(per_set.std(ddof=0)), crqi_b=per_set.tolist())
