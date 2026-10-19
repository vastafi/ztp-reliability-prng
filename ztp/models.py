"""Distribuțiile teoretice ale duratelor de viață U (S-P) și V (P-S): relațiile (6)-(19), (14a)-(14l)."""
import math
import numpy as np
from .config import K, LAMBDA, MU, M_FIX, N_FIX, Z, EPSILON


def A(theta):
    """Funcția generatoare ZTP: A(theta) = e^theta - 1."""
    return np.expm1(theta)


def cdf(model, k=K, lam=LAMBDA, mu=MU, m_fix=M_FIX, n_fix=N_FIX):
    """(F_SP, F_PS) pe suportul i = 0..k pentru Modelul I, II sau III."""
    i = np.arange(k + 1)
    F = (i + 1) / (k + 1)
    Q = 1 - F
    if model in ("I", "III"):                       # N_k ~ ZTP(lambda)
        G_sp = A(lam * F) / A(lam)
        G_ps = 1 - A(lam * Q) / A(lam)
    else:                                           # N constant
        G_sp = F ** n_fix
        G_ps = 1 - Q ** n_fix
    if model == "I":                                # M constant
        return 1 - (1 - G_sp) ** m_fix, G_ps ** m_fix
    return 1 - A(mu * (1 - G_sp)) / A(mu), A(mu * G_ps) / A(mu)   # M ~ ZTP(mu)


def characteristics(F):
    """PMF, CDF, fiabilitate, hazard, momente și entropie pentru o funcție de repartiție F pe {0..K}."""
    i = np.arange(len(F))
    p = np.diff(np.concatenate([[0.0], F]))
    mean = float((i * p).sum())
    var = float(((i - mean) ** 2 * p).sum())
    R = 1 - F
    h = p / np.concatenate([[1.0], R[:-1]])
    nz = p > 0
    return dict(p=p, F=F, R=R, h=h, mean=mean, var=var, sd=math.sqrt(var),
                skew=float(((i - mean) ** 3 * p).sum() / var ** 1.5),
                exc=float(((i - mean) ** 4 * p).sum() / var ** 2 - 3),
                H=float(-(p[nz] * np.log2(p[nz])).sum()))


def theory(model):
    F_sp, F_ps = cdf(model)
    return {"SP": characteristics(F_sp), "PS": characteristics(F_ps)}


def n_min(model):
    """Numărul minim de replici, relația (26): ceil((z * sigma / eps)^2)."""
    sd = theory(model)["SP"]["sd"]
    return math.ceil((Z * sd / EPSILON[model]) ** 2)
