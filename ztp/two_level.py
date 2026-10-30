"""Testul pe două niveluri pentru criteriile D7 și D9 (Secțiunea 2.7.4, relațiile (69)-(70)):
statisticile χ², distribuția lor sub ipoteza nulă (prin simulare) și valorile p calibrate."""
import numpy as np
from .config import K, SET_SIZE

N_NULL = 20_000          # eșantioane sub ipoteza nulă pentru valorile p calibrate


def chi_con(x, p):
    O = np.bincount(x, minlength=K + 1); E = len(x) * p
    return ((O - E) ** 2 / E).sum()


def chi_ind(x):
    n = len(x)
    tab = np.zeros((K + 1, K + 1)); np.add.at(tab, (x[:-1], x[1:]), 1)
    E = np.outer(tab.sum(1), tab.sum(0)) / (n - 1); m = E > 0
    return ((tab[m] - E[m]) ** 2 / E[m]).sum()


def null_stats(p, rng, n=SET_SIZE, S=N_NULL, chunk=2000):
    """Statisticile χ² de concordanță și de independență pentru S eșantioane i.i.d. din p."""
    con, ind = [], []
    k1 = K + 1
    for s0 in range(0, S, chunk):
        s = min(chunk, S - s0)
        x = rng.choice(k1, size=(s, n), p=p)
        O = np.zeros((s, k1)); np.add.at(O, (np.repeat(np.arange(s), n), x.ravel()), 1)
        E = n * p
        con.append(((O - E) ** 2 / E).sum(1))
        code = x[:, :-1] * k1 + x[:, 1:] + (np.arange(s) * k1 * k1)[:, None]
        tab = np.bincount(code.ravel(), minlength=s * k1 * k1).reshape(s, k1, k1).astype(float)
        Er = tab.sum(2)[:, :, None] * tab.sum(1)[:, None, :] / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(Er > 0, (tab - Er) ** 2 / Er, 0.0)
        ind.append(t.sum((1, 2)))
    return np.concatenate(con), np.concatenate(ind)


def p_calibrated(obs, null):
    null = np.sort(null)
    ge = len(null) - np.searchsorted(null, obs, side="left")
    return (1 + ge) / (len(null) + 1)
