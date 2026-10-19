"""Generarea replicilor din șirurile de cifre: Knuth (uniforme din 5 cifre) + respingerea lui 0 pentru ZTP."""
import math
import numpy as np
import pandas as pd
from .config import LAMBDA, MU, M_FIX, N_FIX, DIGITS_PER_UNIFORM


def load_digits(files):
    """Concatenează fișierele unei surse (o cifră pe linie) într-o listă de int."""
    return np.concatenate([pd.read_csv(f, header=None).iloc[:, 0].values for f in files]).astype(int).tolist()


class DigitStream:
    def __init__(self, digits):
        self.d = digits
        self.pos = 0

    def take(self, n):
        if self.pos + n > len(self.d):
            raise StopIteration("cifre insuficiente")
        out = self.d[self.pos:self.pos + n]
        self.pos += n
        return out

    def uniform(self):
        v = 0
        for x in self.take(DIGITS_PER_UNIFORM):
            v = v * 10 + x
        return v / 10 ** DIGITS_PER_UNIFORM


def ztp(stream, lam):
    """Poisson(lam) prin algoritmul lui Knuth; valoarea 0 este respinsă (ZTP)."""
    L = math.exp(-lam)
    while True:
        k, p = 0, 1.0
        while True:
            k += 1
            p *= stream.uniform()
            if p <= L:
                break
        if k - 1 > 0:
            return k - 1


def replica(stream, model):
    """O replică: (U, V) din aceeași configurație de rețea.
    Ordinea consumului de cifre: M (dacă e aleator), apoi pentru fiecare subsistem N_k și duratele unităților."""
    M = M_FIX if model == "I" else ztp(stream, MU)
    t_max, t_min = [], []
    for _ in range(M):
        N = N_FIX if model == "II" else ztp(stream, LAMBDA)
        x = stream.take(N)
        t_max.append(max(x))
        t_min.append(min(x))
    return min(t_max), max(t_min)


def simulate(digits, model, n_replicas):
    stream = DigitStream(digits)
    U, V = [], []
    try:
        while len(U) < n_replicas:
            u, v = replica(stream, model)
            U.append(u); V.append(v)
    except StopIteration:
        pass
    return np.array(U), np.array(V), stream.pos
