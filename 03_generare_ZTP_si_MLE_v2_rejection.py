"""
Script 3 v2 (CORECTAT): Generare ZTP prin metoda coordonatorului
====================================================================

Coordonatorul a confirmat metoda: genereaza Poisson(lambda) SIMPLU;
daca rezultatul e > 0, il accepti ca N_k; daca e = 0, respingi si
repeti. Acesta e algoritmul standard de "rejection sampling" pentru
Poisson trunchiat in 0 (ZTP), diferit de Scriptul 3 v1 (care folosea
inversarea directa a CDF-ului ZTP -- matematic tot corect, dar NU
metoda descrisa de coordonator).

Pentru a genera Poisson(lambda) dintr-un flux de cifre, se foloseste
algoritmul lui Knuth (1998), care are nevoie de un NUMAR VARIABIL de
uniforme (0,1) per esantion Poisson:

    L = exp(-lambda); k = 0; p = 1
    repeta: k += 1; p *= Uniform(0,1)
    pana cand p <= L
    returneaza k - 1

Fiecare Uniform(0,1) e format aici din 4 cifre consecutive (rezolutie
10^-4 -- suficienta pentru precizia Poisson, si mult mai economicoasa
in cifre decat blocurile de 9 cifre folosite pentru CDF direct).

ATENTIE -- consecinta practica: fiindca genereaza un numar VARIABIL
de cifre consumate per N_k (spre deosebire de Scriptul 3 v1, care
folosea exact 9 cifre fixe per N_k), numarul total de realizari de
sistem obtinute dintr-un pool fix de cifre difera fata de rezultatele
anterioare -- de aceea toate cifrele din Tabelul 2/3/5 trebuie
RECALCULATE cu acest script, nu doar comparate.
"""

import glob
import math
import numpy as np
import pandas as pd

DATA_DIR = "data"
M = 3
LAMBDA_TRUE = 2.5           # parametrul Poisson folosit la generare (neschimbat)
R_REPLICATIONS = 1000
GRID = np.linspace(0.05, 8.0, 300)
DIGITS_PER_UNIFORM = 4       # cate cifre formeaza un singur Uniform(0,1) in algoritmul Knuth

SOURCES = {
    "Java ThreadLocalRandom": f"{DATA_DIR}/java_threadlocal/*.csv",
    "Python secrets":         f"{DATA_DIR}/python_secrets/*.csv",
    "Cifrele lui pi":         f"{DATA_DIR}/pi/*.csv",
}


def load_digits(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Niciun fisier gasit pentru: {pattern}")
    frames = [pd.read_csv(f, header=None).iloc[:, 0].values for f in files]
    return np.concatenate(frames).astype(int)


class DigitStream:
    """Consuma secvential un flux de cifre 0-9."""
    def __init__(self, digits: np.ndarray):
        self.digits = digits
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        if self.pos + n > len(self.digits):
            raise StopIteration("Cifre insuficiente in flux")
        out = self.digits[self.pos:self.pos + n]
        self.pos += n
        return out

    def next_uniform(self, n_digits: int = DIGITS_PER_UNIFORM) -> float:
        """Un singur Uniform(0,1) din n_digits cifre consecutive."""
        block = self.take(n_digits)
        s = "".join(str(d) for d in block)
        return int(s) / (10 ** n_digits)


def sample_poisson_knuth(stream: DigitStream, lam: float) -> int:
    """Algoritmul lui Knuth pentru Poisson(lam) -- consuma un numar
    variabil de uniforme (deci de cifre) per esantion generat."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        u = stream.next_uniform()
        p *= u
        if p <= L:
            return k - 1


def sample_ztp_rejection(stream: DigitStream, lam: float = LAMBDA_TRUE) -> int:
    """Metoda EXACTA descrisa de coordonator: genereaza Poisson(lam);
    daca e 0, respinge si repeta; daca e > 0, accepta ca N_k ~ ZTP(lam)."""
    while True:
        y = sample_poisson_knuth(stream, lam)
        if y > 0:
            return y


def generate_one_system_realization(stream: DigitStream, M_=M):
    T_max, T_min = [], []
    for _ in range(M_):
        Nk = sample_ztp_rejection(stream)
        xs = stream.take(Nk)
        T_max.append(xs.max())
        T_min.append(xs.min())
    U = min(T_max)
    V = max(T_min)
    return U, V


def generate_all_realizations(digits: np.ndarray):
    stream = DigitStream(digits)
    Us, Vs = [], []
    while True:
        try:
            u, v = generate_one_system_realization(stream)
            Us.append(u)
            Vs.append(v)
        except StopIteration:
            break
    return np.array(Us), np.array(Vs)


# ---------------------------------------------------------------------
# PMF-ul sistemului S-P / P-S (formulele (13)-(15) din articol) - pt MLE
# ---------------------------------------------------------------------
def F_base(i):
    return (i + 1) / 10


def pmf_SP(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    Fsp = 1 - (1 - (np.exp(lam * F) - 1) / (np.exp(lam) - 1)) ** M_
    p = np.empty(10)
    p[0] = Fsp[0]
    p[1:] = np.diff(Fsp)
    return np.clip(p, 1e-300, None)


def pmf_PS(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    Fps = (1 - (np.exp(lam * (1 - F)) - 1) / (np.exp(lam) - 1)) ** M_
    p = np.empty(10)
    p[0] = Fps[0]
    p[1:] = np.diff(Fps)
    return np.clip(p, 1e-300, None)


def mle_grid_search(sample: np.ndarray, pmf_func, grid=GRID):
    best_lam, best_ll = None, -np.inf
    for lam in grid:
        p = pmf_func(lam)
        ll = np.sum(np.log(p[sample]))
        if ll > best_ll:
            best_ll, best_lam = ll, lam
    return best_lam


def run_mle_replications(sample_U: np.ndarray, sample_V: np.ndarray, R=R_REPLICATIONS):
    n_total = len(sample_U)
    n_per_rep = n_total // R
    if n_per_rep == 0:
        raise ValueError(f"Nu sunt destule realizari ({n_total}) pentru R={R} replicatii")

    lam_sp_hats, lam_ps_hats = [], []
    for r in range(R):
        chunk_U = sample_U[r * n_per_rep:(r + 1) * n_per_rep]
        chunk_V = sample_V[r * n_per_rep:(r + 1) * n_per_rep]
        lam_sp_hats.append(mle_grid_search(chunk_U, pmf_SP))
        lam_ps_hats.append(mle_grid_search(chunk_V, pmf_PS))

    return {
        "n_total_realizari": n_total,
        "n_per_replicatie": n_per_rep,
        "lambda_hat_SP_mediu": np.mean(lam_sp_hats),
        "sigma_lambda_SP": np.std(lam_sp_hats),
        "lambda_hat_PS_mediu": np.mean(lam_ps_hats),
        "sigma_lambda_PS": np.std(lam_ps_hats),
        "lambda_hat_combinat": (np.mean(lam_sp_hats) + np.mean(lam_ps_hats)) / 2,
    }


def main():
    results = []
    for name, pattern in SOURCES.items():
        print(f"\n=== {name} ===")
        digits = load_digits(pattern)
        print(f"Cifre disponibile: {len(digits)}")
        Us, Vs = generate_all_realizations(digits)
        print(f"Realizari de sistem generate (metoda rejection Knuth): {len(Us)}")
        res = run_mle_replications(Us, Vs)
        res["Sursa PRNG"] = name
        results.append(res)
        print(res)

    df = pd.DataFrame(results).set_index("Sursa PRNG")
    print("\n\n===== TABEL FINAL (metoda EXACTA a coordonatorului) =====")
    print(df.to_string())
    df.to_csv("rezultat_03v2_mle_rejection.csv")
    print("\nSalvat in rezultat_03v2_mle_rejection.csv")
    print(f"\n(LAMBDA_TRUE folosit la generare = {LAMBDA_TRUE}, "
          f"DIGITS_PER_UNIFORM = {DIGITS_PER_UNIFORM})")


if __name__ == "__main__":
    main()
