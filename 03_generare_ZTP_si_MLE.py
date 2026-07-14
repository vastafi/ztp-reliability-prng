"""
Script 3: Generare N_k ~ ZTP(lambda) din cifrele PRNG + estimare MLE
======================================================================

ATENTIE - acest script reconstituie metodologia descrisa in text
("blocuri de 9 cifre => o uniforma U in [0,1); pentru fiecare N_k se
trag apoi N_k cifre ca X_kj ~ Uniform{0,...,9}"), dar 2 detalii nu sunt
explicit specificate in draft-ul pe care il am si le-am ASUMAT mai jos:

  1. LAMBDA_TRUE = 2.5  -> parametrul "adevarat" folosit ca sa generezi
     datele sintetice N_k ~ ZTP(lambda_true). Am ales 2.5 pentru ca e o
     valoare "rotunda" si toate cele 3 surse recupereaza lambda_hat
     intre 2.49-2.51, ceea ce e compatibil cu lambda_true = 2.5.
     ==> CONFIRMA / CORECTEAZA aceasta valoare daca ai codul original
         de generare a datelor.

  2. Numarul de replicatii R = 1000, marimea esantionului per replicatie
     n ~ 92 -- scriptul consuma cifrele secvential si iti spune la
     final cate realizari complete a putut construi din cate cifre
     sunt disponibile; ajusteaza R sau numarul de cifre folosite daca
     vrei sa nimeresti exact n ~ 92.

Daca numerele care iti ies difera de Tabelul 2/3 din articol, cel mai
probabil una din aceste 2 asumptii trebuie corectata -- spune-mi
rezultatele si recalibram impreuna.
"""

import glob
import math
import numpy as np
import pandas as pd

DATA_DIR = "data"
M = 3
LAMBDA_TRUE = 2.5          # <-- ASUMAT, vezi nota de mai sus
R_REPLICATIONS = 1000      # numar de seturi independente (ca in articol)
GRID = np.linspace(0.05, 8.0, 300)  # grila pentru MLE (ca in text: 300 puncte)

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


# ---------------------------------------------------------------------
# Distributia ZTP(lambda): PMF si CDF (pentru inversare)
# ---------------------------------------------------------------------
def ztp_cdf_table(lam, k_max=60):
    ks = np.arange(1, k_max + 1)
    from math import exp, factorial
    unnorm = np.array([lam**k * math.exp(-lam) / math.factorial(k) for k in ks])
    unnorm /= (1 - math.exp(-lam))  # normalizare ZTP
    cdf = np.cumsum(unnorm)
    return ks, cdf


ZTP_KS, ZTP_CDF = ztp_cdf_table(LAMBDA_TRUE)


def digits_to_uniform(block: np.ndarray) -> float:
    """Combina un bloc de 9 cifre intr-un numar uniform in [0,1), rezolutie 1e-9."""
    s = "".join(str(d) for d in block)
    return int(s) / 1e9


def sample_ztp_from_uniform(u: float) -> int:
    """Inverseaza CDF-ul ZTP(lambda_true) pentru a obtine N_k."""
    idx = np.searchsorted(ZTP_CDF, u)
    idx = min(idx, len(ZTP_KS) - 1)
    return int(ZTP_KS[idx])


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

    def remaining(self) -> int:
        return len(self.digits) - self.pos


def generate_one_system_realization(stream: DigitStream, M_=M):
    """
    Genereaza o realizare (u, v) a duratelor de viata ale sistemului
    S-P si P-S, extragand N_k ~ ZTP(lambda_true) si X_kj ~ Uniform{0..9}
    pentru fiecare din cele M subsisteme.
    """
    T_max = []  # pentru S-P: max in interiorul subsistemului
    T_min = []  # pentru P-S: min in interiorul subsistemului
    for _ in range(M_):
        block = stream.take(9)
        u_cont = digits_to_uniform(block)
        Nk = sample_ztp_from_uniform(u_cont)
        xs = stream.take(Nk)
        T_max.append(xs.max())
        T_min.append(xs.min())
    U = min(T_max)   # S-P: min peste subsisteme (in serie) din max in interior (paralel)
    V = max(T_min)   # P-S: max peste subsisteme (in paralel) din min in interior (serie)
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
# PMF-ul sistemului S-P / P-S (formulele (7)-(8) din articol) - pt MLE
# ---------------------------------------------------------------------
def F_base(i):
    return (i + 1) / 10


def pmf_SP(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    Fsp = 1 - (1 - (np.exp(lam * F) - 1) / (np.exp(lam) - 1)) ** M_
    p = np.empty(10)
    p[0] = Fsp[0]
    p[1:] = np.diff(Fsp)
    return np.clip(p, 1e-300, None)  # evita log(0)


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
        print(f"Realizari de sistem generate: {len(Us)}")
        res = run_mle_replications(Us, Vs)
        res["Sursa PRNG"] = name
        results.append(res)
        print(res)

    df = pd.DataFrame(results).set_index("Sursa PRNG")
    print("\n\n===== TABEL FINAL (compara cu Tabelul 2 din articol) =====")
    print(df.to_string())
    df.to_csv("rezultat_03_mle.csv")
    print("\nSalvat in rezultat_03_mle.csv")
    print(f"\n(Reminder: LAMBDA_TRUE folosit la generare = {LAMBDA_TRUE}. "
          f"Daca lambda_hat_combinat difera mult de valorile din articol "
          f"(2.49-2.51), verifica daca LAMBDA_TRUE si/sau schema de "
          f"consum al cifrelor corespund codului tau original.)")


if __name__ == "__main__":
    main()
