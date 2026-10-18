"""
Script 4 v2 (CORECTAT): Eroarea CDF empiric vs. teoretic (max|deltaF|)
=========================================================================

Foloseste EXACT aceeasi metoda de generare ca Scriptul 3 v2
(rejection sampling pe Poisson simplu, confirmata de coordonator),
NU metoda veche (inversare CDF) din Scriptul 4 v1.

Reproduce coloanele "max|deltaF_SP|" si "max|deltaF_PS|" din Tabelul 5.
"""

import glob
import math
import numpy as np
import pandas as pd

DATA_DIR = "../data"
M = 3
LAMBDA_TRUE = 2.5
DIGITS_PER_UNIFORM = 4

SOURCES = {
    "Java ThreadLocalRandom": f"{DATA_DIR}/java_threadlocal/*.csv",
    "Python secrets":         f"{DATA_DIR}/python_secrets/*.csv",
    "Cifrele lui pi":         f"{DATA_DIR}/pi/*.csv",
}

# lambda_hat_combinat obtinute din Scriptul 3 v2 (folosite implicit daca
# nu gasim rezultat_03v2_mle_rejection.csv)
DEFAULT_LAMBDA_HAT = {
    "Java ThreadLocalRandom": 2.503971,
    "Python secrets":         2.500501,
    "Cifrele lui pi":         2.496154,
}


def load_digits(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Niciun fisier gasit pentru: {pattern}")
    frames = [pd.read_csv(f, header=None).iloc[:, 0].values for f in files]
    return np.concatenate(frames).astype(int)


class DigitStream:
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
        block = self.take(n_digits)
        s = "".join(str(d) for d in block)
        return int(s) / (10 ** n_digits)


def sample_poisson_knuth(stream: DigitStream, lam: float) -> int:
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
    """Metoda coordonatorului: Poisson simplu, respinge daca 0."""
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
    return min(T_max), max(T_min)


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


def F_base(i):
    return (i + 1) / 10


def F_SP_theoretical(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    return 1 - (1 - (np.exp(lam * F) - 1) / (np.exp(lam) - 1)) ** M_


def F_PS_theoretical(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    return (1 - (np.exp(lam * (1 - F)) - 1) / (np.exp(lam) - 1)) ** M_


def empirical_cdf(sample: np.ndarray) -> np.ndarray:
    n = len(sample)
    return np.array([np.sum(sample <= k) / n for k in range(10)])


def mse(a, b):
    return np.mean((a - b) ** 2)


def main():
    lambda_hats = dict(DEFAULT_LAMBDA_HAT)
    try:
        df_mle = pd.read_csv("rezultat_03v2_mle_rejection.csv", index_col=0)
        for name in lambda_hats:
            if name in df_mle.index:
                lambda_hats[name] = df_mle.loc[name, "lambda_hat_combinat"]
        print("S-a citit rezultat_03v2_mle_rejection.csv -- folosesc lambda_hat de-acolo.\n")
    except FileNotFoundError:
        print("Nu am gasit rezultat_03v2_mle_rejection.csv -- folosesc valorile implicite.\n")

    rows = []
    for name, pattern in SOURCES.items():
        lam = lambda_hats[name]
        print(f"=== {name} (lambda_hat = {lam:.6f}) ===")
        digits = load_digits(pattern)
        Us, Vs = generate_all_realizations(digits)
        n = len(Us)
        print(f"Realizari generate: {n}")

        F_sp_theo = F_SP_theoretical(lam)
        F_ps_theo = F_PS_theoretical(lam)
        F_sp_emp = empirical_cdf(Us)
        F_ps_emp = empirical_cdf(Vs)

        max_dF_sp = np.max(np.abs(F_sp_theo - F_sp_emp))
        max_dF_ps = np.max(np.abs(F_ps_theo - F_ps_emp))
        mse_sp = mse(F_sp_theo, F_sp_emp)
        mse_ps = mse(F_ps_theo, F_ps_emp)

        print(f"max|deltaF_SP| = {max_dF_sp:.4f}   max|deltaF_PS| = {max_dF_ps:.4f}")
        print(f"MSE_SP = {mse_sp:.2e}   MSE_PS = {mse_ps:.2e}\n")

        rows.append({
            "Sursa PRNG": name, "lambda_hat": lam, "n_realizari": n,
            "max|deltaF_SP|": round(max_dF_sp, 4), "max|deltaF_PS|": round(max_dF_ps, 4),
            "MSE_SP": mse_sp, "MSE_PS": mse_ps,
        })

    df = pd.DataFrame(rows).set_index("Sursa PRNG")
    print("===== TABEL FINAL (completeaza Tabelul 5) =====")
    print(df.to_string())
    df.to_csv("rezultat_04v2_eroare_cdf_rejection.csv")
    print("\nSalvat in rezultat_04v2_eroare_cdf_rejection.csv")


if __name__ == "__main__":
    main()
