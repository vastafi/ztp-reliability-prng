"""
Script 4: Eroarea CDF empiric vs. teoretic (max|deltaF|)
===========================================================

Foloseste realizarile de sistem (U, V) generate cu aceeasi metodologie
ca in Scriptul 3 (03_generare_ZTP_si_MLE.py) si le compara cu CDF-ul
teoretic F_S-P / F_P-S evaluat la lambda_hat estimat (Tabelul 2).

Reproduce coloanele "max|deltaF_SP|" si "max|deltaF_PS|" din Tabelul 5
(mentionate la concluzia (5)/(6) din articol).

Ruleaza independent de Scriptul 3 (nu necesita sa il fi rulat inainte),
dar foloseste ACEEASI schema de generare -- daca ai deja rezultat_03_mle.csv
salvat, valorile lambda_hat de-acolo sunt folosite automat ca referinta.
"""

import glob
import numpy as np
import pandas as pd
import math

DATA_DIR = "../data"
M = 3

SOURCES = {
    "Java ThreadLocalRandom": f"{DATA_DIR}/java_threadlocal/*.csv",
    "Python secrets":         f"{DATA_DIR}/python_secrets/*.csv",
    "Cifrele lui pi":         f"{DATA_DIR}/pi/*.csv",
}

# Valorile lambda_hat_combinat obtinute deja (din rulare anterioara a
# Scriptului 3). Daca ai rezultat_03_mle.csv, scriptul le citeste automat;
# altfel foloseste aceste valori implicite (cele raportate).
DEFAULT_LAMBDA_HAT = {
    "Java ThreadLocalRandom": 2.497045,
    "Python secrets":         2.499783,
    "Cifrele lui pi":         2.501764,
}

LAMBDA_TRUE = 2.5  # acelasi parametru folosit la generarea sintetica (Script 3)


# ---------------------------------------------------------------------
# Reincarcam exact aceleasi functii de generare ca in Scriptul 3
# (copiate aici ca sa scriptul fie de sine statator)
# ---------------------------------------------------------------------
def load_digits(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Niciun fisier gasit pentru: {pattern}")
    frames = [pd.read_csv(f, header=None).iloc[:, 0].values for f in files]
    return np.concatenate(frames).astype(int)


def ztp_cdf_table(lam, k_max=60):
    ks = np.arange(1, k_max + 1)
    unnorm = np.array([lam**k * math.exp(-lam) / math.factorial(k) for k in ks])
    unnorm /= (1 - math.exp(-lam))
    cdf = np.cumsum(unnorm)
    return ks, cdf


ZTP_KS, ZTP_CDF = ztp_cdf_table(LAMBDA_TRUE)


def digits_to_uniform(block: np.ndarray) -> float:
    s = "".join(str(d) for d in block)
    return int(s) / 1e9


def sample_ztp_from_uniform(u: float) -> int:
    idx = np.searchsorted(ZTP_CDF, u)
    idx = min(idx, len(ZTP_KS) - 1)
    return int(ZTP_KS[idx])


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


def generate_one_system_realization(stream: DigitStream, M_=M):
    T_max, T_min = [], []
    for _ in range(M_):
        block = stream.take(9)
        u_cont = digits_to_uniform(block)
        Nk = sample_ztp_from_uniform(u_cont)
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
# CDF teoretic F_S-P / F_P-S (formulele (9)-(10) din articol)
# ---------------------------------------------------------------------
def F_base(i):
    return (i + 1) / 10


def F_SP_theoretical(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    return 1 - (1 - (np.exp(lam * F) - 1) / (np.exp(lam) - 1)) ** M_


def F_PS_theoretical(lam, M_=M):
    F = np.array([F_base(i) for i in range(10)])
    return (1 - (np.exp(lam * (1 - F)) - 1) / (np.exp(lam) - 1)) ** M_


def empirical_cdf(sample: np.ndarray) -> np.ndarray:
    """CDF empiric F_hat(k) = P(sample <= k), pentru k = 0..9."""
    n = len(sample)
    return np.array([np.sum(sample <= k) / n for k in range(10)])


def mse(a, b):
    return np.mean((a - b) ** 2)


def main():
    # incearca sa citesti lambda_hat din rularea anterioara a Scriptului 3
    lambda_hats = dict(DEFAULT_LAMBDA_HAT)
    try:
        df_mle = pd.read_csv("rezultat_03_mle.csv", index_col=0)
        for name in lambda_hats:
            if name in df_mle.index:
                lambda_hats[name] = df_mle.loc[name, "lambda_hat_combinat"]
        print("S-a citit rezultat_03_mle.csv -- folosesc lambda_hat de-acolo.\n")
    except FileNotFoundError:
        print("Nu am gasit rezultat_03_mle.csv -- folosesc valorile implicite.\n")

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
    print("===== TABEL FINAL (completeaza Tabelul 5 / concluzia (5)-(6)) =====")
    print(df.to_string())
    df.to_csv("rezultat_04_eroare_cdf.csv")
    print("\nSalvat in rezultat_04_eroare_cdf.csv")


if __name__ == "__main__":
    main()
