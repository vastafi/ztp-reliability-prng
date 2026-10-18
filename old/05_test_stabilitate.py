"""
Script 5: Test de stabilitate / reproductibilitate (K subseturi independente)
================================================================================

Ideea: nu putem genera cifre noi de la aceleasi surse PRNG (avem un
fisier fix de ~15,7 milioane cifre per sursa), dar putem imparti acest
flux in K bucati DISJUNCTE si rula pipeline-ul complet (generare ZTP
prin rejection sampling + estimare MLE, ca in Scriptul 3 v2) independent
pe fiecare bucata. Daca rezultatele (lambda_hat, sigma) sunt stabile
intre cele K bucati, asta confirma ca rezultatul nu e un artefact al
unei singure treceri prin date, ci reflecta robust distributia ZTP(lambda)
si calitatea sursei PRNG.

Foloseste exact aceeasi metoda de generare ca Scriptul 3 v2 (rejection
sampling pe Poisson simplu, confirmata de coordonator).
"""

import glob
import math
import numpy as np
import pandas as pd

DATA_DIR = "../data"
M = 3
LAMBDA_TRUE = 2.5
DIGITS_PER_UNIFORM = 4
K_SPLITS = 5          # numarul de subseturi independente (poti pune 3)
R_PER_SPLIT = 200     # replicatii MLE in interiorul fiecarui subset (mai mic decat 1000,
                      # pentru ca fiecare subset are ~1/K din date)
GRID = np.linspace(0.05, 8.0, 300)

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


def run_mle_on_subset(Us, Vs, R):
    n_total = len(Us)
    n_per_rep = n_total // R
    if n_per_rep == 0:
        R = max(1, n_total // 10)
        n_per_rep = n_total // R
    lam_sp_hats, lam_ps_hats = [], []
    for r in range(R):
        chunk_U = Us[r * n_per_rep:(r + 1) * n_per_rep]
        chunk_V = Vs[r * n_per_rep:(r + 1) * n_per_rep]
        lam_sp_hats.append(mle_grid_search(chunk_U, pmf_SP))
        lam_ps_hats.append(mle_grid_search(chunk_V, pmf_PS))
    lam_combined = (np.mean(lam_sp_hats) + np.mean(lam_ps_hats)) / 2
    return n_total, n_per_rep, lam_combined


def main():
    all_rows = []
    for name, pattern in SOURCES.items():
        print(f"\n=== {name} ===")
        digits = load_digits(pattern)
        chunk_size = len(digits) // K_SPLITS
        split_lambdas = []
        for k in range(K_SPLITS):
            chunk_digits = digits[k * chunk_size:(k + 1) * chunk_size]
            Us, Vs = generate_all_realizations(chunk_digits)
            n_total, n_per_rep, lam_combined = run_mle_on_subset(Us, Vs, R_PER_SPLIT)
            split_lambdas.append(lam_combined)
            print(f"  Subset {k+1}/{K_SPLITS}: n_realizari={n_total}, lambda_hat_combinat={lam_combined:.4f}")
            all_rows.append({
                "Sursa PRNG": name, "Subset": k + 1, "n_realizari": n_total,
                "lambda_hat_combinat": lam_combined,
            })
        mean_lam = np.mean(split_lambdas)
        std_lam = np.std(split_lambdas)
        rel_range = (max(split_lambdas) - min(split_lambdas)) / mean_lam * 100
        print(f"  --> Media pe cele {K_SPLITS} subseturi: {mean_lam:.4f}, "
              f"std={std_lam:.4f}, interval relativ={rel_range:.2f}%")

    df = pd.DataFrame(all_rows)
    print("\n\n===== TABEL FINAL (stabilitate pe subseturi) =====")
    print(df.to_string(index=False))
    df.to_csv("rezultat_05_stabilitate.csv", index=False)

    summary = df.groupby("Sursa PRNG")["lambda_hat_combinat"].agg(["mean", "std", "min", "max"])
    print("\n===== REZUMAT PE SURSA =====")
    print(summary.to_string())
    summary.to_csv("rezultat_05_stabilitate_rezumat.csv")
    print("\nSalvat in rezultat_05_stabilitate.csv si rezultat_05_stabilitate_rezumat.csv")


if __name__ == "__main__":
    main()
