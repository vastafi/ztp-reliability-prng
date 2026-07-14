"""
Script 1: Testul chi^2 de concordanta cu distributia uniforma discreta
=======================================================================

Reproduce Tabelul 1 din articol (sectiunea 2.4 / 4.1).

Foloseste: data/java_threadlocal/*.csv, data/python_secrets/*.csv, data/pi/*.csv
(dezarhiveaza data.zip in acelasi folder cu acest script, sau ajusteaza DATA_DIR).

Fiecare fisier CSV are o singura coloana, o cifra (0-9) pe linie.
"""

import glob
import numpy as np
import pandas as pd
from scipy.stats import chisquare

DATA_DIR = "data"  # ajusteaza daca ai dezarhivat in alta parte

SOURCES = {
    "Java ThreadLocalRandom": f"{DATA_DIR}/java_threadlocal/*.csv",
    "Python secrets":         f"{DATA_DIR}/python_secrets/*.csv",
    "Cifrele lui pi":         f"{DATA_DIR}/pi/*.csv",
}


def load_digits(pattern: str) -> np.ndarray:
    """Incarca toate fisierele CSV care se potrivesc cu pattern-ul si le concateneaza."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Niciun fisier gasit pentru: {pattern}")
    frames = [pd.read_csv(f, header=None).iloc[:, 0].values for f in files]
    digits = np.concatenate(frames)
    if digits.min() < 0 or digits.max() > 9:
        raise ValueError("Cifrele trebuie sa fie in intervalul 0-9")
    return digits.astype(int)


def chi2_uniform_test(digits: np.ndarray):
    n = len(digits)
    observed = np.bincount(digits, minlength=10)
    expected = np.full(10, n / 10)
    stat, p = chisquare(observed, expected)
    return n, observed, expected, stat, p


def main():
    rows = []
    for name, pattern in SOURCES.items():
        digits = load_digits(pattern)
        n, observed, expected, stat, p = chi2_uniform_test(digits)
        rows.append({
            "Sursa PRNG": name,
            "n (cifre)": n,
            "chi2 calculat": round(stat, 4),
            "chi2 critic (df=9, a=0.05)": 16.919,
            "p-valoare": round(p, 4),
            "Decizie": "Nu respinge H0" if stat < 16.919 else "Respinge H0",
        })
        print(f"\n=== {name} ===")
        print(f"n = {n}")
        print(f"Frecvente observate (cifre 0-9): {observed}")
        print(f"chi2 = {stat:.4f}   p = {p:.4f}")

    df = pd.DataFrame(rows)
    print("\n\n===== TABEL FINAL (compara cu Tabelul 1 din articol) =====")
    print(df.to_string(index=False))
    df.to_csv("rezultat_01_chi2.csv", index=False)
    print("\nSalvat in rezultat_01_chi2.csv")


if __name__ == "__main__":
    main()
