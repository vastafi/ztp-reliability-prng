"""Parametri, căi și surse de date — un singur loc pentru toate scripturile."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figure"

# ---- model ----
K = 9                 # duratele unităților: uniform pe {0, 1, ..., K}
LAMBDA = 2.5          # N_k ~ ZTP(lambda)   (Modelele I și III)
MU = 2.5              # M   ~ ZTP(mu)       (Modelele II și III)
M_FIX = 3             # Modelul I:  M constant
N_FIX = 3             # Modelul II: N constant
MODELS = ("I", "II", "III")
MODEL_TITLES = {"I": "Modelul I: M = 3, N ~ ZTP(2,5)",
                "II": "Modelul II: N = 3, M ~ ZTP(2,5)",
                "III": "Modelul III: M, N ~ ZTP(2,5)"}

# ---- volumul simulării (Secțiunea 2.6) ----
Z = 2.05375                                  # cuantila de ordin 0,98 (IC bilateral 96%)
EPSILON = {"I": 0.01, "II": 0.01, "III": 0.0121}
N_SETS = {"I": 236, "II": 236, "III": 199}   # B seturi
SET_SIZE = 1000                              # n replici / set
DIGITS_PER_UNIFORM = 5                       # uniforma (0,1) din 5 cifre consecutive (Knuth)

# ---- CRQI: ponderi D1..D10 (7 x 9/70, 3 x 1/30) ----
CRITERIA = ["D1 - Medie", "D2 - Dispersie", "D3 - Asimetrie", "D4 - Exces de boltire",
            "D5 - Entropie", "D6 - Autocorelație", "D7 - chi2 concordanță", "D8 - KS",
            "D9 - chi2 independență", "D10 - Hazard"]
WEIGHTS = [9/70, 1/30, 1/30, 1/30, 9/70, 9/70, 9/70, 9/70, 9/70, 9/70]

# ---- sursele de cifre (ordinea fișierelor contează: definește șirul de cifre) ----
SOURCES = {
    "Java ThreadLocalRandom": [DATA_DIR / "java_threadlocal" / f"threadlocal_data{s}.csv" for s in ["", "1", "2", "3", "4", "5"]],
    "Python secrets": [DATA_DIR / "python_secrets" / f"secrets_data{s}.csv" for s in ["", "1", "2", "3", "4", "5"]],
    "Cifrele lui pi": [DATA_DIR / "pi" / f"pi_digits_part{s}.csv" for s in ["", "1", "2", "3", "4", "5"]],
}
SOURCE_LABELS = {"Java ThreadLocalRandom": "Java ThreadLocalRandom",
                 "Python secrets": "Python secrets",
                 "Cifrele lui pi": "Cifrele lui π"}

# ---- extinderea la 10 surse (Secțiunea 3.5): setul public PRNG-Random-Digits-0-9-Dataset ----
#   git clone https://github.com/vastafi/PRNG-Random-Digits-0-9-Dataset.git   (lângă acest repozitoriu)
# sau setați variabila de mediu ZTP_DATASET cu calea către setul de date.
import os
DATASET_DIR = Path(os.environ.get("ZTP_DATASET", ROOT.parent / "PRNG-Random-Digits-0-9-Dataset"))
SOURCE_CLASSES = {
    "A": "PRNG algoritmic (implicit într-un mediu de calcul)",
    "B": "CSPRNG (criptografic sigur)",
    "C": "Secvență matematică deterministă",
    "D": "TRNG (sursă fizică)",
}
# nume -> (clasa, folder, prefixul fișierelor); fișierele: prefix + "", "1", ..., "5" (aceeași ordine ca SOURCES)
SOURCES_10 = {
    "C++ rand()":             ("A", "C++", "cpp_random"),
    "Excel RANDBETWEEN":      ("A", "Excel", "excel_num"),
    "Java ThreadLocalRandom": ("A", "Java thread local", "threadlocal_data"),
    "Mathematica":            ("A", "Mathematica", "mathematica_random"),
    "NumPy default_rng":      ("A", "Python numpy", "numpy_data"),
    "R sample":               ("A", "R", "r_numbers"),
    "Java SecureRandom":      ("B", "Java secure random", "securerandom_data"),
    "Python secrets":         ("B", "Python secrets", "secrets_data"),
    "Cifrele lui pi":         ("C", "Pi", "pi_digits_part"),
    "Random.org":             ("D", "Randomorg", "random_org"),
}


def source_files_10(name):
    _, folder, prefix = SOURCES_10[name]
    return [DATASET_DIR / folder / f"{prefix}{s}.csv" for s in ["", "1", "2", "3", "4", "5"]]
