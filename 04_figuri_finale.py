
"""
04_figuri_finale.py

Generează figurile finale pentru articol, folosind exact aceeași metodologie
ca simularea principală:
    B = 236 seturi
    n = 1000 replici / set
    N = 236000 replici / sursă

Figuri:
  Figura 1 - PMF teoretice pentru U și V
  Figura 2 - CDF teoretice și empirice pentru cele 3 surse
  Figura 3 - Ratele hazardului teoretice și empirice
  Figura 4 - CRQI ponderat pentru U și V, cu abateri standard
  Figura 5 - Clasamentul global după CRQI ponderat (opțional)

Rulați din directorul proiectului, unde există subdirectorul data/.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 1. PARAMETRI
# ============================================================

K = 9
M = 3
LAMBDA = 2.5

B = 236
N_PER_SET = 1000

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figure"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "Java ThreadLocalRandom": [
        DATA_DIR / "java_threadlocal" / "threadlocal_data.csv",
        DATA_DIR / "java_threadlocal" / "threadlocal_data1.csv",
        DATA_DIR / "java_threadlocal" / "threadlocal_data2.csv",
        DATA_DIR / "java_threadlocal" / "threadlocal_data3.csv",
        DATA_DIR / "java_threadlocal" / "threadlocal_data4.csv",
        DATA_DIR / "java_threadlocal" / "threadlocal_data5.csv",
    ],
    "Python secrets": [
        DATA_DIR / "python_secrets" / "secrets_data.csv",
        DATA_DIR / "python_secrets" / "secrets_data1.csv",
        DATA_DIR / "python_secrets" / "secrets_data2.csv",
        DATA_DIR / "python_secrets" / "secrets_data3.csv",
        DATA_DIR / "python_secrets" / "secrets_data4.csv",
        DATA_DIR / "python_secrets" / "secrets_data5.csv",
    ],
    "Cifrele lui π": [
        DATA_DIR / "pi" / "pi_digits_part.csv",
        DATA_DIR / "pi" / "pi_digits_part1.csv",
        DATA_DIR / "pi" / "pi_digits_part2.csv",
        DATA_DIR / "pi" / "pi_digits_part3.csv",
        DATA_DIR / "pi" / "pi_digits_part4.csv",
        DATA_DIR / "pi" / "pi_digits_part5.csv",
    ],
}

GROUP_7 = [
    "mean",
    "entropy",
    "autocorr",
    "chi2",
    "independence",
    "ks",
    "hazard",
]

GROUP_3 = ["var", "As", "Exc"]

WEIGHTS = {name: 9 / 70 for name in GROUP_7}
WEIGHTS.update({name: 1 / 30 for name in GROUP_3})


# ============================================================
# 2. CITIREA CIFRELOR
# ============================================================

def load_digits(files: list[Path]) -> np.ndarray:
    digits: list[int] = []

    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Fișierul nu există: {file_path}\n"
                "Verificați structura directoarelor din DATA_DIR."
            )

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                value = line.strip()
                if value:
                    digit = int(value)
                    if digit < 0 or digit > 9:
                        raise ValueError(
                            f"Valoare nevalidă în {file_path}: {digit}"
                        )
                    digits.append(digit)

    return np.asarray(digits, dtype=np.int8)


class DigitStream:
    def __init__(self, digits: np.ndarray):
        self.digits = digits
        self.position = 0
        self.length = len(digits)

    def next_digit(self) -> int:
        if self.position >= self.length:
            raise RuntimeError(
                "Sursa de cifre a fost epuizată în timpul simulării."
            )

        value = int(self.digits[self.position])
        self.position += 1
        return value

    def next_uniform(self, number_of_digits: int = 5) -> float:
        value = 0

        for _ in range(number_of_digits):
            value = value * 10 + self.next_digit()

        return value / (10 ** number_of_digits)


# ============================================================
# 3. GENERAREA ZTP ȘI SIMULAREA REȚELELOR
# ============================================================

def knuth_poisson(lam: float, stream: DigitStream) -> int:
    threshold = np.exp(-lam)
    product = 1.0
    k = 0

    while True:
        k += 1
        product *= stream.next_uniform()

        if product <= threshold:
            return k - 1


def ztp_via_rejection(lam: float, stream: DigitStream) -> int:
    while True:
        value = knuth_poisson(lam, stream)

        if value > 0:
            return value


def simulate_set(
    stream: DigitStream,
    number_of_replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    U = np.empty(number_of_replicates, dtype=np.int64)
    V = np.empty(number_of_replicates, dtype=np.int64)

    for replicate in range(number_of_replicates):
        subsystem_maxima = np.empty(M, dtype=np.int64)
        subsystem_minima = np.empty(M, dtype=np.int64)

        for subsystem in range(M):
            number_of_units = ztp_via_rejection(LAMBDA, stream)

            lifetimes = np.fromiter(
                (stream.next_digit() for _ in range(number_of_units)),
                dtype=np.int64,
                count=number_of_units,
            )

            subsystem_maxima[subsystem] = lifetimes.max()
            subsystem_minima[subsystem] = lifetimes.min()

        U[replicate] = subsystem_maxima.min()
        V[replicate] = subsystem_minima.max()

    return U, V


# ============================================================
# 4. VALORI TEORETICE
# ============================================================

def A(theta: float) -> float:
    return np.exp(theta) - 1


def F_SP(i: int) -> float:
    if i < 0:
        return 0.0

    return 1 - (
        1 - A(LAMBDA * (i + 1) / (K + 1)) / A(LAMBDA)
    ) ** M


def F_PS(i: int) -> float:
    if i < 0:
        return 0.0

    return (
        1 - A(LAMBDA * (K - i) / (K + 1)) / A(LAMBDA)
    ) ** M


SUPPORT = np.arange(K + 1)

P_SP = np.array(
    [F_SP(i) - F_SP(i - 1) for i in range(K + 1)],
    dtype=float,
)
P_PS = np.array(
    [F_PS(i) - F_PS(i - 1) for i in range(K + 1)],
    dtype=float,
)

P_SP = np.clip(P_SP, 0.0, None)
P_PS = np.clip(P_PS, 0.0, None)

P_SP /= P_SP.sum()
P_PS /= P_PS.sum()

F_SP_THEORY = np.cumsum(P_SP)
F_PS_THEORY = np.cumsum(P_PS)


def hazard(probabilities: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    previous_cdf = np.concatenate(([0.0], cumulative[:-1]))
    survival_before = 1.0 - previous_cdf

    result = np.full_like(probabilities, np.nan, dtype=float)
    valid = survival_before > 0
    result[valid] = probabilities[valid] / survival_before[valid]

    return result


H_SP_THEORY = hazard(P_SP)
H_PS_THEORY = hazard(P_PS)


def theoretical_moments(
    probabilities: np.ndarray,
) -> dict[str, float]:
    mean = np.sum(SUPPORT * probabilities)
    variance = np.sum((SUPPORT - mean) ** 2 * probabilities)
    standard_deviation = np.sqrt(variance)

    skewness = (
        np.sum((SUPPORT - mean) ** 3 * probabilities)
        / standard_deviation ** 3
    )

    excess_kurtosis = (
        np.sum((SUPPORT - mean) ** 4 * probabilities)
        / standard_deviation ** 4
        - 3
    )

    entropy = -np.sum(
        probabilities[probabilities > 0]
        * np.log2(probabilities[probabilities > 0])
    )

    return {
        "mean": float(mean),
        "var": float(variance),
        "As": float(skewness),
        "Exc": float(excess_kurtosis),
        "H": float(entropy),
    }


THEORY_SP = theoretical_moments(P_SP)
THEORY_PS = theoretical_moments(P_PS)


# ============================================================
# 5. CRITERIILE D1-D10
# ============================================================

def lag1_autocorrelation(values: np.ndarray) -> float:
    centered = values.astype(float) - values.mean()
    denominator = np.sum(centered ** 2)

    if denominator == 0:
        return 0.0

    return float(
        np.sum(centered[:-1] * centered[1:]) / denominator
    )


def chi2_independence_p_value(
    values: np.ndarray,
) -> float:
    number_of_categories = K + 1
    observed = np.zeros(
        (number_of_categories, number_of_categories),
        dtype=float,
    )

    for first, second in zip(values[:-1], values[1:]):
        observed[int(first), int(second)] += 1

    row_sums = observed.sum(axis=1, keepdims=True)
    column_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    expected = row_sums @ column_sums / total
    valid = expected > 0

    statistic = np.sum(
        (observed[valid] - expected[valid]) ** 2
        / expected[valid]
    )

    degrees_of_freedom = K ** 2

    return float(
        stats.chi2.sf(statistic, degrees_of_freedom)
    )


def calculate_deviations(
    values: np.ndarray,
    theory: dict[str, float],
    theoretical_pmf: np.ndarray,
    theoretical_cdf: np.ndarray,
    theoretical_hazard: np.ndarray,
) -> dict[str, float]:
    empirical_mean = float(values.mean())
    empirical_variance = float(values.var(ddof=1))
    empirical_skewness = float(stats.skew(values, bias=True))
    empirical_excess = float(
        stats.kurtosis(values, fisher=True, bias=True)
    )

    counts = np.bincount(
        values,
        minlength=K + 1,
    )[: K + 1]

    empirical_pmf = counts / counts.sum()

    positive_probabilities = empirical_pmf[empirical_pmf > 0]
    empirical_entropy = float(
        -np.sum(
            positive_probabilities
            * np.log2(positive_probabilities)
        )
    )

    empirical_cdf = np.cumsum(empirical_pmf)
    empirical_hazard = hazard(empirical_pmf)

    valid_hazard = (
        np.isfinite(empirical_hazard)
        & np.isfinite(theoretical_hazard)
    )

    hazard_deviation = float(
        np.max(
            np.abs(
                empirical_hazard[valid_hazard]
                - theoretical_hazard[valid_hazard]
            )
        )
    )

    _, chi2_p_value = stats.chisquare(
        f_obs=counts,
        f_exp=theoretical_pmf * len(values),
    )

    independence_p_value = chi2_independence_p_value(values)

    return {
        "mean": abs(empirical_mean - theory["mean"]) / K,
        "var": min(
            abs(empirical_variance - theory["var"])
            / theory["var"],
            1.0,
        ),
        "As": min(
            abs(empirical_skewness - theory["As"]),
            1.0,
        ),
        "Exc": min(
            abs(empirical_excess - theory["Exc"])
            / (abs(theory["Exc"]) + 1e-6),
            1.0,
        ),
        "entropy": min(
            abs(empirical_entropy - theory["H"])
            / theory["H"],
            1.0,
        ),
        "hazard": min(hazard_deviation, 1.0),
        "autocorr": min(
            abs(lag1_autocorrelation(values)),
            1.0,
        ),
        "chi2": 1.0 - float(chi2_p_value),
        "ks": float(
            np.max(
                np.abs(empirical_cdf - theoretical_cdf)
            )
        ),
        "independence": 1.0 - independence_p_value,
    }


def weighted_crqi(deviations: dict[str, float]) -> float:
    return 1.0 - sum(
        WEIGHTS[name] * deviations[name]
        for name in deviations
    )


# ============================================================
# 6. SIMULAREA COMPLETĂ ȘI AGREGAREA REZULTATELOR
# ============================================================

def run_full_simulation() -> dict[str, dict]:
    results: dict[str, dict] = {}

    for source_name, files in SOURCES.items():
        digits = load_digits(files)
        stream = DigitStream(digits)

        total_counts_U = np.zeros(K + 1, dtype=np.int64)
        total_counts_V = np.zeros(K + 1, dtype=np.int64)

        crqi_trials_U: list[float] = []
        crqi_trials_V: list[float] = []

        for set_index in range(B):
            U, V = simulate_set(stream, N_PER_SET)

            total_counts_U += np.bincount(
                U,
                minlength=K + 1,
            )[: K + 1]

            total_counts_V += np.bincount(
                V,
                minlength=K + 1,
            )[: K + 1]

            deviations_U = calculate_deviations(
                U,
                THEORY_SP,
                P_SP,
                F_SP_THEORY,
                H_SP_THEORY,
            )

            deviations_V = calculate_deviations(
                V,
                THEORY_PS,
                P_PS,
                F_PS_THEORY,
                H_PS_THEORY,
            )

            crqi_trials_U.append(weighted_crqi(deviations_U))
            crqi_trials_V.append(weighted_crqi(deviations_V))

        empirical_pmf_U = total_counts_U / total_counts_U.sum()
        empirical_pmf_V = total_counts_V / total_counts_V.sum()

        results[source_name] = {
            "pmf_U": empirical_pmf_U,
            "pmf_V": empirical_pmf_V,
            "cdf_U": np.cumsum(empirical_pmf_U),
            "cdf_V": np.cumsum(empirical_pmf_V),
            "hazard_U": hazard(empirical_pmf_U),
            "hazard_V": hazard(empirical_pmf_V),
            "crqi_U_mean": float(np.mean(crqi_trials_U)),
            "crqi_U_std": float(
                np.std(crqi_trials_U, ddof=1)
            ),
            "crqi_V_mean": float(np.mean(crqi_trials_V)),
            "crqi_V_std": float(
                np.std(crqi_trials_V, ddof=1)
            ),
        }

        results[source_name]["crqi_global"] = (
            results[source_name]["crqi_U_mean"]
            + results[source_name]["crqi_V_mean"]
        ) / 2

        print(
            f"{source_name}: "
            f"utilizate {stream.position:,} din {stream.length:,} cifre "
            f"({100 * stream.position / stream.length:.2f}%)."
        )

    return results


# ============================================================
# 7. FIGURA 1 - PMF TEORETICE
# ============================================================

def figure_1_theoretical_pmf() -> None:
    plt.figure(figsize=(9, 5.5))

    plt.plot(
        SUPPORT,
        P_SP,
        marker="o",
        linewidth=2,
        label="Rețeaua S-P, U",
    )

    plt.plot(
        SUPPORT,
        P_PS,
        marker="s",
        linewidth=2,
        label="Rețeaua P-S, V",
    )

    plt.xlabel("Durata de viață i")
    plt.ylabel("Probabilitatea p(i)")
    plt.xticks(SUPPORT)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = FIGURE_DIR / "figura1_pmf_teoretice.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Salvată: {output}")


# ============================================================
# 8. FIGURA 2 - CDF TEORETICE ȘI EMPIRICE
# ============================================================

def figure_2_cdf(results: dict[str, dict]) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharey=True,
    )

    axes[0].step(
        SUPPORT,
        F_SP_THEORY,
        where="post",
        linewidth=3,
        label="Teoretic",
    )

    for source_name, values in results.items():
        axes[0].step(
            SUPPORT,
            values["cdf_U"],
            where="post",
            linewidth=1.8,
            linestyle="--",
            label=source_name,
        )

    axes[0].set_title("(a) Rețeaua S-P, U")
    axes[0].set_xlabel("Durata de viață i")
    axes[0].set_ylabel("Funcția de repartiție F(i)")
    axes[0].set_xticks(SUPPORT)
    axes[0].grid(alpha=0.3)

    axes[1].step(
        SUPPORT,
        F_PS_THEORY,
        where="post",
        linewidth=3,
        label="Teoretic",
    )

    for source_name, values in results.items():
        axes[1].step(
            SUPPORT,
            values["cdf_V"],
            where="post",
            linewidth=1.8,
            linestyle="--",
            label=source_name,
        )

    axes[1].set_title("(b) Rețeaua P-S, V")
    axes[1].set_xlabel("Durata de viață i")
    axes[1].set_xticks(SUPPORT)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
    )

    figure.tight_layout(rect=(0, 0.08, 1, 1))

    output = FIGURE_DIR / "figura2_cdf_teoretic_empiric.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Salvată: {output}")


# ============================================================
# 9. FIGURA 3 - HAZARD TEORETIC ȘI EMPIRIC
# ============================================================

def figure_3_hazard(results: dict[str, dict]) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharey=True,
    )

    axes[0].plot(
        SUPPORT,
        H_SP_THEORY,
        marker="o",
        linewidth=3,
        label="Teoretic",
    )

    for source_name, values in results.items():
        axes[0].plot(
            SUPPORT,
            values["hazard_U"],
            marker=".",
            linewidth=1.8,
            linestyle="--",
            label=source_name,
        )

    axes[0].set_title("(a) Rețeaua S-P, U")
    axes[0].set_xlabel("Durata de viață i")
    axes[0].set_ylabel("Rata hazardului h(i)")
    axes[0].set_xticks(SUPPORT)
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        SUPPORT,
        H_PS_THEORY,
        marker="s",
        linewidth=3,
        label="Teoretic",
    )

    for source_name, values in results.items():
        axes[1].plot(
            SUPPORT,
            values["hazard_V"],
            marker=".",
            linewidth=1.8,
            linestyle="--",
            label=source_name,
        )

    axes[1].set_title("(b) Rețeaua P-S, V")
    axes[1].set_xlabel("Durata de viață i")
    axes[1].set_xticks(SUPPORT)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.03),
    )

    figure.tight_layout(rect=(0, 0.08, 1, 1))

    output = FIGURE_DIR / "figura3_hazard_teoretic_empiric.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Salvată: {output}")


# ============================================================
# 10. FIGURA 4 - CRQI PONDERAT CU ABATERI STANDARD
# ============================================================

def figure_4_crqi_weighted(results: dict[str, dict]) -> None:
    source_names = list(results.keys())
    x_positions = np.arange(len(source_names))
    width = 0.36

    crqi_U = np.array(
        [results[name]["crqi_U_mean"] for name in source_names]
    )

    crqi_V = np.array(
        [results[name]["crqi_V_mean"] for name in source_names]
    )

    std_U = np.array(
        [results[name]["crqi_U_std"] for name in source_names]
    )

    std_V = np.array(
        [results[name]["crqi_V_std"] for name in source_names]
    )

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        x_positions - width / 2,
        crqi_U,
        width,
        yerr=std_U,
        capsize=5,
        label="Rețeaua S-P, U",
    )

    axis.bar(
        x_positions + width / 2,
        crqi_V,
        width,
        yerr=std_V,
        capsize=5,
        label="Rețeaua P-S, V",
    )

    axis.set_ylabel("CRQI ponderat")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(source_names)
    axis.set_ylim(0.75, 0.95)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output = FIGURE_DIR / "figura4_crqi_ponderat.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Salvată: {output}")


# ============================================================
# 11. FIGURA 5 - CLASAMENT GLOBAL, OPȚIONAL
# ============================================================

def figure_5_global_ranking(results: dict[str, dict]) -> None:
    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["crqi_global"],
        reverse=True,
    )

    source_names = [item[0] for item in ranking]
    global_scores = [item[1]["crqi_global"] for item in ranking]

    figure, axis = plt.subplots(figsize=(9, 5.5))

    bars = axis.bar(source_names, global_scores)

    axis.set_ylabel("CRQI ponderat global")
    axis.set_ylim(0.84, 0.87)
    axis.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, global_scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.00025,
            f"{score:.6f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    output = FIGURE_DIR / "figura5_clasament_global.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Salvată: {output}")


# ============================================================
# 12. SALVAREA DATELOR PENTRU REUTILIZARE
# ============================================================

def save_results(results: dict[str, dict]) -> None:
    serializable = {}

    for source_name, values in results.items():
        serializable[source_name] = {}

        for key, value in values.items():
            if isinstance(value, np.ndarray):
                serializable[source_name][key] = value.tolist()
            else:
                serializable[source_name][key] = float(value)

    output = BASE_DIR / "figure_results.json"

    with output.open("w", encoding="utf-8") as file:
        json.dump(
            serializable,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Rezultatele figurilor au fost salvate în: {output}")


# ============================================================
# 13. EXECUȚIE
# ============================================================

def main() -> None:
    figure_1_theoretical_pmf()

    results = run_full_simulation()

    figure_2_cdf(results)
    figure_3_hazard(results)
    figure_4_crqi_weighted(results)
    figure_5_global_ranking(results)

    save_results(results)

    print("\nToate figurile au fost generate cu succes.")


if __name__ == "__main__":
    main()
