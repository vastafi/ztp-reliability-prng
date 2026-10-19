"""
01_valori_teoretice.py — caracteristicile teoretice ale lui U (S-P) și V (P-S) pentru Modelele I, II, III
și volumul minim de simulare (Tabelele 2 și 7, Secțiunea 2.6).

Ieșire: results/valori_teoretice.json, results/tabel_valori_teoretice.csv
"""
import json
import pandas as pd
from ztp.config import RESULTS_DIR, MODELS, EPSILON, N_SETS, SET_SIZE
from ztp.models import theory, n_min


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    out, rows = {}, []
    for m in MODELS:
        T = theory(m)
        out[m] = {net: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in T[net].items()} for net in ("SP", "PS")}
        out[m].update(epsilon=EPSILON[m], N_min=n_min(m), B=N_SETS[m], n=SET_SIZE, N=N_SETS[m] * SET_SIZE)
        rows.append({"Model": m, "E(U)": T["SP"]["mean"], "E(V)": T["PS"]["mean"], "D(U)=D(V)": T["SP"]["var"],
                     "sigma": T["SP"]["sd"], "As(U)": T["SP"]["skew"], "Exc": T["SP"]["exc"], "H (biti)": T["SP"]["H"],
                     "epsilon": EPSILON[m], "N_min": n_min(m), "N utilizat": N_SETS[m] * SET_SIZE})
        print(f"Modelul {m}: E(U)={T['SP']['mean']:.6f}  E(V)={T['PS']['mean']:.6f}  D={T['SP']['var']:.6f}  N_min={n_min(m)}")
    json.dump(out, open(RESULTS_DIR / "valori_teoretice.json", "w", encoding="utf-8"), indent=1)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tabel_valori_teoretice.csv", index=False, float_format="%.6f")
    print(f"Salvat: {RESULTS_DIR / 'valori_teoretice.json'}")


if __name__ == "__main__":
    main()
