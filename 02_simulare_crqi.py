"""
02_simulare_crqi.py — simularea Monte Carlo și indicele CRQI pentru Modelele I, II, III și cele 3 surse.

    python 02_simulare_crqi.py          # toate modelele (~1 minut)
    python 02_simulare_crqi.py II       # un singur model

Ieșire (în results/):
    rezultate_model_I.json, ..._II.json, ..._III.json   — toate valorile (deviații, CRQI, CRQI pe seturi, PMF empirice)
    tabel_deviatii_model_*.csv                          — Tabelele 3, 4 (Modelul I), 9, 10 (Modelele II, III)
    tabel_crqi.csv                                      — Tabelele 5, 6 și 8
Pentru fiecare model, simularea pornește de la începutul șirului de cifre al fiecărei surse.
"""
import json
import sys
import numpy as np
import pandas as pd
from ztp.config import RESULTS_DIR, MODELS, SOURCES, N_SETS, SET_SIZE, CRITERIA, K
from ztp.models import theory
from ztp.simulation import load_digits, simulate
from ztp.criteria import deviations, crqi


def run(model, digits_cache):
    T = theory(model)
    out = {"model": model,
           "theory_U": {k: T["SP"][k] for k in ("mean", "var", "skew", "exc", "H")},
           "theory_V": {k: T["PS"][k] for k in ("mean", "var", "skew", "exc", "H")},
           "sources": {}}
    for name, files in SOURCES.items():
        digits = digits_cache.setdefault(name, load_digits(files))
        U, V, used = simulate(digits, model, N_SETS[model] * SET_SIZE)
        B = len(U) // SET_SIZE
        res = {"replicas": len(U), "digits_used": used, "digits_total": len(digits), "sets": B}
        for net, X in (("SP", U), ("PS", V)):
            D = np.array([deviations(X[b * SET_SIZE:(b + 1) * SET_SIZE], T[net]) for b in range(B)])
            res[net] = crqi(D)
            res[net].update(emp_mean=float(X.mean()), emp_pmf=(np.bincount(X, minlength=K + 1) / len(X)).tolist())
        out["sources"][name] = res
        print(f"Modelul {model} | {name}: {len(U)} replici, cifre {used:,}/{len(digits):,} | "
              f"CRQI ponderat S-P {res['SP']['CRQI_ponderat']:.6f}, P-S {res['PS']['CRQI_ponderat']:.6f}", flush=True)
    path = RESULTS_DIR / f"rezultate_model_{model}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"Salvat: {path}")
    # tabelul deviațiilor
    cols = {f"{net} {name}": out["sources"][name][net]["dbar"] for net in ("SP", "PS") for name in SOURCES}
    pd.DataFrame(cols, index=CRITERIA).to_csv(RESULTS_DIR / f"tabel_deviatii_model_{model}.csv", float_format="%.6f")
    return out


def crqi_table():
    rows = []
    for m in MODELS:
        path = RESULTS_DIR / f"rezultate_model_{m}.json"
        if not path.exists():
            continue
        r = json.load(open(path, encoding="utf-8"))
        for name, s in r["sources"].items():
            rows.append({"Model": m, "Sursa": name,
                         "CRQI simplu S-P": s["SP"]["CRQI_simplu"], "CRQI ponderat S-P": s["SP"]["CRQI_ponderat"], "s S-P": s["SP"]["sd_ponderat"],
                         "CRQI simplu P-S": s["PS"]["CRQI_simplu"], "CRQI ponderat P-S": s["PS"]["CRQI_ponderat"], "s P-S": s["PS"]["sd_ponderat"],
                         "CRQI simplu global": (s["SP"]["CRQI_simplu"] + s["PS"]["CRQI_simplu"]) / 2,
                         "CRQI ponderat global": (s["SP"]["CRQI_ponderat"] + s["PS"]["CRQI_ponderat"]) / 2})
    df = pd.DataFrame(rows)
    df["Loc"] = df.groupby("Model")["CRQI ponderat global"].rank(ascending=False).astype(int)
    df.to_csv(RESULTS_DIR / "tabel_crqi.csv", index=False, float_format="%.6f")
    print(f"Salvat: {RESULTS_DIR / 'tabel_crqi.csv'}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    models = [sys.argv[1]] if len(sys.argv) > 1 else list(MODELS)
    cache = {}
    for m in models:
        run(m, cache)
    crqi_table()


if __name__ == "__main__":
    main()
