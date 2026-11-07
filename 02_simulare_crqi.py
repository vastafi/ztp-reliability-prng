"""
02_simulare_crqi.py — simularea Monte Carlo și indicele CRQI pentru Modelele I, II, III și cele 3 surse.

    python 02_simulare_crqi.py          # toate modelele (~1 minut)
    python 02_simulare_crqi.py II       # un singur model

Ieșire (în results/):
    rezultate_model_I.json, ..._II.json, ..._III.json   — toate valorile (deviații, CRQI, CRQI pe seturi, PMF empirice)
    tabel_deviatii_model_*.csv                          — Tabelele 3, 4 (Modelul I), 9, 10 (Modelele II, III)
    tabel_crqi.csv                                      — Tabelele 5, 6 și 8
Pe lângă cele 3 surse reale se simulează și 2 generatoare de control cu defecte cunoscute
(ztp/controls.py: RANDU și LCG mod 10), ca să se verifice că metoda detectează un generator slab.
Pentru fiecare model, simularea pornește de la începutul șirului de cifre al fiecărei surse.
"""
import json
import sys
import numpy as np
import pandas as pd
from ztp.config import RESULTS_DIR, MODELS, SOURCES, N_SETS, SET_SIZE, CRITERIA, K
from ztp.models import theory
from ztp.simulation import load_digits, simulate
from ztp.controls import CONTROLS, N_CONTROL_DIGITS
from ztp.criteria import deviations, crqi
from ztp.two_level import chi_con, chi_ind, null_stats, p_calibrated, holm_significant

SEED_NULL = 2026


def run(model, digits_cache):
    T = theory(model)
    out = {"model": model,
           "theory_U": {k: T["SP"][k] for k in ("mean", "var", "skew", "exc", "H")},
           "theory_V": {k: T["PS"][k] for k in ("mean", "var", "skew", "exc", "H")},
           "sources": {}}
    all_sources = {**{k: ("file", v) for k, v in SOURCES.items()}, **{k: ("gen", g) for k, g in CONTROLS.items()}}
    rng = np.random.default_rng(SEED_NULL)
    nulls = {net: null_stats(T[net]["p"], rng) for net in ("SP", "PS")}
    ks_rows = []  # (name, net, crit) -> p-value KS, pentru corecția Holm pe toată familia modelului
    for name, (kind, src) in all_sources.items():
        if name not in digits_cache:
            digits_cache[name] = load_digits(src) if kind == "file" else src(N_CONTROL_DIGITS)
        digits = digits_cache[name]
        U, V, used = simulate(digits, model, N_SETS[model] * SET_SIZE)
        B = len(U) // SET_SIZE
        res = {"replicas": len(U), "digits_used": used, "digits_total": len(digits), "sets": B}
        for net, X in (("SP", U), ("PS", V)):
            sets = [X[b * SET_SIZE:(b + 1) * SET_SIZE] for b in range(B)]
            D = np.array([deviations(s, T[net]) for s in sets])
            res[net] = crqi(D)
            res[net].update(emp_mean=float(X.mean()), emp_pmf=(np.bincount(X, minlength=K + 1) / len(X)).tolist())
            # D7/D9: test de semnificație separat (p calibrat prin simulare, nu agregat în CRQI)
            from scipy import stats as _stats
            p7 = p_calibrated(np.array([chi_con(s, T[net]["p"]) for s in sets]), nulls[net][0])
            p9 = p_calibrated(np.array([chi_ind(s) for s in sets]), nulls[net][1])
            ks7 = float(_stats.kstest(p7, "uniform").pvalue)
            ks9 = float(_stats.kstest(p9, "uniform").pvalue)
            res[net].update(ks_D7=ks7, ks_D9=ks9)
            ks_rows.append((name, net, "D7", ks7)); ks_rows.append((name, net, "D9", ks9))
        out["sources"][name] = res
        print(f"Modelul {model} | {name}: {len(U)} replici, cifre {used:,}/{len(digits):,} | "
              f"CRQI ponderat S-P {res['SP']['CRQI_ponderat']:.6f}, P-S {res['PS']['CRQI_ponderat']:.6f}", flush=True)
    # corecție Holm pe toată familia de teste D7/D9 ale modelului (surse x rețele x criterii)
    sig = holm_significant([r[3] for r in ks_rows])
    for (name, net, crit, _), s in zip(ks_rows, sig):
        out["sources"][name][net][f"holm_{crit}"] = bool(s)
    out["holm_family_size"] = len(ks_rows)
    out["holm_significant_count"] = int(sig.sum())
    path = RESULTS_DIR / f"rezultate_model_{model}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print(f"Salvat: {path} | Holm: {int(sig.sum())}/{len(ks_rows)} semnificative (alpha=0,05)")
    # tabelul deviațiilor (D1-D6,D8,D10)
    cols = {f"{net} {name}": out["sources"][name][net]["dbar"] for net in ("SP", "PS") for name in out["sources"]}
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
                         "CRQI ponderat global": (s["SP"]["CRQI_ponderat"] + s["PS"]["CRQI_ponderat"]) / 2,
                         "KS2 S-P D7": s["SP"]["ks_D7"], "KS2 S-P D9": s["SP"]["ks_D9"],
                         "KS2 P-S D7": s["PS"]["ks_D7"], "KS2 P-S D9": s["PS"]["ks_D9"],
                         "Holm S-P D7": s["SP"]["holm_D7"], "Holm S-P D9": s["SP"]["holm_D9"],
                         "Holm P-S D7": s["PS"]["holm_D7"], "Holm P-S D9": s["PS"]["holm_D9"]})
    df = pd.DataFrame(rows)
    df["Control"] = df["Sursa"].str.contains("control")
    df["Loc"] = df[~df["Control"]].groupby("Model")["CRQI ponderat global"].rank(ascending=False).astype("Int64")
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
