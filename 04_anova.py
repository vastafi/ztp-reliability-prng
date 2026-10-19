"""
04_anova.py — testează dacă sursele PRNG diferă semnificativ prin CRQI ponderat pe seturi
(ANOVA cu un factor și testul Kruskal-Wallis), pentru fiecare model și rețea.

Rulați după 02_simulare_crqi.py. Ieșire: results/anova.json
"""
import json
import numpy as np
from scipy import stats
from ztp.config import RESULTS_DIR, MODELS


def main():
    out = {}
    for m in MODELS:
        r = json.load(open(RESULTS_DIR / f"rezultate_model_{m}.json", encoding="utf-8"))["sources"]
        for net in ("SP", "PS", "global"):
            groups = {}
            for name, s in r.items():
                sp, ps = np.array(s["SP"]["crqi_b"]), np.array(s["PS"]["crqi_b"])
                groups[name] = (sp + ps) / 2 if net == "global" else (sp if net == "SP" else ps)
            F, pF = stats.f_oneway(*groups.values())
            Hk, pK = stats.kruskal(*groups.values())
            out[f"{m}|{net}"] = dict(F=float(F), p_anova=float(pF), H=float(Hk), p_kruskal=float(pK),
                                    means={k: float(v.mean()) for k, v in groups.items()})
            print(f"Modelul {m:3} {net:6}: ANOVA F={F:.3f} p={pF:.3f} | Kruskal-Wallis H={Hk:.3f} p={pK:.3f}")
    json.dump(out, open(RESULTS_DIR / "anova.json", "w", encoding="utf-8"), indent=1)
    print(f"Salvat: {RESULTS_DIR / 'anova.json'}")


if __name__ == "__main__":
    main()
