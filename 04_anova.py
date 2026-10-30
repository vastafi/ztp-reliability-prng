"""
04_anova.py — testează dacă sursele PRNG diferă semnificativ prin CRQI ponderat pe seturi
(ANOVA cu un factor și testul Kruskal-Wallis, cu mărimile efectului eta^2 și epsilon^2),
pentru fiecare model și rețea; se compară doar cele 3 surse reale.

Rulați după 02_simulare_crqi.py. Ieșire: results/anova.json
"""
import json
import numpy as np
from scipy import stats
from ztp.config import RESULTS_DIR, MODELS, SOURCES


def main():
    out = {}
    for m in MODELS:
        r = json.load(open(RESULTS_DIR / f"rezultate_model_{m}.json", encoding="utf-8"))["sources"]
        for net in ("SP", "PS", "global"):
            groups = {}
            for name, s in r.items():
                if name not in SOURCES:          # doar sursele reale; controalele sunt analizate în 05
                    continue
                sp, ps = np.array(s["SP"]["crqi_b"]), np.array(s["PS"]["crqi_b"])
                groups[name] = (sp + ps) / 2 if net == "global" else (sp if net == "SP" else ps)
            F, pF = stats.f_oneway(*groups.values())
            Hk, pK = stats.kruskal(*groups.values())
            allv = np.concatenate(list(groups.values())); n = len(allv)
            ss_b = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups.values())
            eta2 = ss_b / ((allv - allv.mean()) ** 2).sum()          # mărimea efectului ANOVA
            eps2 = Hk / (n - 1)                                       # mărimea efectului Kruskal-Wallis
            out[f"{m}|{net}"] = dict(F=float(F), p_anova=float(pF), H=float(Hk), p_kruskal=float(pK),
                                    eta2=float(eta2), epsilon2=float(eps2),
                                    means={k: float(v.mean()) for k, v in groups.items()})
            print(f"Modelul {m:3} {net:6}: ANOVA F={F:.3f} p={pF:.3f} eta2={eta2:.4f} | Kruskal-Wallis H={Hk:.3f} p={pK:.3f} eps2={eps2:.4f}")
    json.dump(out, open(RESULTS_DIR / "anova.json", "w", encoding="utf-8"), indent=1)
    print(f"Salvat: {RESULTS_DIR / 'anova.json'}")


if __name__ == "__main__":
    main()
