"""
05_sensibilitate.py — analize suplimentare cerute de recenzare:

  1. Detectarea generatoarelor de control (RANDU, LCG mod 10): test Mann–Whitney între CRQI global
     pe seturi al fiecărui generator de control și cel al celor 3 surse reale.
  2. Sensibilitatea clasamentului la ponderi: 10 000 de vectori de ponderi Dirichlet(1,...,1),
     agregare aritmetică (CRQI) și geometrică; frecvența locului 1 pentru fiecare sursă reală.
  3. Ordinea stocastică uzuală între U (S-P) și V (P-S): semnul lui R_U(i) - R_V(i) pe tot suportul.

Rulați după 02_simulare_crqi.py. Ieșire: results/sensibilitate.json
"""
import json
import numpy as np
from scipy import stats
from ztp.config import RESULTS_DIR, MODELS, SOURCES, WEIGHTS
from ztp.models import theory

REAL = list(SOURCES)
N_DRAWS = 10_000
rng = np.random.default_rng(2026)


def global_per_set(s):
    return (np.array(s["SP"]["crqi_b"]) + np.array(s["PS"]["crqi_b"])) / 2


def main():
    out = {}
    W = rng.dirichlet(np.ones(10), size=N_DRAWS)          # aceleași ponderi pentru toate modelele
    for m in MODELS:
        r = json.load(open(RESULTS_DIR / f"rezultate_model_{m}.json", encoding="utf-8"))["sources"]
        res = {}
        # 1. controale
        pool = np.concatenate([global_per_set(r[s]) for s in REAL])
        res["control"] = {}
        for c in [k for k in r if k not in REAL]:
            x = global_per_set(r[c])
            mw = stats.mannwhitneyu(x, pool, alternative="two-sided")
            res["control"][c] = dict(crqi_global=float(x.mean()), crqi_reale=float(pool.mean()), p_mann_whitney=float(mw.pvalue))
        # 2. sensibilitatea ponderilor (pe deviațiile medii, global = media S-P și P-S)
        names = list(r)
        D = np.array([(np.array(r[s]["SP"]["dbar"]) + np.array(r[s]["PS"]["dbar"])) / 2 for s in names])   # (surse, 10)
        arit = 1 - W @ D.T                                        # (draws, surse)
        geom = np.exp(W @ np.log(1 - np.clip(D, 0, 0.999999)).T)
        idx_real = [names.index(s) for s in REAL]
        res["ponderi"] = {}
        for lab, sc in (("aritmetica", arit), ("geometrica", geom)):
            best = np.array(REAL)[sc[:, idx_real].argmax(1)]
            res["ponderi"][lab] = {s: float((best == s).mean()) for s in REAL}
            res["ponderi"][lab + "_control_ultimul"] = {c: float((sc[:, names.index(c)] <= sc.min(1)).mean())
                                                        for c in names if c not in REAL}
        w = np.array(WEIGHTS)
        res["ponderi"]["geometrica_ponderi_articol"] = {s: float(np.exp(w @ np.log(1 - D[names.index(s)]))) for s in names}
        # 3. ordinea stocastică
        T = theory(m)
        diff = T["SP"]["R"][:-1] - T["PS"]["R"][:-1]
        res["ordine"] = dict(R_U_minus_R_V=diff.tolist(),
                             U_st_mai_mare=bool((diff >= -1e-15).all()), V_st_mai_mare=bool((diff <= 1e-15).all()))
        out[m] = res
        print(f"Modelul {m}: control {{{', '.join(f'{k}: p={v['p_mann_whitney']:.1e}' for k, v in res['control'].items())}}}")
        print(f"   locul 1 (aritmetic): {res['ponderi']['aritmetica']}")
        print(f"   ordine: U>=V pe tot suportul: {res['ordine']['U_st_mai_mare']}, V>=U: {res['ordine']['V_st_mai_mare']}")
    json.dump(out, open(RESULTS_DIR / "sensibilitate.json", "w", encoding="utf-8"), indent=1)
    print(f"Salvat: {RESULTS_DIR / 'sensibilitate.json'}")


if __name__ == "__main__":
    main()
