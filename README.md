# ztp-reliability-prng

Rețele serial-paralel (S-P) și paralel-serial (P-S) cu durate de viață ale unităților uniforme pe {0, …, 9}:

| Model | Subsisteme M | Unități per subsistem N |
|---|---|---|
| I   | M = 3        | N ~ ZTP(2,5) |
| II  | M ~ ZTP(2,5) | N = 3 |
| III | M ~ ZTP(2,5) | N ~ ZTP(2,5) |

Sursele de cifre comparate: Java ThreadLocalRandom, Python secrets și cifrele zecimale ale lui π.
Ca control pozitiv se simulează și două generatoare cu defecte cunoscute: RANDU și un LCG din care se folosește cifra x mod 10.

## Structura

```
data/                      cifrele celor trei surse (6 fișiere CSV / sursă)
ztp/                       codul comun
  config.py                parametri, căi, surse, ponderile CRQI
  models.py                distribuțiile teoretice (CDF, PMF, fiabilitate, hazard, momente), N_min
  simulation.py            Knuth + ZTP prin respingere, generarea replicilor (U, V)
  criteria.py              criteriile D1–D10 și indicele CRQI
  controls.py              generatoarele de control (RANDU, LCG mod 10)
01_valori_teoretice.py     Tabelele 2 și 7, N_min           -> results/
02_simulare_crqi.py        simularea și CRQI (Tabelele 3–6, 8–10) -> results/
03_figuri.py               Figurile 1–6, din results/ (fără resimulare) -> figure/
04_anova.py                ANOVA / Kruskal-Wallis între surse, cu mărimile efectului -> results/
05_sensibilitate.py        controale, sensibilitatea ponderilor CRQI, ordinea stocastică -> results/
06_referinte_doua_niveluri.py  cazul clasic M = N = 3, generatorul de referință PCG64,
                           testul pe două niveluri pentru D7/D9 (Tabelele 8 și 14) -> results/
results/                   toate rezultatele numerice (JSON + CSV)
figure/                    figurile articolului (numerotarea din articol)
caim2026/                  figurile pentru prezentarea CAIM 2026
```

## Reproducere

```bash
pip install -r requirements.txt
python 01_valori_teoretice.py
python 02_simulare_crqi.py        # ~1 minut; sau: python 02_simulare_crqi.py II
python 03_figuri.py
python 04_anova.py
python 05_sensibilitate.py
python 06_referinte_doua_niveluri.py   # ~1,5 minute
```
