# ztp-reliability-prng

Cod și date pentru articolul *Quality of Pseudorandom Number Generators Approached Through the Lens of
Monte Carlo Simulation of New Probabilistic Models in Network Reliability* (V. Astafi, A. Leahu, UTM).

Rețele serial-paralel (S-P) și paralel-serial (P-S) cu durate de viață ale unităților uniforme pe {0, …, 9}:

| Model | Subsisteme M | Unități per subsistem N |
|---|---|---|
| I   | M = 3        | N ~ ZTP(2,5) |
| II  | M ~ ZTP(2,5) | N = 3 |
| III | M ~ ZTP(2,5) | N ~ ZTP(2,5) |

Sursele de cifre comparate: Java ThreadLocalRandom, Python secrets și cifrele zecimale ale lui π.

## Structura

```
data/                      cifrele celor trei surse (6 fișiere CSV / sursă)
ztp/                       codul comun
  config.py                parametri, căi, surse, ponderile CRQI
  models.py                distribuțiile teoretice (CDF, PMF, fiabilitate, hazard, momente), N_min
  simulation.py            Knuth + ZTP prin respingere, generarea replicilor (U, V)
  criteria.py              criteriile D1–D10 și indicele CRQI
01_valori_teoretice.py     Tabelele 2 și 7, N_min           -> results/
02_simulare_crqi.py        simularea și CRQI (Tabelele 3–6, 8–10) -> results/
03_figuri.py               Figurile 1–6, din results/ (fără resimulare) -> figure/
04_anova.py                ANOVA / Kruskal-Wallis între surse -> results/
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
```
