import json
import numpy as np
from scipy import stats

with open("crqi_repeated_trials.json") as f:
    data = json.load(f)

groups = {
    "Java ThreadLocalRandom": data["Java_ThreadLocalRandom"]["crqi_weighted_trials_overall"],
    "Python secrets": data["Python_secrets"]["crqi_weighted_trials_overall"],
    "Cifrele π": data["Pi_digits"]["crqi_weighted_trials_overall"],
}

print("=== Descriptive stats per group (n=100 trials each) ===")
for name, vals in groups.items():
    vals = np.array(vals)
    print(f"{name}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, n={len(vals)}")

print("\n=== One-way ANOVA ===")
f_stat, p_value = stats.f_oneway(*groups.values())
print(f"F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

# Effect size: eta-squared
all_vals = np.concatenate(list(groups.values()))
grand_mean = all_vals.mean()
ss_between = sum(len(v) * (np.mean(v) - grand_mean)**2 for v in groups.values())
ss_total = sum((all_vals - grand_mean)**2)
eta_squared = ss_between / ss_total
print(f"Eta-squared (effect size) = {eta_squared:.5f}")

print("\n=== Post-hoc pairwise t-tests (Welch, unequal variance) ===")
names = list(groups.keys())
pvals = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        t_stat, p = stats.ttest_ind(groups[names[i]], groups[names[j]], equal_var=False)
        pvals.append((names[i], names[j], t_stat, p))
        print(f"{names[i]} vs {names[j]}: t={t_stat:.4f}, p={p:.4f}")

# Bonferroni correction for 3 pairwise comparisons
print("\n=== Bonferroni-corrected significance (alpha=0.05, 3 comparisons -> alpha_adj=0.0167) ===")
alpha_adj = 0.05 / 3
for name1, name2, t_stat, p in pvals:
    sig = "SIGNIFICANT" if p < alpha_adj else "not significant"
    print(f"{name1} vs {name2}: p={p:.4f} -> {sig}")

with open("anova_results.json", "w") as f:
    json.dump({
        "f_stat": f_stat, "p_value": p_value, "eta_squared": eta_squared,
        "pairwise": [{"pair": f"{a} vs {b}", "t": t, "p": p} for a, b, t, p in pvals]
    }, f, indent=2)
print("\nSaved anova_results.json")
