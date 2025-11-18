# Hypothesis Testing - Interview Preparation

## Core Concepts

### Hypotheses
- **H₀ (Null)**: No effect, status quo
- **H₁ (Alternative)**: There is an effect

### P-value
```
P-value = P(data as extreme | H₀ is true)

Low p-value → reject H₀
```

### Significance Level (α)
- Threshold for decision (typically 0.05)
- P < α → reject H₀
- P ≥ α → fail to reject H₀

### Errors

| Reality | Decision | Result |
|---------|----------|--------|
| H₀ true | Reject H₀ | **Type I Error** (α) |
| H₀ false | Fail to reject | **Type II Error** (β) |
| H₀ true | Fail to reject | Correct |
| H₀ false | Reject H₀ | Correct (Power = 1-β) |

---

## Test Selection Guide

| Scenario | Test |
|----------|------|
| One group vs value | One-sample t-test |
| Two independent groups | Two-sample t-test |
| Two related groups | Paired t-test |
| 3+ independent groups | One-way ANOVA |
| Categorical association | Chi-square independence |
| Goodness of fit | Chi-square GoF |
| Non-normal, 2 groups | Mann-Whitney U |
| Non-normal, paired | Wilcoxon Signed-Rank |
| Non-normal, 3+ groups | Kruskal-Wallis |

---

## Key Formulas

### One-Sample t-test
```
t = (x̄ - μ₀) / (s/√n)
DF = n - 1
```

### Two-Sample t-test
```
t = (x̄₁ - x̄₂) / (s_p √(1/n₁ + 1/n₂))

s²_p = ((n₁-1)s²₁ + (n₂-1)s²₂) / (n₁ + n₂ - 2)

DF = n₁ + n₂ - 2
```

### Paired t-test
```
t = d̄ / (s_d/√n)
where d̄ = mean of differences
DF = n - 1
```

### Chi-Square
```
χ² = Σ(Observed - Expected)² / Expected
```

### ANOVA F-statistic
```
F = MSB / MSW
MSB = Between-group variance
MSW = Within-group variance
```

---

## Common Interview Questions

**Q: What is a p-value?**
A: Probability of observing data as extreme as (or more than) what we observed, assuming H₀ is true. NOT the probability that H₀ is true!

**Q: What does "statistically significant" mean?**
A: P-value < α (typically 0.05). Strong evidence against H₀. But doesn't necessarily mean practically important!

**Q: Type I vs Type II error?**
A:
- **Type I (α)**: False positive - reject true H₀ (saying there's effect when there isn't)
- **Type II (β)**: False negative - fail to reject false H₀ (missing real effect)
- **Trade-off**: Decreasing one increases the other (for fixed n)

**Q: What is statistical power?**
A: Power = 1 - β = Probability of correctly rejecting false H₀ (detecting real effect). Should be ≥ 0.80.

**Factors increasing power**:
1. Larger sample size
2. Larger effect size
3. Higher α (but more Type I errors)
4. Lower variance

**Q: One-tailed vs two-tailed test?**
A:
- **Two-tailed**: H₁: μ ≠ μ₀ (testing for any difference)
- **One-tailed**: H₁: μ > μ₀ or μ < μ₀ (testing specific direction)
- Use two-tailed unless strong theoretical reason for direction
- One-tailed has more power in specified direction but can't detect opposite

**Q: When to use t-test vs z-test?**
A:
- **z-test**: Population σ known (rare in practice)
- **t-test**: Sample s used to estimate σ (almost always)
- With large n (>30), t and z distributions converge

**Q: Assumptions of t-test?**
A:
1. Random sample
2. Normality (or large n by CLT)
3. Independence
4. (For two-sample: equal variances, or use Welch's)

**Q: What if assumptions violated?**
A:
- **Non-normality**: Use non-parametric test (Mann-Whitney, Wilcoxon)
- **Unequal variances**: Use Welch's t-test
- **Non-independence**: Use specialized methods (time series, mixed models)
- **Outliers**: Robust methods, transformation, or remove if erroneous

**Q: Paired vs two-sample t-test?**
A:
- **Paired**: Related measurements (before/after, matched pairs) - more powerful
- **Two-sample**: Independent groups
- Pairing reduces variance by controlling for individual differences

**Q: Why ANOVA instead of multiple t-tests?**
A: Multiple t-tests inflate Type I error rate.
- 3 groups: 3 t-tests, α_family ≈ 0.14 (not 0.05!)
- ANOVA controls family-wise error rate
- If significant, use post-hoc tests with correction

**Q: What if ANOVA is significant?**
A: ANOVA only tells us "at least one group differs," not which ones. Need post-hoc tests:
- **Tukey HSD**: All pairwise comparisons, balanced
- **Bonferroni**: Conservative, simple
- **Scheffé**: Most conservative, for complex comparisons

**Q: When to use Chi-square vs t-test?**
A:
- **Chi-square**: Categorical data (counts, frequencies)
- **t-test**: Continuous data (measurements)

**Q: Chi-square independence vs goodness of fit?**
A:
- **Independence**: Test association between two categorical variables (contingency table)
- **Goodness of fit**: Test if one variable follows specified distribution

**Q: What is effect size? Why important?**
A: Magnitude of effect (e.g., Cohen's d, η²). Important because:
- P-value depends on sample size
- With large n, tiny effects can be "significant"
- Effect size tells us if result is practically meaningful
- Should always report with p-value

**Q: Cohen's d interpretation?**
A:
- Small: 0.2
- Medium: 0.5
- Large: 0.8

**Q: Confidence interval vs hypothesis test?**
A:
- CI provides range of plausible values
- Hypothesis test gives binary decision
- CI is more informative (shows magnitude and precision)
- If 95% CI excludes null value → reject H₀ at α=0.05

**Q: What is multiple testing problem?**
A: Running many tests increases chance of false positives.
- 20 tests at α=0.05 → expect 1 false positive by chance
- Need corrections: Bonferroni, FDR, permutation tests

**Q: Non-parametric tests: when and why?**
A:
- **When**: Assumptions violated (non-normality, small n, ordinal data)
- **Tests**: Mann-Whitney (2 groups), Wilcoxon (paired), Kruskal-Wallis (3+ groups)
- **Trade-off**: Less powerful than parametric if assumptions met

**Q: How to check normality?**
A:
1. Visual: Q-Q plot, histogram
2. Tests: Shapiro-Wilk (n<50), Kolmogorov-Smirnov
3. Rule: With large n, CLT makes t-test robust to non-normality

**Q: Sample size determination?**
A: Power analysis (a priori):
1. Specify desired power (typically 0.80)
2. Specify α (typically 0.05)
3. Specify effect size (from literature or pilot study)
4. Calculate required n

**Q: What does "fail to reject H₀" mean?**
A: Insufficient evidence to conclude effect exists. NOT the same as "H₀ is true" or "no effect."
- Absence of evidence ≠ Evidence of absence
- Could be due to low power, small sample

---

## Code Snippets

### One-Sample t-test
```python
from scipy import stats

t_stat, p_value = stats.ttest_1samp(data, popmean=0)
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
```

### Two-Sample t-test
```python
# Equal variances
t, p = stats.ttest_ind(group1, group2)

# Unequal variances (Welch's)
t, p = stats.ttest_ind(group1, group2, equal_var=False)
```

### Paired t-test
```python
t, p = stats.ttest_rel(before, after)
```

### Chi-Square Independence
```python
from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(contingency_table)
```

### One-Way ANOVA
```python
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Post-hoc: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(data, groups, alpha=0.05)
print(tukey)
```

### Mann-Whitney U (non-parametric)
```python
u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
```

### Wilcoxon Signed-Rank (non-parametric paired)
```python
w_stat, p_value = stats.wilcoxon(before, after)
```

### Check Normality
```python
# Shapiro-Wilk test
stat, p = stats.shapiro(data)
if p > 0.05:
    print("Data appears normal")

# Q-Q plot
import matplotlib.pyplot as plt
stats.probplot(data, dist="norm", plot=plt)
```

### Effect Size
```python
# Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std
```

---

## Common Pitfalls

1. ❌ Confusing p-value with probability H₀ is true
2. ❌ Not checking assumptions before testing
3. ❌ Using accuracy/statistical significance for practical significance
4. ❌ Running multiple tests without correction
5. ❌ Using one-tailed test without strong justification
6. ❌ Accepting H₀ (should "fail to reject")
7. ❌ Not reporting effect sizes and confidence intervals
8. ❌ P-hacking (trying different tests until one is significant)

---

## Decision Tree

```
What are you comparing?
  |
  ├─ One group vs value
  │   ├─ Normal? → One-sample t-test
  │   └─ Non-normal? → Wilcoxon Signed-Rank
  |
  ├─ Two groups
  │   ├─ Independent?
  │   │   ├─ Normal? → Two-sample t-test (or Welch's)
  │   │   └─ Non-normal? → Mann-Whitney U
  │   └─ Paired?
  │       ├─ Normal? → Paired t-test
  │       └─ Non-normal? → Wilcoxon Signed-Rank
  |
  ├─ 3+ groups
  │   ├─ Normal? → One-way ANOVA → Tukey HSD
  │   └─ Non-normal? → Kruskal-Wallis
  |
  └─ Categorical data
      ├─ One variable → Chi-square GoF
      └─ Two variables → Chi-square Independence
```

---

## One-Minute Summary

"Hypothesis testing evaluates if observed effects are statistically significant. Set up null (H₀: no effect) and alternative (H₁: effect exists) hypotheses. Choose significance level α (typically 0.05). Calculate test statistic (t, F, χ²) and p-value. If p < α, reject H₀. Type I error (α): false positive. Type II error (β): false negative. Power = 1-β (should be ≥0.80). Use t-tests for comparing means (one-sample, two-sample, paired), ANOVA for 3+ groups, Chi-square for categorical data. Non-parametric alternatives when assumptions violated. Always report effect sizes (Cohen's d, η²) not just p-values. Correct for multiple testing. Statistical significance ≠ practical significance."
