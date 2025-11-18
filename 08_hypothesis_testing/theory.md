# Hypothesis Testing - Theoretical Foundation

## Introduction

Hypothesis testing is a statistical method to make inferences about population parameters based on sample data.

**Core Question**: Is the observed effect real, or could it have occurred by chance?

---

## Fundamental Concepts

### Hypotheses

**Null Hypothesis (H₀)**: No effect, no difference, status quo
**Alternative Hypothesis (H₁ or Hₐ)**: There is an effect, difference exists

**Examples**:
- H₀: μ = 0 (mean is zero)
- H₁: μ ≠ 0 (mean is not zero)

### Types of Tests

**Two-tailed**: H₁: μ ≠ μ₀ (different from)
**Right-tailed**: H₁: μ > μ₀ (greater than)
**Left-tailed**: H₁: μ < μ₀ (less than)

### P-value

**Definition**: Probability of observing data as extreme as (or more extreme than) what we observed, assuming H₀ is true.

**Interpretation**:
- Small p-value (< α): Strong evidence against H₀ → reject H₀
- Large p-value (≥ α): Weak evidence against H₀ → fail to reject H₀

**Common misconceptions**:
- ❌ P-value is NOT probability H₀ is true
- ❌ P-value is NOT effect size
- ✅ P-value is conditional: P(data | H₀)

### Significance Level (α)

**Definition**: Threshold for decision making. Typically α = 0.05 (5%)

**Interpretation**:
- If p < α: Reject H₀ (statistically significant)
- If p ≥ α: Fail to reject H₀ (not statistically significant)

### Errors

**Type I Error (False Positive)**:
- Reject H₀ when H₀ is true
- Probability = α (significance level)
- Example: Concluding drug works when it doesn't

**Type II Error (False Negative)**:
- Fail to reject H₀ when H₀ is false
- Probability = β
- Example: Concluding drug doesn't work when it does

**Power**:
```
Power = 1 - β
```
Probability of correctly rejecting false H₀. Typically want power ≥ 0.8

### Confidence Intervals

Alternative to hypothesis testing:
```
CI = estimate ± (critical value) × SE
```

**95% CI interpretation**: If we repeat experiment many times, 95% of CIs would contain true parameter.

**Relationship to hypothesis testing**:
- If 95% CI doesn't include null value → reject H₀ at α = 0.05
- If 95% CI includes null value → fail to reject H₀

---

## One-Sample t-test

**Purpose**: Test if population mean equals specific value

**Assumptions**:
1. Random sample
2. Normality (or large n by CLT)
3. Independent observations

**Hypotheses**:
```
H₀: μ = μ₀
H₁: μ ≠ μ₀  (or μ > μ₀, or μ < μ₀)
```

**Test Statistic**:
```
t = (x̄ - μ₀) / (s / √n)

where:
x̄ = sample mean
μ₀ = hypothesized mean
s = sample std
n = sample size
```

**Distribution**: t-distribution with (n-1) degrees of freedom

**Decision**: Reject H₀ if |t| > t_{α/2, n-1} (two-tailed)

---

## Two-Sample t-test

**Purpose**: Compare means of two independent groups

**Assumptions**:
1. Independence between and within groups
2. Normality in both groups
3. Equal variances (for standard t-test)

### Equal Variances (Pooled t-test)

**Hypotheses**:
```
H₀: μ₁ = μ₂
H₁: μ₁ ≠ μ₂
```

**Pooled Variance**:
```
s²_p = ((n₁-1)s²₁ + (n₂-1)s²₂) / (n₁ + n₂ - 2)
```

**Test Statistic**:
```
t = (x̄₁ - x̄₂) / (s_p √(1/n₁ + 1/n₂))
```

**DF**: n₁ + n₂ - 2

### Unequal Variances (Welch's t-test)

**Test Statistic**:
```
t = (x̄₁ - x̄₂) / √(s²₁/n₁ + s²₂/n₂)
```

**DF**: Satterthwaite approximation (fractional)

**When to use**: Always safer to use Welch's (robust to unequal variances)

---

## Paired t-test

**Purpose**: Compare two related measurements (before/after, matched pairs)

**Key**: Each subject serves as own control

**Hypotheses**:
```
H₀: μ_diff = 0
H₁: μ_diff ≠ 0
```

**Test Statistic**:
```
t = d̄ / (s_d / √n)

where:
d̄ = mean of differences
s_d = std of differences
n = number of pairs
```

**DF**: n - 1

**Example**: Test scores before and after tutoring for same students

---

## Chi-Square Tests

### Chi-Square Goodness of Fit

**Purpose**: Test if observed frequencies match expected distribution

**Hypotheses**:
```
H₀: Data follows specified distribution
H₁: Data doesn't follow specified distribution
```

**Test Statistic**:
```
χ² = Σ (Observed - Expected)² / Expected
```

**DF**: k - 1 - p
- k = number of categories
- p = number of estimated parameters

**Example**: Testing if die is fair (expected: 1/6 for each face)

### Chi-Square Test of Independence

**Purpose**: Test if two categorical variables are independent

**Hypotheses**:
```
H₀: Variables are independent
H₁: Variables are associated
```

**Test Statistic**:
```
χ² = Σ (O_ij - E_ij)² / E_ij

where:
O_ij = observed count in cell (i,j)
E_ij = (row total × column total) / grand total
```

**DF**: (rows - 1) × (columns - 1)

**Example**: Is there association between gender and voting preference?

**Requirements**:
- Expected count ≥ 5 in at least 80% of cells
- No expected count < 1

---

## ANOVA (Analysis of Variance)

**Purpose**: Compare means across 3+ groups simultaneously

**Why not multiple t-tests?**
- Multiple comparisons inflate Type I error rate
- k groups → k(k-1)/2 pairwise tests
- Family-wise error rate grows rapidly

### One-Way ANOVA

**Hypotheses**:
```
H₀: μ₁ = μ₂ = ... = μ_k
H₁: At least one mean differs
```

**Assumptions**:
1. Independence
2. Normality in each group
3. Equal variances (homoscedasticity)

**Test Statistic**:
```
F = MSB / MSW = (Between-group variance) / (Within-group variance)

MSB = SSB / (k-1)
MSW = SSW / (n-k)

where:
SSB = Σnᵢ(ȳᵢ - ȳ)²  (between)
SSW = Σ Σ(yᵢⱼ - ȳᵢ)²  (within)
```

**Distribution**: F-distribution with DF1 = k-1, DF2 = n-k

**Decision**: Reject H₀ if F > F_{α, k-1, n-k}

**Post-hoc tests** (if ANOVA significant):
- Tukey HSD: All pairwise comparisons
- Bonferroni: Conservative, controls family-wise error
- Scheffé: Most conservative

### Two-Way ANOVA

**Purpose**: Test effects of two factors and their interaction

**Hypotheses**:
```
H₀ᴬ: No effect of factor A
H₀ᴮ: No effect of factor B
H₀ᴬᴮ: No interaction between A and B
```

**Three F-tests**: One for each hypothesis

**Interaction**: Effect of one factor depends on level of other factor

---

## Non-Parametric Tests

When assumptions (especially normality) are violated.

### Mann-Whitney U Test (Wilcoxon Rank-Sum)

**Purpose**: Non-parametric alternative to two-sample t-test

**Hypotheses**:
```
H₀: Distributions are equal
H₁: Distributions differ
```

**Method**:
1. Rank all observations together
2. Sum ranks for each group
3. Calculate U statistic

**Use when**: Non-normal data, small samples, ordinal data

### Wilcoxon Signed-Rank Test

**Purpose**: Non-parametric alternative to paired t-test

**Method**:
1. Calculate differences
2. Rank absolute differences (ignoring sign)
3. Sum ranks for positive and negative differences

### Kruskal-Wallis Test

**Purpose**: Non-parametric alternative to one-way ANOVA

**Method**: Rank-based extension of Mann-Whitney to 3+ groups

---

## Effect Size

**Why**: P-value doesn't tell us magnitude of effect!

**Common Measures**:

**Cohen's d** (for t-tests):
```
d = (μ₁ - μ₂) / pooled_std

Small: 0.2, Medium: 0.5, Large: 0.8
```

**Eta-squared (η²)** (for ANOVA):
```
η² = SSB / SST

Proportion of variance explained by group differences
```

**Cramér's V** (for Chi-square):
```
V = √(χ² / (n × min(rows-1, cols-1)))

Range: [0, 1]
```

---

## Statistical Power

**Definition**: Probability of detecting effect when it exists

**Factors affecting power**:
1. **Effect size**: Larger effect → higher power
2. **Sample size**: Larger n → higher power
3. **Significance level (α)**: Larger α → higher power (but more Type I errors)
4. **Variance**: Lower variance → higher power

**Power Analysis**:
- **A priori**: Determine required n for desired power
- **Post hoc**: Calculate achieved power with given n

**Rule of thumb**: Aim for power ≥ 0.80

---

## Common Misconceptions

1. **"Fail to reject H₀" ≠ "Accept H₀"**
   - Absence of evidence ≠ Evidence of absence

2. **P-value ≠ Probability H₀ is true**
   - P-value = P(data | H₀), not P(H₀ | data)

3. **Statistical significance ≠ Practical significance**
   - Small p-value doesn't mean large effect
   - With large n, tiny effects can be "significant"

4. **P = 0.049 vs P = 0.051**
   - Arbitrary threshold! Results are practically identical

5. **Multiple testing problem**
   - More tests → higher chance of Type I error
   - Need multiple testing corrections

---

## Decision Framework

```
1. State hypotheses (H₀, H₁)
2. Choose significance level (α)
3. Check assumptions
4. Select appropriate test
5. Calculate test statistic
6. Find p-value
7. Make decision (reject or fail to reject H₀)
8. Report effect size and confidence interval
9. Interpret in context
```

---

## Summary Table

| Test | Purpose | Data Type | Assumptions |
|------|---------|-----------|-------------|
| **One-sample t-test** | μ = μ₀ | Continuous | Normality |
| **Two-sample t-test** | μ₁ = μ₂ | Continuous | Normality, independence |
| **Paired t-test** | Before = After | Continuous | Normality of differences |
| **Chi-square GoF** | Fits distribution | Categorical | Expected ≥ 5 |
| **Chi-square Independence** | Association | Categorical | Expected ≥ 5 |
| **One-way ANOVA** | μ₁=μ₂=...=μₖ | Continuous | Normality, equal variance |
| **Mann-Whitney** | Distributions equal | Ordinal/Non-normal | Independence |
| **Wilcoxon Signed-Rank** | Paired differences | Ordinal/Non-normal | Paired data |
| **Kruskal-Wallis** | 3+ groups | Ordinal/Non-normal | Independence |

---

## Practical Recommendations

1. **Always check assumptions** before running tests
2. **Report effect sizes** along with p-values
3. **Use confidence intervals** for better interpretation
4. **Consider practical significance**, not just statistical
5. **Pre-register analyses** to avoid p-hacking
6. **Correct for multiple testing** when appropriate
7. **Use non-parametric tests** when assumptions violated
8. **Report exact p-values** (not just p < 0.05)
