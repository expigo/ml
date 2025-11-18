# Multiple Testing - Interview Preparation

## The Problem

**Multiple Testing Problem**: Running many hypothesis tests increases chance of false positives.

**Example**:
- Test 20 hypotheses at α = 0.05
- Expected false positives = 20 × 0.05 = 1
- Even if all H₀ are true!

---

## Key Concepts

### Family-Wise Error Rate (FWER)
Probability of making at least one Type I error across all tests.

```
FWER = P(at least one false positive)

Without correction: FWER = 1 - (1-α)^m
For m=20, α=0.05: FWER ≈ 0.64 (64%!)
```

### False Discovery Rate (FDR)
Expected proportion of false positives among rejected hypotheses.

```
FDR = E[False Positives / All Rejections]

Less conservative than FWER
```

---

## Correction Methods

### Bonferroni Correction (FWER)

**Most conservative, simple**

```
α_corrected = α / m

where m = number of tests
```

**Decision**: Reject H₀ if p < α/m

**Example**: 10 tests, α=0.05 → reject if p < 0.005

**Pros**: Simple, controls FWER
**Cons**: Very conservative (low power), assumes independence

**When to use**: Few tests, want strong control

### Holm-Bonferroni (FWER)

**Less conservative than Bonferroni**

**Procedure**:
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. For i = 1 to m:
   - If pᵢ > α/(m-i+1), stop (don't reject this or later)
   - Else reject H₀ᵢ and continue

**Pros**: More powerful than Bonferroni, controls FWER
**Cons**: Still conservative

### Benjamini-Hochberg (FDR)

**Controls FDR, less conservative**

**Procedure**:
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest i where: pᵢ ≤ (i/m)α
3. Reject H₀₁, H₀₂, ..., H₀ᵢ

**Example**: m=10, α=0.05
- p₁ ≤ 0.005? → reject
- p₂ ≤ 0.010? → reject
- p₃ ≤ 0.015? → reject
- ...

**Pros**: More powerful than FWER methods
**Cons**: Allows some false positives (by design)

**When to use**: Exploratory analysis, many tests, can tolerate some false positives

### Permutation Tests

**Non-parametric, exact p-values**

**Procedure**:
1. Calculate test statistic on real data
2. Randomly permute labels many times (e.g., 10,000)
3. Calculate test statistic for each permutation
4. P-value = proportion of permutations ≥ observed statistic

**Pros**:
- Exact p-values (no assumptions)
- Can use any test statistic
- Accounts for dependency structure

**Cons**: Computationally expensive

---

## Comparison

| Method | Controls | Power | Assumptions | Use Case |
|--------|----------|-------|-------------|----------|
| **Bonferroni** | FWER | Low | Independence | Few tests, strong control |
| **Holm-Bonferroni** | FWER | Medium | None | More powerful than Bonferroni |
| **Benjamini-Hochberg** | FDR | High | Independence/weak dependency | Many tests, exploratory |
| **Permutation** | FWER | Medium-High | None | Complex dependencies |

---

## Interview Q&A

**Q: Why is multiple testing a problem?**
A: Each test has α probability of false positive. With m tests, chance of at least one false positive grows rapidly:
- 1 test: 5% chance
- 20 tests: 64% chance
Need to control family-wise error rate or FDR.

**Q: Bonferroni vs Benjamini-Hochberg?**
A:
- **Bonferroni**: Controls FWER (no false positives), very conservative, low power
- **BH**: Controls FDR (some false positives OK), less conservative, higher power
- Use Bonferroni when false positives very costly, BH for exploration

**Q: What is FDR?**
A: False Discovery Rate = expected proportion of false positives among all rejections.
- Example: FDR = 0.05 means ~5% of discoveries are false
- Less stringent than FWER (which allows NO false positives)

**Q: When is multiple testing correction not needed?**
A:
- Single primary hypothesis (other tests secondary)
- Exploratory analysis (explicitly stated)
- Tests on independent datasets
- But: Be transparent about number of tests run!

**Q: Bonferroni too conservative?**
A: Yes, especially with many tests:
- m=100, Bonferroni requires p < 0.0005
- Miss many real effects (low power)
- Alternatives: Holm-Bonferroni, FDR methods

**Q: How do permutation tests work?**
A:
1. Calculate observed test statistic
2. Shuffle labels randomly (null: no relationship)
3. Recalculate statistic many times
4. P-value = proportion of shuffles ≥ observed
5. Directly estimates null distribution without assumptions

**Q: Multiple testing in ML feature selection?**
A: Problem: Testing many features for association with target
- Naive approach: Test each, take p<0.05 → many false positives
- Better: Use FDR correction (BH)
- Best: Use dedicated methods (Lasso, stability selection)

**Q: A/B testing and multiple testing?**
A: Common issue: Testing multiple metrics or peeking at results repeatedly
- **Solution 1**: Pre-specify primary metric
- **Solution 2**: Bonferroni correction for multiple metrics
- **Solution 3**: Sequential testing with adjusted boundaries

---

## Code Examples

### Bonferroni
```python
from statsmodels.stats.multitest import multipletests

# p-values from multiple tests
p_values = [0.01, 0.04, 0.03, 0.50, 0.002]

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

print("Bonferroni corrected p-values:", p_corrected)
print("Reject:", reject)
```

### Benjamini-Hochberg (FDR)
```python
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("BH corrected p-values:", p_corrected)
print("Reject:", reject)
```

### Manual Bonferroni
```python
alpha = 0.05
m = len(p_values)
alpha_corrected = alpha / m

reject = [p < alpha_corrected for p in p_values]
```

### Permutation Test
```python
import numpy as np
from scipy import stats

def permutation_test(group1, group2, n_permutations=10000):
    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        perm_diffs.append(perm_diff)

    # P-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value, perm_diffs

# Usage
diff, p, perm_dist = permutation_test(group1, group2)
print(f"Observed difference: {diff:.3f}")
print(f"P-value: {p:.4f}")
```

---

## Decision Guide

```
How many tests?
  |
  ├─ Few (<10)
  │   └─ Use Bonferroni or Holm-Bonferroni
  |
  ├─ Many (>10)
  │   ├─ Need strong control (no false positives)?
  │   │   └─ Yes → Holm-Bonferroni
  │   └─ Exploratory (some false positives OK)?
  │       └─ Yes → Benjamini-Hochberg (FDR)
  |
  ├─ Complex dependencies or non-standard test?
  │   └─ Use Permutation test
  |
  └─ Single primary hypothesis with secondary tests?
      └─ No correction needed (but be transparent)
```

---

## Common Scenarios

**Scenario 1: Feature Selection**
- Testing 100 features for association with target
- **Solution**: Use BH procedure (FDR), or better yet, cross-validated Lasso

**Scenario 2: Post-hoc ANOVA**
- Significant ANOVA, now testing all pairwise comparisons
- **Solution**: Tukey HSD (built-in correction) or Bonferroni

**Scenario 3: A/B Testing Multiple Metrics**
- Testing effect on 5 metrics (clicks, time on site, etc.)
- **Solution**: Pre-specify primary metric, or Bonferroni for all 5

**Scenario 4: Genomics (millions of tests)**
- Testing association of 1M SNPs with disease
- **Solution**: BH with very low FDR (e.g., 0.01), or permutation-based methods

---

## Common Pitfalls

1. ❌ Ignoring multiple testing completely
2. ❌ Using Bonferroni for hundreds of tests (too conservative)
3. ❌ Not pre-specifying analysis plan (p-hacking)
4. ❌ Peeking at results and testing more (inflates α)
5. ❌ Applying correction when not needed (single primary hypothesis)
6. ❌ Not reporting number of tests conducted

---

## Best Practices

1. ✅ Pre-specify hypotheses and analysis plan
2. ✅ Choose correction method before seeing data
3. ✅ Report number of tests conducted
4. ✅ Use FDR methods for exploratory analyses
5. ✅ Consider effect sizes, not just p-values
6. ✅ Be transparent about all tests run (even non-significant)

---

## One-Minute Summary

"Multiple testing inflates Type I error rate. With m tests at α=0.05, FWER ≈ 1-(1-0.05)^m grows rapidly. Control this with corrections. **Bonferroni** (α/m) is most conservative, controls FWER, simple but low power. **Holm-Bonferroni** improves power while controlling FWER. **Benjamini-Hochberg** controls FDR instead of FWER, more powerful, good for exploration. **Permutation tests** make no assumptions, computationally expensive. Use Bonferroni for few tests and strong control, BH for many tests and exploratory work. Always pre-specify analyses and report all tests conducted. Remember: statistical significance with correction is more convincing but don't ignore effect sizes."
