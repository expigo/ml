# Evaluation Metrics - Interview Preparation

## Why Metrics Matter

**Choosing the right metric is critical** - it defines what "good" means for your model!

---

## Classification Metrics

### Confusion Matrix

|       | Pred + | Pred - |
|-------|--------|--------|
| **Act +** | TP     | FN     |
| **Act -** | FP     | TN     |

- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interview Q&A**

**Q: When to use accuracy?**
A: Balanced datasets where all classes equally important.

**Q: When NOT to use accuracy?**
A: **Imbalanced datasets!**
- Example: 95% class 0, 5% class 1
- Always predicting class 0 → 95% accuracy but useless model

**Q: Accuracy paradox?**
A: Model with higher accuracy can have worse performance on minority class. Always check per-class metrics for imbalanced data.

### Precision

```
Precision = TP / (TP + FP)
```

**Interpretation**: "Of all predicted positives, how many are actually positive?"

**When to optimize**: Minimize false positives
- **Spam detection**: Don't want to mark important emails as spam
- **Medical diagnosis**: Don't want to tell healthy person they're sick (expensive follow-up tests)

### Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Interpretation**: "Of all actual positives, how many did we catch?"

**When to optimize**: Minimize false negatives
- **Disease detection**: Don't want to miss sick patients
- **Fraud detection**: Don't want to miss fraudulent transactions
- **Airport security**: Don't want to miss threats

### Precision-Recall Tradeoff

**Fundamental tradeoff**: Increasing one often decreases the other

**Example**: Lowering classification threshold
- More predictions as positive → Higher recall, lower precision

**Q: Which to prioritize?**
A: Depends on cost of errors:
- **False positives expensive**: Prioritize precision
- **False negatives expensive**: Prioritize recall
- **Both important**: Use F1 score

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Harmonic mean** of precision and recall

**Properties**:
- Range: [0, 1], higher is better
- Only high if both precision and recall are high
- Balanced measure

**Q: F1 vs Accuracy?**
A:
- **F1**: Better for imbalanced data, focuses on positive class
- **Accuracy**: Good for balanced data, treats all classes equally

**Q: When to use F1?**
A:
- Imbalanced datasets
- Need balance between precision and recall
- Binary classification
- Want single metric

### F-Beta Score

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

**β controls precision/recall weight**:
- **β < 1**: Emphasizes precision
- **β > 1**: Emphasizes recall
- **β = 1**: F1 score (balanced)
- **β = 2**: F2 score (2× weight on recall)
- **β = 0.5**: F0.5 score (2× weight on precision)

### Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)
```

**Interpretation**: "Of all actual negatives, how many did we correctly identify?"

**Use**: Medical tests, where correctly identifying healthy is important

### ROC Curve & AUC

**ROC (Receiver Operating Characteristic)**: Plot of TPR vs FPR at different thresholds

```
TPR (Recall) = TP / (TP + FN)
FPR = FP / (FP + TN)
```

**AUC (Area Under Curve)**:
- Range: [0, 1]
- 0.5: Random classifier (diagonal line)
- 1.0: Perfect classifier
- >0.8: Generally good
- <0.6: Poor

**Interview Q&A**

**Q: What is ROC-AUC?**
A: ROC curve plots true positive rate vs false positive rate at various classification thresholds. AUC measures area under this curve - represents probability that model ranks random positive higher than random negative.

**Q: When to use AUC?**
A:
- Compare models (threshold-independent)
- Imbalanced datasets
- Care about ranking quality
- Need probabilistic predictions

**Q: ROC-AUC vs Accuracy?**
A:
- **AUC**: Threshold-independent, good for imbalanced data
- **Accuracy**: Threshold-dependent (usually 0.5), misleading for imbalanced data

**Q: Limitations of AUC?**
A:
- Doesn't tell you anything about optimal threshold
- Can be optimistic for very imbalanced data
- Doesn't distinguish types of errors

### Precision-Recall Curve & Average Precision

**PR Curve**: Plot of Precision vs Recall at different thresholds

**When to use PR curve vs ROC**:
- **Use PR curve**: Highly imbalanced datasets, care more about positive class
- **Use ROC curve**: Balanced datasets, care about both classes

**Average Precision (AP)**: Area under PR curve

**Q: Why PR curve for imbalanced data?**
A: ROC can be overly optimistic when negative class dominates. PR curve focuses on positive class performance.

### Multi-Class Metrics

**Macro Average**: Average of per-class metrics (treats all classes equally)
```
Macro = (Metric_class1 + Metric_class2 + ... + Metric_classK) / K
```

**Micro Average**: Calculate from total TP, FP, FN across all classes
```
Micro Precision = Σ TP_i / Σ(TP_i + FP_i)
```

**Weighted Average**: Average of per-class metrics weighted by support (# samples)

**Q: Macro vs Micro vs Weighted?**
A:
- **Macro**: All classes equal weight (good for balanced classes)
- **Micro**: All samples equal weight (dominated by frequent classes)
- **Weighted**: Classes weighted by frequency (accounts for imbalance)

---

## Regression Metrics

### Mean Squared Error (MSE)

```
MSE = (1/n) Σ(y_i - ŷ_i)²
```

**Properties**:
- Range: [0, ∞), lower is better
- Units: target variable squared
- Penalizes large errors heavily (quadratic)
- Differentiable (good for optimization)

**Q: When to use MSE?**
A: Want to penalize large errors more, outliers are important, need differentiable loss.

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE
```

**Properties**:
- Same units as target variable (interpretable!)
- Still penalizes large errors
- Most commonly reported

**Q: RMSE vs MSE?**
A: RMSE has same units as target (easier to interpret), otherwise equivalent. Always use RMSE for reporting.

### Mean Absolute Error (MAE)

```
MAE = (1/n) Σ|y_i - ŷ_i|
```

**Properties**:
- Same units as target
- Linear penalty (all errors weighted equally)
- Robust to outliers
- Not differentiable at zero

**Q: MAE vs RMSE?**
A:
| Aspect | MAE | RMSE |
|--------|-----|------|
| Outlier sensitivity | Low | High |
| Interpretation | Average error | Emphasizes large errors |
| Units | Target | Target |
| When to use | Robust metric needed | Outliers important |

**Example**:
- Errors: [1, 1, 1, 10]
- MAE = 3.25, RMSE = 5.12
- RMSE penalizes the 10 more heavily

### R² (Coefficient of Determination)

```
R² = 1 - (SSE / SST)
   = 1 - (Σ(y_i - ŷ_i)²) / (Σ(y_i - ȳ)²)
```

Where:
- **SSE**: Sum of Squared Errors (model)
- **SST**: Total Sum of Squares (baseline = mean)

**Interpretation**:
- Proportion of variance explained by model
- R² = 0: Model as good as predicting mean
- R² = 1: Perfect predictions
- R² < 0: Model worse than predicting mean (bad!)

**Q: What does R² = 0.8 mean?**
A: Model explains 80% of variance in target variable. Remaining 20% is unexplained (noise, missing features, etc.).

**Q: Can R² be negative?**
A: Yes! If model performs worse than simply predicting the mean. Indicates very poor model.

**Q: Limitations of R²?**
A:
- Always increases when adding features (even if irrelevant)
- Doesn't indicate if model is appropriate
- Can be high even with biased predictions
- Not comparable across different datasets

### Adjusted R²

```
R²_adj = 1 - (1 - R²)(n - 1)/(n - p - 1)
```

Where:
- n: number of observations
- p: number of predictors

**Purpose**: Penalizes adding irrelevant features

**Q: Adjusted R² vs R²?**
A: Adjusted R² only increases if new feature improves model more than expected by chance. Use for comparing models with different numbers of features.

### Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) Σ|(y_i - ŷ_i) / y_i|
```

**Properties**:
- Percentage (scale-independent)
- Interpretable: "average % error"
- Undefined when y_i = 0!

**Q: When to use MAPE?**
A: When you need scale-independent metric, care about relative errors. Don't use if target can be zero or close to zero.

---

## Clustering Metrics

### Silhouette Score

```
s_i = (b_i - a_i) / max(a_i, b_i)

where:
a_i = avg distance to points in same cluster
b_i = avg distance to points in nearest cluster
```

**Range**: [-1, 1]
- +1: Perfect (far from other clusters)
- 0: On cluster boundary
- -1: Probably in wrong cluster

**Average across all points**: Overall clustering quality

**Q: How to interpret?**
A:
- >0.5: Good clustering
- 0.2-0.5: Weak structure
- <0.2: No meaningful clusters

### Davies-Bouldin Index

```
DB = (1/K) Σ max_{j≠i} ((σ_i + σ_j) / d(c_i, c_j))
```

**Lower is better**

**Interpretation**: Average similarity between each cluster and its most similar cluster. Low value = well-separated clusters.

### Calinski-Harabasz Index (Variance Ratio)

```
CH = (Between-cluster variance / Within-cluster variance) × ((n-K)/(K-1))
```

**Higher is better**

**Interpretation**: Ratio of between-cluster to within-cluster variance. Higher = denser, better separated clusters.

---

## Metric Selection Guide

### Classification

```
Balanced data, all classes important?
  → Accuracy

Imbalanced data?
  → F1, Precision, Recall, ROC-AUC

False positives expensive (spam, medical)?
  → Precision

False negatives expensive (disease, fraud)?
  → Recall

Need balance?
  → F1 Score

Need probability ranking?
  → ROC-AUC

Multi-class balanced?
  → Macro-averaged F1

Multi-class imbalanced?
  → Weighted F1
```

### Regression

```
Want to penalize large errors?
  → RMSE

Want robust to outliers?
  → MAE

Want interpretable percentage?
  → MAPE (if no zeros)

Want % variance explained?
  → R²

Comparing models with different # features?
  → Adjusted R²
```

---

## Common Interview Questions

**Q: Why not always use accuracy?**
A: Misleading for imbalanced data. 95% accuracy sounds good but could mean always predicting majority class.

**Q: Precision or Recall for spam filter?**
A: **Precision**. Better to let some spam through (false negative) than block important email (false positive).

**Q: Precision or Recall for cancer detection?**
A: **Recall**. Better to have false alarms (false positive) than miss cancer (false negative).

**Q: F1 score vs ROC-AUC?**
A:
- **F1**: Single threshold, balances precision/recall
- **ROC-AUC**: All thresholds, probability ranking quality
- Use F1 for deployment (fixed threshold), AUC for model comparison

**Q: When is high accuracy but low F1 possible?**
A: Imbalanced data. Model predicts majority class → high accuracy, but misses minority class → low precision/recall/F1.

**Q: R² of 0.5 - is this good?**
A: Depends on domain! Social sciences: 0.5 is great. Physics: 0.5 is poor. Always compare to baseline and domain expectations.

**Q: MAE of 10 - is this good?**
A: Cannot tell without context! If predicting house prices ($100k-$1M), great. If predicting temperature (0-100°F), terrible. Always report metric in context of target scale.

---

## Best Practices

1. ✅ Use multiple metrics (no single metric tells full story)
2. ✅ Always check confusion matrix for classification
3. ✅ Use stratified metrics for imbalanced data
4. ✅ Report metrics with confidence intervals
5. ✅ Consider business cost of different errors
6. ✅ Visualize: ROC curves, PR curves, residual plots
7. ✅ Compare to baselines (random, majority class, mean)
8. ✅ Understand metric units and scale

---

## One-Minute Summary

"Choose metrics based on problem and data. **Classification**: Accuracy for balanced data; Precision (minimize false positives), Recall (minimize false negatives), F1 (balance both) for imbalanced data; ROC-AUC for threshold-independent comparison. **Regression**: RMSE (penalize large errors), MAE (robust to outliers), R² (variance explained). **Clustering**: Silhouette ([-1,1], higher better), Davies-Bouldin (lower better). Always use multiple metrics. For imbalanced classification, never rely on accuracy alone - use F1, precision, recall, or AUC. For regression, report both RMSE/MAE and R². Consider business costs when choosing metrics. Visualize confusion matrices, ROC curves, and residual plots."
