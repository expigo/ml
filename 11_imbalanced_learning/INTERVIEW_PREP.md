# Imbalanced Learning - Interview Preparation

## The Problem

**Imbalanced Data**: One class has significantly more samples than others

**Common in**:
- Fraud detection (0.1% fraudulent)
- Disease diagnosis (1-5% diseased)
- Spam detection (10-20% spam)
- Defect detection (< 1% defective)
- Churn prediction (5-20% churn)

**Why it's a problem**:
- Models biased toward majority class
- High accuracy but poor minority class performance
- Difficult to learn minority class patterns

---

## Measuring Imbalance

**Imbalance Ratio (IR)**:
```
IR = # majority class samples / # minority class samples
```

**Classification**:
- **Mild**: IR < 10 (e.g., 90% vs 10%)
- **Moderate**: IR 10-100 (e.g., 99% vs 1%)
- **Severe**: IR > 100 (e.g., 99.9% vs 0.1%)

---

## Strategies

### 1. Data-Level Approaches (Resampling)

#### Undersampling (Remove Majority)

**Random Undersampling**: Randomly remove majority class samples

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Pros**:
- Fast
- Reduces training time
- Balances classes

**Cons**:
- **Loss of information** (throws away data!)
- May remove important samples
- Poor for small datasets

**When to use**: Large datasets where majority class is huge

#### Oversampling (Add Minority)

**Random Oversampling**: Randomly duplicate minority class samples

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**Pros**:
- No information loss
- Simple

**Cons**:
- **Overfitting risk** (exact duplicates!)
- Increases training time
- Doesn't add new information

#### SMOTE (Synthetic Minority Over-sampling Technique)

**Best oversampling method!**

**How it works**:
1. For each minority sample
2. Find K nearest neighbors (default K=5)
3. Create synthetic sample along line to random neighbor
4. Repeat until balanced

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Formula**:
```
x_new = x_i + λ × (x_neighbor - x_i)
where λ ∈ [0, 1] randomly chosen
```

**Interview Q&A**

**Q: What is SMOTE?**
A: Creates synthetic minority samples by interpolating between existing minority samples and their nearest neighbors. Better than random oversampling because it creates new, diverse samples rather than exact duplicates.

**Q: SMOTE advantages?**
A:
- No overfitting (unlike random oversampling)
- Creates new information (unlike undersampling)
- Most popular approach

**Q: SMOTE limitations?**
A:
- Can create noisy samples if minority class overlaps with majority
- Doesn't work well with high-dimensional categorical data
- Can amplify outliers
- Assumes continuous features

**Q: Variants of SMOTE?**
A:
- **BorderlineSMOTE**: Only samples near decision boundary
- **ADASYN**: More synthetic samples for harder-to-learn instances
- **SMOTE-ENN**: SMOTE + remove overlapping samples
- **SMOTE-Tomek**: SMOTE + remove Tomek links

#### Combination: SMOTE + Undersampling

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

**Idea**: Oversample minority (SMOTE), then clean up overlapping samples

### 2. Algorithm-Level Approaches

#### Class Weights

**Penalize misclassifications** of minority class more

```python
from sklearn.ensemble import RandomForestClassifier

# Automatically balance based on class frequencies
model = RandomForestClassifier(class_weight='balanced')

# Or manually
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
```

**How it works**:
```
weight_class_i = n_samples / (n_classes × n_samples_class_i)
```

**Interview Q&A**

**Q: What does class_weight='balanced' do?**
A: Automatically adjusts weights inversely proportional to class frequencies. Minority class errors penalized more during training, forcing model to pay attention to minority class.

**Q: Class weights vs SMOTE?**
A:
- **Class weights**: No data modification, works with all algorithms that support it
- **SMOTE**: Creates new data, works with any algorithm
- Can use both together!

#### Threshold Adjustment

**Don't use default 0.5 threshold!**

```python
# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find optimal threshold (e.g., where F1 is max)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Predict with optimal threshold
y_pred = (y_proba >= optimal_threshold).astype(int)
```

**Q: Why adjust threshold?**
A: Default 0.5 assumes balanced classes and equal error costs. For imbalanced data, lowering threshold (e.g., to 0.3) increases recall at cost of precision.

#### Ensemble Methods

**Balanced Random Forest**: Bootstrap minority class to same size as majority

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
```

**EasyEnsemble**: Multiple balanced subsets, ensemble

**BalancedBagging**: Bagging with resampling

### 3. Metric-Based Approaches

**Use appropriate metrics!**

❌ **Don't use**: Accuracy
✅ **Use**:
- **Precision, Recall, F1**: Focus on minority class
- **ROC-AUC**: Threshold-independent
- **PR-AUC**: Better for severe imbalance
- **G-Mean**: √(Sensitivity × Specificity)

**Q: Why not accuracy for imbalanced data?**
A:
Example: 99% majority, 1% minority
- Predict all majority → 99% accuracy!
- But completely fails at minority class (0% recall)

### 4. Anomaly Detection

**For extreme imbalance** (IR > 1000), treat minority class as anomalies

**Algorithms**:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Autoencoders

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train)  # Only on normal class!
predictions = iso_forest.predict(X_test)  # -1 = anomaly, 1 = normal
```

---

## Decision Guide

```
What's the imbalance ratio?
  |
  ├─ Mild (< 10)?
  │   └─ Class weights usually sufficient
  |
  ├─ Moderate (10-100)?
  │   ├─ Large dataset? → Undersampling or class weights
  │   ├─ Small dataset? → SMOTE + class weights
  │   └─ Tree-based? → Balanced Random Forest
  |
  ├─ Severe (100-1000)?
  │   ├─ SMOTE + class weights + threshold tuning
  │   └─ Ensemble methods (BalancedBagging, EasyEnsemble)
  |
  └─ Extreme (> 1000)?
      └─ Anomaly detection (Isolation Forest, One-Class SVM)
```

---

## Best Practices

1. ✅ **Always use stratified splits** (`stratify=y`)
2. ✅ **Use appropriate metrics** (F1, PR-AUC, not accuracy)
3. ✅ **Try multiple approaches** and compare
4. ✅ **Apply resampling AFTER train-test split** (avoid data leakage!)
5. ✅ **Consider business costs** (false positives vs false negatives)
6. ✅ **Collect more minority data** if possible
7. ✅ **Feature engineering** can help more than resampling
8. ✅ **Cross-validate** with stratified folds

---

## Common Pitfalls

1. ❌ Using accuracy as metric
2. ❌ Resampling before train-test split (data leakage!)
3. ❌ Applying SMOTE to test set (only train!)
4. ❌ Using default 0.5 threshold
5. ❌ Not using stratified split/CV
6. ❌ Ignoring class distribution in production
7. ❌ Over-relying on resampling without trying class weights

---

## Interview Questions

**Q: How to handle imbalanced data?**
A: Multiple strategies:
1. **Resampling**: SMOTE (oversample minority), undersampling (remove majority)
2. **Class weights**: Penalize minority misclassifications more
3. **Threshold tuning**: Lower threshold to increase recall
4. **Ensemble methods**: Balanced Random Forest, EasyEnsemble
5. **Anomaly detection**: For extreme imbalance
6. **Metrics**: Use F1, PR-AUC instead of accuracy

**Q: SMOTE vs Random Oversampling?**
A:
- **Random**: Duplicates exact samples → overfitting risk
- **SMOTE**: Creates synthetic samples → more diverse, less overfitting
- **SMOTE** is better in most cases

**Q: When to use undersampling vs oversampling?**
A:
- **Undersampling**: Large dataset, majority class dominates
- **Oversampling (SMOTE)**: Small-medium dataset, can't afford to lose data
- **Combination**: Often best (SMOTE + clean up with undersampling)

**Q: Why not oversample before train-test split?**
A: **Data leakage!** Synthetic samples in train might be very similar to samples in test, leading to overly optimistic evaluation.

**Q: Class weights vs SMOTE?**
A:
- **Class weights**: Faster, no data modification, works with any algorithm supporting it
- **SMOTE**: Creates actual new samples, works with any algorithm
- Try both, often combine for best results

**Q: How to evaluate on imbalanced data?**
A: Never use accuracy alone!
- **Classification report**: Precision, recall, F1 per class
- **Confusion matrix**: See actual TP, FP, TN, FN
- **ROC-AUC**: Threshold-independent
- **PR-AUC**: Better for severe imbalance
- **Stratified CV**: Maintains class distribution

**Q: Balanced Random Forest vs SMOTE + Regular RF?**
A:
- **Balanced RF**: Built-in balancing, each tree sees balanced sample
- **SMOTE + RF**: Creates balanced dataset once, all trees see same data
- **Balanced RF** often more robust

**Q: What if resampling doesn't work?**
A:
1. Check if problem is truly about imbalance or poor features
2. Try anomaly detection approach
3. Collect more minority samples if possible
4. Ensemble different resampling strategies
5. Consider cost-sensitive learning

---

## Code Example: Complete Workflow

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Stratified split (IMPORTANT!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Pipeline with SMOTE (prevents data leakage)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 3. Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print(f"CV F1 scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 4. Train on full training set
pipeline.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = pipeline.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# 6. Threshold tuning
y_proba = pipeline.predict_proba(X_test)[:, 1]
# Find optimal threshold using PR curve or F1 maximization
```

---

## One-Minute Summary

"**Imbalanced data**: One class dominates (fraud 0.1%, disease 1%, spam 20%). **Problem**: Models biased to majority, high accuracy but poor minority performance. **Solutions**: (1) **Resampling**: SMOTE creates synthetic minority samples (better than random oversampling), undersampling removes majority; (2) **Class weights**: Penalize minority errors more (`class_weight='balanced'`); (3) **Threshold tuning**: Lower from 0.5 to increase recall; (4) **Ensemble**: Balanced Random Forest, EasyEnsemble; (5) **Anomaly detection**: For extreme imbalance. **Metrics**: Never use accuracy! Use F1, precision, recall, ROC-AUC, PR-AUC. **Critical**: Apply resampling AFTER train-test split (data leakage!), always use stratified splits. Best practice: Combine SMOTE + class weights + threshold tuning."
