# Machine Learning - Complete Interview Preparation Guide

## Table of Contents
1. [Algorithm Cheat Sheet](#algorithm-cheat-sheet)
2. [Key Formulas](#key-formulas)
3. [Common Interview Questions](#common-interview-questions)
4. [Coding Patterns](#coding-patterns)
5. [Model Selection Guide](#model-selection-guide)
6. [Important Concepts](#important-concepts)

---

## Algorithm Cheat Sheet

### Supervised Learning

| Algorithm | Type | Pros | Cons | Scaling Required | Interpretability |
|-----------|------|------|------|------------------|------------------|
| **Linear Regression** | Regression | Fast, interpretable | Linear only | Yes | High |
| **Logistic Regression** | Classification | Probabilistic, fast | Linear boundary | Yes | High |
| **Decision Tree** | Both | Interpretable, no scaling | Overfits | No | Very High |
| **Random Forest** | Both | High performance, robust | Less interpretable | No | Low |
| **SVM** | Both | High-dim data, kernel trick | Slow, parameter tuning | Yes | Low |
| **KNN** | Both | Simple, non-linear | Slow prediction, memory | Yes | Medium |
| **Naive Bayes** | Classification | Fast, small data | Independence assumption | No | Medium |
| **Gradient Boosting** | Both | Excellent performance | Can overfit, sequential | No | Low |

### Unsupervised Learning

| Algorithm | Purpose | Pros | Cons | K Required |
|-----------|---------|------|------|------------|
| **K-Means** | Clustering | Fast, scalable | Spherical clusters, needs K | Yes |
| **Hierarchical** | Clustering | Dendrogram, no K needed | Slow O(n³) | No |
| **DBSCAN** | Clustering | Arbitrary shapes, finds outliers | Parameter sensitive | No |
| **GMM** | Clustering | Soft clustering, elliptical | Needs K, assumes Gaussian | Yes |
| **PCA** | Dimensionality Reduction | Fast, interpretable | Linear only | No |
| **t-SNE** | Visualization | Non-linear, beautiful plots | Slow, not for prediction | No |
| **UMAP** | Dim Reduction/Viz | Fast, local+global, scalable | Less interpretable than PCA | No |

---

## Key Formulas

### Regression
```
Linear Regression: Y = β₀ + β₁X₁ + ... + βₚXₚ
Normal Equation: β = (XᵀX)⁻¹XᵀY
MSE: (1/n)Σ(y - ŷ)²
RMSE: √MSE
R²: 1 - SSE/SST
Ridge: min[MSE + λΣβ²]
Lasso: min[MSE + λΣ|β|]
```

### Classification
```
Logistic: P(Y=1|X) = 1/(1 + e^(-z)), z = β₀ + βᵀX
Log Loss: -(1/n)Σ[y log(ŷ) + (1-y)log(1-ŷ)]
Accuracy: (TP + TN) / Total
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1: 2PR / (P + R)
```

### Clustering
```
K-Means: min Σᵏᵢ₌₁ Σₓ∈Cᵢ ||x - μᵢ||²
Silhouette: (b - a) / max(a, b), range [-1, 1]
GMM: P(x) = Σᵏᵢ₌₁ πᵢ𝒩(x|μᵢ, Σᵢ)
```

### Hypothesis Testing
```
t-statistic: t = (x̄ - μ₀) / (s/√n)
Chi-square: χ² = Σ(O - E)² / E
F-statistic: F = MSB / MSW
P-value: P(data as extreme | H₀ true)
Power: 1 - β (Type II error rate)
```

### PCA
```
Explained Variance: λᵢ / Σλⱼ
PC: Xw where w is eigenvector
```

---

## Common Interview Questions

### General ML

**Q: Bias vs Variance?**
A:
- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- Trade-off: Complex models (low bias, high variance), Simple models (high bias, low variance)
- Goal: Minimize total error = Bias² + Variance + Irreducible Error

**Q: Overfitting vs Underfitting?**
A:
- **Overfitting**: Model too complex, learns noise, great on train, poor on test
- **Underfitting**: Model too simple, can't capture patterns, poor on both
- **Detection**: Train vs test performance gap
- **Fix overfitting**: Regularization, more data, simpler model, dropout
- **Fix underfitting**: More complex model, more features, reduce regularization

**Q: Cross-validation?**
A: Split data into K folds, train on K-1, validate on 1, repeat K times, average performance.
- **K-Fold**: Typical K=5 or 10
- **Stratified K-Fold**: Maintains class distribution (for classification, ALWAYS use for imbalanced!)
- **Leave-One-Out**: K = n (small datasets, expensive)
- **Time Series**: TimeSeriesSplit (no future data in training)
- **Provides**: More robust estimate than single train-test split

**Q: Grid Search vs Random Search?**
A:
- **Grid Search**: Exhaustive, tries all combinations, slow, guarantees finding best in grid
- **Random Search**: Samples randomly, faster, good for many parameters, often 60 trials ≈ full grid
- **When**: Grid for 2-3 parameters, Random for many parameters or wide ranges

**Q: Regularization?**
A: Add penalty to loss function to prevent overfitting
- **L1 (Lasso)**: Σ|β| → sparse, feature selection
- **L2 (Ridge)**: Σβ² → shrinks coefficients
- **Elastic Net**: Combines L1 + L2
- **λ**: Controls strength (higher = more regularization)

**Q: Feature scaling?**
A: Transform features to similar scales
- **Standardization**: (x - μ) / σ, mean=0, std=1
- **Normalization**: (x - min) / (max - min), range=[0,1]
- **When**: Distance-based (KNN, SVM, K-Means), gradient descent (faster convergence)
- **When NOT**: Tree-based (Decision Tree, Random Forest, XGBoost)

**Q: Handling imbalanced data?**
A:
1. **Resampling**: SMOTE (creates synthetic minority samples - best!), random oversampling, undersampling
2. **Class weights**: `class_weight='balanced'` - penalizes minority errors more
3. **Threshold tuning**: Lower from 0.5 to increase recall
4. **Ensemble**: Balanced Random Forest, EasyEnsemble
5. **Metrics**: NEVER use accuracy! Use F1, precision, recall, ROC-AUC, PR-AUC
6. **Anomaly detection**: For extreme imbalance (IR > 1000)
7. **Critical**: Apply SMOTE AFTER train-test split (data leakage!), use stratified splits

**Q: SMOTE vs Random Oversampling?**
A:
- **Random**: Duplicates exact samples → overfitting
- **SMOTE**: Creates synthetic samples by interpolating between neighbors → better generalization
- **SMOTE** is almost always better

**Q: Handling missing data?**
A:
1. **Drop**: If <5% missing and random
2. **Imputation**:
   - Mean/Median/Mode (simple)
   - KNN imputer (use similar samples)
   - Model-based (predict missing values)
3. **Indicator**: Add binary "was missing" feature
4. **Advanced**: Multiple imputation, MICE

**Q: Feature selection methods?**
A:
- **Filter**: Chi-square, correlation, mutual information (before training)
- **Wrapper**: RFE (recursive feature elimination), forward/backward selection
- **Embedded**: Lasso (L1), tree importance
- **Why**: Reduce overfitting, faster training, interpretability

**Q: Training, validation, test sets?**
A:
- **Train (60-80%)**: Fit model parameters
- **Validation (10-20%)**: Tune hyperparameters, model selection
- **Test (10-20%)**: Final evaluation, never touched during development
- **Why separate**: Test estimates real-world performance

### Algorithm-Specific

**Q: Why sigmoid in logistic regression?**
A: Maps any real number to [0,1], interpretable as probability, differentiable (gradient descent), S-shaped (smooth transition)

**Q: Kernel trick in SVM?**
A: Map data to higher dimension where linearly separable, but compute K(x,x')=φ(x)·φ(x') directly without explicitly computing φ(x). Avoids expensive high-dim computation.

**Q: How Random Forest reduces overfitting?**
A: 1) Bagging (different data per tree) reduces variance, 2) Feature randomness decorrelates trees, 3) Averaging many trees smooths predictions

**Q: Gradient Boosting vs AdaBoost?**
A:
- **AdaBoost**: Reweights samples, focuses on misclassified
- **GB**: Fits residuals, more general (any loss function)
- **GB**: More powerful, used in practice (XGBoost, LightGBM)

**Q: K-Means vs GMM?**
A:
- **K-Means**: Hard clustering, spherical, distance-based
- **GMM**: Soft clustering (probabilities), elliptical, probabilistic
- **GMM**: More flexible but slower

**Q: PCA vs t-SNE vs UMAP?**
A:
- **PCA**: Linear, fast, preserves variance, interpretable, use for preprocessing
- **t-SNE**: Non-linear, slow O(n²), preserves local only, visualization ONLY
- **UMAP**: Non-linear, fast O(n log n), preserves local+global, can transform new data, use for viz AND preprocessing
- **Use**: PCA for linear preprocessing, UMAP for non-linear (better than t-SNE), t-SNE only if <10k samples and just viz

### Statistics

**Q: P-value < 0.05 meaning?**
A: If H₀ is true, only 5% chance of seeing data this extreme. Strong evidence against H₀. NOT probability H₀ is true!

**Q: Confidence interval?**
A: Range that likely contains true parameter. 95% CI: if repeat experiment many times, 95% of intervals contain true value.

**Q: Type I vs Type II error?**
A:
- **Type I (α)**: False positive, reject true H₀
- **Type II (β)**: False negative, fail to reject false H₀
- **Power**: 1-β, probability of detecting real effect

**Q: When ANOVA vs multiple t-tests?**
A: ANOVA controls family-wise error rate. 3 groups → 3 t-tests → inflated α. ANOVA tests all at once with α = 0.05.

---

## Coding Patterns

### Standard ML Pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)

# 2. Scale features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NEVER fit on test!

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predict
y_pred = model.predict(X_test_scaled)

# 5. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified K-Fold for classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

### Pipeline
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## Model Selection Guide

### Regression
```
Linear relationship? → Linear/Ridge/Lasso Regression
Non-linear? → Polynomial Regression, Random Forest, Gradient Boosting
High dimensions (p >> n)? → Lasso (feature selection), Ridge
Interpretability critical? → Linear Regression
Best performance? → XGBoost, Gradient Boosting
```

### Classification
```
Linear boundary? → Logistic Regression
Need probabilities? → Logistic Regression, Naive Bayes, Random Forest
Small dataset? → Naive Bayes, KNN
High dimensions? → SVM, Naive Bayes
Text classification? → Naive Bayes, Logistic Regression
Best performance (general)? → Random Forest, XGBoost
Interpretability? → Decision Tree, Logistic Regression
```

### Clustering
```
Know K? → K-Means (spherical), GMM (elliptical)
Don't know K? → DBSCAN, Hierarchical
Arbitrary shapes? → DBSCAN
Need dendrogram? → Hierarchical
Need probabilities? → GMM
Large dataset? → K-Means, DBSCAN
Small dataset (<1000)? → Hierarchical
```

---

## Important Concepts

### Confusion Matrix
```
                Predicted
                 +    -
Actual  +       TP   FN
        -       FP   TN

Accuracy = (TP+TN)/Total
Precision = TP/(TP+FP) - "How many selected are correct?"
Recall = TP/(TP+FN) - "How many correct are selected?"
F1 = 2PR/(P+R) - Harmonic mean
```

### ROC-AUC
- Plot TPR (recall) vs FPR at different thresholds
- AUC = 0.5: Random, AUC = 1.0: Perfect
- Good for imbalanced data (threshold-independent)

### Learning Curves
- Plot train/validation score vs training size
- **High bias (underfitting)**: Both scores low, converged, plateau
- **High variance (overfitting)**: Large gap, train score high, val score low

### Feature Importance
- **Tree-based**: Average decrease in impurity
- **Permutation**: Decrease in score when feature shuffled
- **Coefficients**: For linear models (if scaled)

### Curse of Dimensionality
- High dimensions → all points equidistant → distance metrics break
- Solutions: PCA, feature selection, specialized algorithms

---

## Quick Decision Trees

### Regression vs Classification
```
Target continuous? → Regression
Target categorical? → Classification
```

### Parametric vs Non-parametric
```
Parametric (assumes distribution): Linear/Logistic Regression, Naive Bayes
Non-parametric: KNN, Decision Trees, Random Forest
```

### Supervised vs Unsupervised
```
Have labels? → Supervised (Classification/Regression)
No labels? → Unsupervised (Clustering, Dimensionality Reduction)
```

### Online vs Batch Learning
```
Data arrives continuously? → Online (SGD, streaming algorithms)
All data at once? → Batch (most algorithms)
```

---

## Common Mistakes to Avoid

1. ❌ Data leakage (using test data in training)
2. ❌ Not scaling features for distance-based algorithms
3. ❌ Using accuracy for imbalanced data
4. ❌ Fitting scaler on test data
5. ❌ Not using cross-validation
6. ❌ Tuning hyperparameters on test set
7. ❌ Ignoring multicollinearity in linear models
8. ❌ Not checking assumptions (normality, independence, etc.)
9. ❌ Treating missing values as zeros
10. ❌ P-hacking (testing until significant)

---

## Resources for Deep Dive

- **Theory**: "Introduction to Statistical Learning" (ISLR)
- **Practice**: Kaggle competitions
- **Intuition**: StatQuest (YouTube)
- **Implementation**: Scikit-learn documentation
- **Advanced**: "Elements of Statistical Learning" (ESL)

---

## Interview Pro Tips

1. ✅ Always explain assumptions
2. ✅ Discuss bias-variance tradeoff
3. ✅ Mention overfitting concerns and solutions
4. ✅ Explain why you choose specific metrics
5. ✅ Discuss computational complexity when relevant
6. ✅ Use simple examples to explain complex concepts
7. ✅ Know when NOT to use certain algorithms
8. ✅ Understand real-world implications (fairness, interpretability)
9. ✅ Be honest when you don't know something
10. ✅ Think about production considerations (speed, memory, maintenance)
