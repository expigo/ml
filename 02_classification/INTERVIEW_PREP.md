# Classification - Interview Preparation

Quick reference for classification algorithms and common interview questions.

---

## Algorithms at a Glance

| Algorithm | Type | Decision Boundary | Pros | Cons | Scaling Required |
|-----------|------|-------------------|------|------|------------------|
| **Logistic Regression** | Linear | Linear | Interpretable, probabilistic | Linear only | Yes |
| **KNN** | Instance-based | Non-linear | Simple, no training | Slow, memory-heavy | Yes |
| **Decision Tree** | Tree | Non-linear | Interpretable, no scaling | Overfits easily | No |
| **Random Forest** | Ensemble | Non-linear | High performance | Less interpretable | No |
| **SVM** | Kernel | Linear/Non-linear | High-dim data | Slow training | Yes |
| **Naive Bayes** | Probabilistic | Non-linear | Fast, small data | Independence assumption | No |

---

## Logistic Regression

### Key Formula
```
P(Y=1|X) = σ(z) = 1/(1 + e^(-z))
where z = β₀ + β₁X₁ + ... + βₚXₚ
```

### Loss Function
```
Log Loss = -1/n Σ[y log(ŷ) + (1-y)log(1-ŷ)]
```

### Key Points
- **Output**: Probabilities [0, 1]
- **Decision boundary**: Linear (z = 0)
- **Optimization**: Gradient descent (no closed form)
- **Regularization**: L1 (Lasso) or L2 (Ridge)
- **Multi-class**: One-vs-Rest or Softmax

### Interview Q&A

**Q: Why can't we use MSE loss for logistic regression?**
A: MSE is non-convex for sigmoid function → multiple local minima → gradient descent may not converge to global minimum. Log loss is convex.

**Q: Interpret coefficient β₁ = 0.5**
A: Odds ratio = exp(0.5) ≈ 1.65. One unit increase in X₁ multiplies odds of Y=1 by 1.65 (65% increase).

**Q: Logistic regression vs Linear regression?**
A:
- Logistic: Classification, sigmoid output [0,1], log loss
- Linear: Regression, continuous output, MSE loss

---

## K-Nearest Neighbors (KNN)

### Algorithm
1. Choose K
2. Calculate distances to all training points
3. Find K nearest neighbors
4. Majority vote → class prediction

### Key Points
- **Lazy learning**: No training phase
- **Distance metric**: Usually Euclidean
- **Choosing K**: Cross-validation (odd values avoid ties)
- **Must scale features!** Distance-based algorithm

### Interview Q&A

**Q: How to choose K?**
A:
- Small K: Low bias, high variance (overfitting)
- Large K: High bias, low variance (underfitting)
- Optimal: Cross-validation
- Rule of thumb: K = √n

**Q: Curse of dimensionality?**
A: In high dimensions, all points become equidistant → KNN performance degrades. Distances lose meaning. Solution: dimensionality reduction (PCA).

**Q: Why scale features?**
A: Features with larger scales dominate distance calculation. Example: age (0-100) vs income (0-1000000). Always use StandardScaler or MinMaxScaler.

**Q: Time complexity?**
A:
- Training: O(1) (no training!)
- Prediction: O(nd) where n=samples, d=dimensions
- Can be improved with KD-trees or Ball trees to O(d log n)

---

## Decision Trees

### Splitting Criteria

**Gini Impurity**:
```
Gini = 1 - Σpᵢ²
```

**Entropy/Information Gain**:
```
Entropy = -Σpᵢ log₂(pᵢ)
Info Gain = Entropy(parent) - weighted_avg(Entropy(children))
```

### Key Points
- **Greedy algorithm**: Best split at each node (not globally optimal)
- **Overfits easily**: Use pruning, max_depth, min_samples_split
- **No feature scaling needed**
- **Feature importance**: Based on impurity reduction

### Interview Q&A

**Q: Gini vs Entropy?**
A:
- Gini: Faster (no log), range [0, 0.5]
- Entropy: More theoretically sound, range [0, 1]
- Practice: Similar results

**Q: How to prevent overfitting?**
A:
- Pre-pruning: Limit max_depth, min_samples_split, min_samples_leaf
- Post-pruning: Grow full tree, then remove branches (cost-complexity)
- Cross-validation for hyperparameters

**Q: Advantages over linear models?**
A:
- Handles non-linear relationships
- Automatic feature interaction detection
- No feature scaling needed
- Interpretable (can visualize)

**Q: Why are trees unstable?**
A: Small changes in data can lead to completely different tree structure (high variance). Solution: Ensemble methods (Random Forest).

---

## Random Forests

### How It Works
1. Create B bootstrap samples (sample with replacement)
2. For each bootstrap:
   - Grow tree with random feature subset at each split
   - Don't prune
3. Aggregate: Majority vote

### Key Parameters
- `n_estimators`: # of trees (100-500)
- `max_features`: √p for classification, p/3 for regression
- `max_depth`: Tree depth (often unlimited)
- `min_samples_split`: Min samples to split node

### Interview Q&A

**Q: Why does Random Forest reduce overfitting?**
A: Two mechanisms:
1. **Bagging**: Each tree sees different data (bootstrap) → reduces variance
2. **Feature randomness**: Decorrelates trees → ensemble is more robust

**Q: What is out-of-bag (OOB) error?**
A: Each tree trained on ~63% of data. Remaining ~37% used for validation. Free validation without separate test set!

**Q: Random Forest vs Boosting?**
A:
- RF: Parallel training, reduces variance, less prone to overfitting
- Boosting: Sequential training, reduces bias, can overfit

**Q: How to tune Random Forest?**
A: Priority order:
1. n_estimators: More is better (diminishing returns after ~300)
2. max_features: Try √p, log₂(p), p/3
3. max_depth: Often left unlimited
4. min_samples_split, min_samples_leaf: To control overfitting

---

## Support Vector Machines (SVM)

### Objective
```
Minimize: (1/2)||w||² + C Σξᵢ
Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
```

Find hyperplane that maximizes margin while allowing some errors (ξᵢ).

### Kernels

**Linear**: `K(x, x') = x·x'`

**RBF**: `K(x, x') = exp(-γ||x-x'||²)`
- Most popular
- γ large → narrow influence (overfit risk)
- γ small → wide influence (underfit risk)

**Polynomial**: `K(x, x') = (γx·x' + r)^d`

### Key Parameters
- **C**: Regularization (large C → small margin, fewer errors)
- **gamma (γ)**: Kernel coefficient (large → complex boundary)

### Interview Q&A

**Q: What is the kernel trick?**
A: Map data to higher dimension where it's linearly separable, but compute kernel K(x,x') = φ(x)·φ(x') directly without explicitly computing φ(x). Avoids expensive high-dimensional computation.

**Q: What are support vectors?**
A: Training points that lie on the margin boundary. Only these points determine the hyperplane. Removing other points doesn't change the model.

**Q: C parameter interpretation?**
A:
- Large C: Small margin, fewer training errors (overfit risk)
- Small C: Large margin, more training errors (underfit risk)
- Controls bias-variance tradeoff

**Q: When to use SVM?**
A:
- High-dimensional data (d >> n)
- Clear margin of separation
- Small-medium datasets
- Need robust model

**Q: Why scale features?**
A: SVM uses distances (in kernel functions). Features on larger scales dominate. Always standardize.

---

## Naive Bayes

### Bayes' Theorem
```
P(Y|X) = P(X|Y)P(Y) / P(X)

Classification: ŷ = argmax_y P(Y=y) Π P(Xᵢ|Y=y)
```

### Types

**Gaussian**: Continuous features, assumes normal distribution
**Multinomial**: Count data (text classification)
**Bernoulli**: Binary features

### Interview Q&A

**Q: What is the naive assumption?**
A: Features are conditionally independent given class: P(X₁,X₂|Y) = P(X₁|Y)P(X₂|Y). Rarely true but works surprisingly well!

**Q: Why is it called naive?**
A: The independence assumption is naive (unrealistic). Real features are often correlated.

**Q: What is Laplace smoothing?**
A: Add-one smoothing to handle zero probabilities:
```
P(Xᵢ|Y) = (count(Xᵢ, Y) + α) / (count(Y) + αK)
```
Prevents P(X|Y)=0 which would make entire posterior 0.

**Q: When to use Naive Bayes?**
A:
- Text classification (spam detection, sentiment analysis)
- Small datasets
- Need fast training/prediction
- Features relatively independent
- Need probability estimates

**Q: Why does it work despite violated assumptions?**
A: Classification only needs correct relative ordering of probabilities, not exact values. Even with dependencies, NB often ranks classes correctly.

---

## Evaluation Metrics

### Confusion Matrix
|       | Pred + | Pred - |
|-------|--------|--------|
| **Act +** | TP     | FN     |
| **Act -** | FP     | TN     |

### Metrics
```
Accuracy   = (TP + TN) / Total
Precision  = TP / (TP + FP)  "How many selected are relevant?"
Recall     = TP / (TP + FN)  "How many relevant are selected?"
F1         = 2·P·R / (P + R)  "Harmonic mean"
Specificity = TN / (TN + FP)
```

### Interview Q&A

**Q: When is accuracy misleading?**
A: Imbalanced datasets! Example: 95% class 0, 5% class 1. Always predicting class 0 gives 95% accuracy but is useless. Use precision, recall, F1, or AUC instead.

**Q: Precision vs Recall tradeoff?**
A:
- High Precision: Minimize false positives (spam detection)
- High Recall: Minimize false negatives (disease detection)
- F1: Balance both

**Q: What is ROC-AUC?**
A: ROC curve plots TPR vs FPR at different thresholds. AUC (area under curve):
- 0.5: Random classifier
- 1.0: Perfect classifier
- Threshold-independent
- Good for imbalanced data

**Q: Precision-Recall vs ROC curve?**
A:
- Use PR curve when: Highly imbalanced data, care more about positive class
- Use ROC when: Balanced data, care about both classes

---

## Common Interview Questions

**Q: Compare Random Forest and SVM**
A:
| Aspect | Random Forest | SVM |
|--------|---------------|-----|
| Training | Faster (parallelizable) | Slower (quadratic) |
| Interpretability | Medium (feature importance) | Low |
| High dimensions | Good | Excellent |
| Large datasets | Excellent | Poor |
| Tuning | Easier | Harder (kernel, C, γ) |

**Q: How to handle imbalanced datasets?**
A:
1. Collect more data (best but often impossible)
2. Resampling: SMOTE (oversample minority), undersample majority
3. Class weights: Penalize misclassifying minority class more
4. Anomaly detection: If extremely imbalanced
5. Metrics: Use F1, precision-recall, ROC-AUC (not accuracy)
6. Ensemble: Balanced Random Forest

**Q: Feature selection for classification?**
A:
1. Filter methods: Chi-square, mutual information, correlation
2. Wrapper methods: RFE (recursive feature elimination)
3. Embedded: Lasso (L1), tree-based importance
4. Domain knowledge

**Q: Cross-validation for classification?**
A:
- Use **Stratified K-Fold**: Maintains class distribution in each fold
- Important for imbalanced data
- Standard K-Fold might create folds with very few/no minority class samples

**Q: Multi-class classification strategies?**
A:
- **One-vs-Rest (OvR)**: K binary classifiers, pick highest probability
  - Pros: Fewer classifiers, interpretable
  - Cons: Imbalanced classes for each classifier
- **One-vs-One (OvO)**: K(K-1)/2 pairwise classifiers, majority vote
  - Pros: Each classifier sees balanced data
  - Cons: Many classifiers, slower

---

## Code Snippets

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
model.fit(X_scaled, y_train)

# Get probabilities
proba = model.predict_proba(X_test_scaled)
```

### KNN
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_scaled, y_train)
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10
)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# OOB score
model = RandomForestClassifier(oob_score=True, ...)
model.fit(X_train, y_train)
print(model.oob_score_)
```

### SVM
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_scaled, y_train)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Gaussian (continuous features)
model = GaussianNB()
model.fit(X_train, y_train)

# Multinomial (count data, text)
model = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
model.fit(X_train, y_train)
```

### Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Basic metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary')  # or 'macro', 'weighted'
rec = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Full report
print(classification_report(y_test, y_pred))

# ROC-AUC (needs probabilities)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
```

---

## Algorithm Selection Flowchart

```
Binary/Multi-class classification?
  |
  ├─ Need interpretability?
  │   ├─ Yes → Logistic Regression or Decision Tree
  │   └─ No → Continue
  |
  ├─ Dataset size?
  │   ├─ Small → Naive Bayes or KNN
  │   ├─ Medium → SVM or Random Forest
  │   └─ Large → Logistic Regression or Random Forest
  |
  ├─ Linear decision boundary?
  │   ├─ Yes → Logistic Regression
  │   └─ No → Tree-based or SVM with kernel
  |
  └─ High dimensional (d >> n)?
      ├─ Yes → SVM or Naive Bayes
      └─ No → Random Forest (best default choice)
```

---

## Quick Comparison Table

| Scenario | Best Algorithm | Why |
|----------|----------------|-----|
| Text classification | Naive Bayes | Fast, handles high dimensions well |
| Imbalanced data | Random Forest (with class_weight) | Can handle imbalance, robust |
| Need probabilities | Logistic Regression | Direct probability output |
| Small dataset | Naive Bayes, KNN | Work well with limited data |
| Need speed | Naive Bayes | Fastest training/prediction |
| Best performance (general) | Random Forest, Gradient Boosting | Often wins competitions |
| High dimensions | SVM, Naive Bayes | Curse of dimensionality resistant |
| Interpretability critical | Decision Tree, Logistic Regression | Clear decision rules/coefficients |

---

## Common Pitfalls

1. ❌ Not scaling features for KNN/SVM/Logistic Regression
2. ❌ Using accuracy for imbalanced datasets
3. ❌ Not using stratified CV for classification
4. ❌ Forgetting to set `probability=True` for SVM when needing probabilities
5. ❌ Using default hyperparameters without tuning
6. ❌ Not checking class distribution
7. ❌ Overfitting with Decision Trees (not pruning)
8. ❌ Using too small K in KNN (overfitting)

---

## One-Minute Summary

"Classification predicts discrete class labels using various algorithms. **Logistic Regression** uses sigmoid for probability estimation with linear decision boundary. **KNN** uses majority vote of K nearest neighbors - simple but slow for prediction. **Decision Trees** recursively split data using Gini/entropy - interpretable but overfit. **Random Forests** ensemble many trees with bagging and feature randomness - excellent performance. **SVM** finds maximum-margin hyperplane, uses kernel trick for non-linear boundaries. **Naive Bayes** applies Bayes' theorem with independence assumption - fast, works with small data. Evaluate using precision/recall/F1 for imbalanced data, ROC-AUC for threshold-independent comparison. Always use stratified CV for classification and scale features for distance-based methods."
