# Ensemble Methods - Interview Preparation

## Core Concept

**Ensemble**: Combine multiple models to create a stronger model.

**Wisdom of crowds**: Many weak learners → strong learner

---

## Main Techniques

### Bagging (Bootstrap Aggregating)

**Idea**: Train models on different bootstrapped samples, average predictions

**Algorithm**:
1. Create B bootstrap samples (sample with replacement)
2. Train model on each sample
3. Aggregate: Average (regression) or Vote (classification)

**Reduces**: Variance
**Example**: Random Forest

### Boosting

**Idea**: Sequentially train models, each focusing on previous errors

**Algorithm**:
1. Train model on data
2. Increase weights of misclassified samples
3. Train next model on reweighted data
4. Repeat, combine with weighted vote

**Reduces**: Bias
**Examples**: AdaBoost, Gradient Boosting, XGBoost

### Stacking

**Idea**: Train meta-model on predictions of base models

**Algorithm**:
1. Train base models (diverse types)
2. Use base predictions as features
3. Train meta-model on these features

**Reduces**: Both bias and variance
**Example**: Stacking classifier with LR, SVM, RF → Logistic Regression meta-model

---

## Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel | Sequential |
| **Focus** | Reduce variance | Reduce bias |
| **Sampling** | Bootstrap (uniform) | Weighted samples |
| **Combination** | Equal weights | Weighted by performance |
| **Overfitting** | Less prone | Can overfit |
| **Examples** | Random Forest | AdaBoost, XGBoost |
| **Speed** | Faster (parallel) | Slower (sequential) |

---

## Random Forest

**Type**: Bagging + Feature Randomness

### How it Works
1. Create B bootstrap samples
2. For each sample:
   - Grow tree with random feature subset at each split (√p features)
   - Don't prune
3. Aggregate: Majority vote (classification) or average (regression)

### Key Parameters
- `n_estimators`: Number of trees (100-500)
- `max_features`: Features per split (√p for classification)
- `max_depth`: Tree depth (often unlimited)
- `min_samples_split`: Min samples to split

### Interview Q&A

**Q: Why Random Forest works?**
A: Two mechanisms reduce variance:
1. **Bagging**: Different data for each tree
2. **Feature randomness**: Decorrelates trees (even strong features not always available)
Result: Low correlation between trees → ensemble variance reduction

**Q: OOB error?**
A: Out-of-bag error. Each tree trained on ~63% of data (bootstrap). Remaining ~37% used for validation. Free validation without separate test set!

**Q: Feature importance?**
A: Average decrease in impurity (Gini/entropy) when splitting on feature, averaged across all trees. Normalized to sum to 1.

**Q: Why not prune trees?**
A: Individual trees overfit, but averaging many overfit trees reduces variance. Fully grown trees ensure low bias.

---

## Gradient Boosting

**Type**: Boosting with gradient descent

### How it Works
1. Start with simple model (e.g., mean)
2. Calculate residuals (errors)
3. Train new model to predict residuals
4. Add to ensemble with learning rate
5. Repeat

```
F₀(x) = initial prediction
For m = 1 to M:
    residuals = y - Fₘ₋₁(x)
    hₘ(x) = model trained on residuals
    Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)  # η = learning rate
```

### Key Parameters
- `n_estimators`: Number of boosting stages
- `learning_rate (η)`: Shrinkage (0.01-0.1)
- `max_depth`: Tree depth (usually shallow, 3-6)
- `subsample`: Fraction of samples per tree (0.8)

### Interview Q&A

**Q: Learning rate intuition?**
A: Controls contribution of each tree
- High η (e.g., 1.0): Fast learning, may overfit
- Low η (e.g., 0.01): Slow learning, need more trees, better generalization
- Trade-off: n_estimators ↔ learning_rate

**Q: Why shallow trees?**
A: Boosting reduces bias. Don't need deep trees (high variance). Shallow trees (stumps or depth 3-6) are weak learners that boosting makes strong.

**Q: Gradient Boosting vs AdaBoost?**
A:
- AdaBoost: Reweights samples, focuses on misclassified
- Gradient Boosting: Fits residuals directly, more general
- GB can optimize any differentiable loss function

---

## XGBoost

**Type**: Optimized Gradient Boosting

### Improvements over Standard GB
1. **Regularization**: L1/L2 on leaf weights (prevents overfitting)
2. **Tree pruning**: Max depth first, then prune back
3. **Missing values**: Learns best direction for missing values
4. **Parallel processing**: Feature-level parallelism
5. **Built-in CV**: Easy cross-validation

### Key Parameters
```python
xgb.XGBClassifier(
    n_estimators=100,      # Number of trees
    learning_rate=0.1,     # η
    max_depth=6,           # Tree depth
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
    reg_alpha=0,           # L1 regularization
    reg_lambda=1,          # L2 regularization
)
```

### Interview Q&A

**Q: XGBoost vs Random Forest?**
A:
| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| Type | Bagging | Boosting |
| Training | Parallel | Sequential |
| Trees | Deep, unpruned | Shallow |
| Overfitting | Less prone | Can overfit (use regularization) |
| Speed | Faster | Slower |
| Tuning | Easier | More parameters |
| Performance | Good | Often better (competitions) |

**Q: How does XGBoost handle missing values?**
A: Learns optimal direction for missing values during training. For each split, tries sending missing values left or right, chooses direction that improves gain.

**Q: Regularization in XGBoost?**
A:
- **L1 (alpha)**: Sparsity in leaf weights
- **L2 (lambda)**: Smooth leaf weights
- **gamma**: Minimum loss reduction to split
- Helps prevent overfitting

---

## AdaBoost

**Type**: Adaptive Boosting

### How it Works
1. Start with equal weights: wᵢ = 1/n
2. For each iteration:
   - Train classifier on weighted data
   - Calculate error: ε = Σ wᵢ × I(yᵢ ≠ ŷᵢ)
   - Calculate classifier weight: α = log((1-ε)/ε)
   - Update sample weights: wᵢ = wᵢ × exp(α × I(incorrect))
3. Final prediction: sign(Σ αₜhₜ(x))

### Interview Q&A

**Q: How does AdaBoost focus on errors?**
A: Increases weights of misclassified samples. Next classifier sees harder examples with higher weight, focuses on correcting previous mistakes.

**Q: When does AdaBoost fail?**
A:
- Noisy data / outliers (keeps boosting noise)
- Base learner too complex (overfits)
- Too many iterations (overfits)

---

## Stacking

### How it Works
**Level 0** (Base models):
- Train diverse models (e.g., Logistic Regression, Random Forest, SVM)
- Generate predictions on validation set

**Level 1** (Meta-model):
- Use base predictions as features
- Train meta-model (often Logistic Regression)

**Key**: Use cross-validation to generate base predictions (avoid overfitting)

### Interview Q&A

**Q: Why use diverse base models?**
A: Diversity is key to ensemble success. Models with different assumptions/biases make different errors. Meta-model learns which model to trust when.

**Q: Stacking vs Bagging vs Boosting?**
A:
- **Bagging**: Same algorithm, different data → reduce variance
- **Boosting**: Same algorithm, sequential focus on errors → reduce bias
- **Stacking**: Different algorithms, meta-model combines → reduce both

**Q: Overfitting in stacking?**
A: Base models trained on train set, meta-model on validation predictions. Use K-fold CV to generate out-of-fold predictions for entire train set.

---

## Interview Q&A (General)

**Q: Why do ensembles work?**
A: Reduce variance (bagging) and/or bias (boosting) through combination. Errors of individual models cancel out if models are diverse and errors uncorrelated.

**Q: Bias-Variance trade-off in ensembles?**
A:
- **Bagging**: Reduces variance (averaging), slight bias increase
- **Boosting**: Reduces bias (sequential fitting), can increase variance
- **Stacking**: Can reduce both

**Q: When NOT to use ensembles?**
A:
- Need interpretability (single tree better)
- Limited computational resources
- Small dataset (may overfit)
- Real-time prediction critical (ensembles slower)

**Q: How to prevent overfitting in boosting?**
A:
1. Lower learning rate (more trees, less contribution each)
2. Limit tree depth (max_depth=3-6)
3. Subsample data (0.5-0.8)
4. Early stopping with validation set
5. Regularization (XGBoost)

**Q: Voting classifier?**
A: Simple ensemble: Train multiple models, take majority vote (classification) or average (regression).
- **Hard voting**: Majority class
- **Soft voting**: Average probabilities, then pick class
Soft voting often better (uses confidence).

---

## Code Snippets

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=2,
    oob_score=True,  # Use OOB error
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print("OOB Score:", rf.oob_score_)
print("Feature Importances:", rf.feature_importances_)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
```

### XGBoost
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Stumps
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
```

### Stacking
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

base_models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5  # Cross-validation for base predictions
)
stacking.fit(X_train, y_train)
```

### Voting
```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('svm', SVC(probability=True))],
    voting='soft'  # or 'hard'
)
voting.fit(X_train, y_train)
```

---

## Decision Guide

```
What's your goal?
  |
  ├─ Reduce variance (high variance model like decision tree)?
  │   └─ Use Bagging (Random Forest)
  |
  ├─ Reduce bias (underfitting)?
  │   └─ Use Boosting (Gradient Boosting, XGBoost)
  |
  ├─ Best performance (competition)?
  │   └─ Try XGBoost, LightGBM, CatBoost
  |
  ├─ Need interpretability?
  │   └─ Avoid ensembles (or use simple voting)
  |
  └─ Combine diverse models?
      └─ Use Stacking or Voting
```

---

## Common Pitfalls

1. ❌ Using too many trees in RF (diminishing returns after ~300)
2. ❌ High learning rate in boosting (overfits)
3. ❌ Deep trees in boosting (overfits)
4. ❌ Not using cross-validation for stacking base predictions
5. ❌ Ensembling similar models (need diversity!)
6. ❌ Forgetting feature scaling for some base models (stacking)

---

## One-Minute Summary

"Ensemble methods combine multiple models for better performance. **Bagging** trains models on bootstrap samples, averages predictions, reduces variance (Random Forest). **Boosting** sequentially trains models focusing on errors, reduces bias (AdaBoost, Gradient Boosting, XGBoost). **Stacking** trains meta-model on base model predictions, reduces both. Random Forest: parallel, robust, less overfitting. XGBoost: sequential, optimized GB with regularization, often wins competitions. Key parameters: n_estimators (number of models), learning_rate (boosting contribution), max_depth (tree complexity). Use bagging for variance reduction, boosting for bias reduction, stacking for combining diverse models. Trade-off: performance vs interpretability and speed."
