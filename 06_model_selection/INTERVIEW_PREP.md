# Model Selection & Cross-Validation - Interview Preparation

## Core Concept

**Model Selection**: Choosing the best model and hyperparameters for your data
**Cross-Validation**: Technique to assess how model generalizes to unseen data

---

## Train-Test Split

### Basic Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Interview Q&A

**Q: What is train-test split?**
A: Dividing data into training set (to fit model) and test set (to evaluate generalization). Prevents overfitting assessment.

**Q: Typical split ratios?**
A:
- **80-20**: Most common (80% train, 20% test)
- **70-30**: For smaller datasets
- **60-20-20**: Train-validation-test (for hyperparameter tuning)

**Q: Why stratify?**
A: For classification, `stratify=y` maintains class distribution in both sets. Critical for imbalanced data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Q: Why set random_state?**
A: Reproducibility. Same split every time you run code.

---

## Cross-Validation

### K-Fold Cross-Validation

**How it works**:
1. Split data into K folds
2. For each fold:
   - Use it as validation set
   - Use remaining K-1 folds as training
3. Average performance across K folds

**Formula**:
```
CV Score = (1/K) Σ score_i
```

### Interview Q&A

**Q: What is cross-validation?**
A: Technique to evaluate model by training on K-1 folds and validating on 1 fold, repeating K times. Provides more robust estimate of performance than single train-test split.

**Q: Why use cross-validation?**
A:
1. Better performance estimate (reduces variance)
2. Uses all data for both training and validation
3. Detects overfitting
4. More reliable than single split (especially small datasets)

**Q: What's typical K value?**
A:
- **K=5**: Most common, good balance
- **K=10**: More thorough, slower
- **K=n (LOOCV)**: Small datasets, expensive
- **K=3**: Quick experiments

**Q: K-Fold vs Train-Test Split?**
A:
| Aspect | Train-Test | K-Fold CV |
|--------|-----------|-----------|
| Data usage | Less efficient | Uses all data |
| Variance | Higher | Lower |
| Speed | Faster | K times slower |
| Small datasets | Unreliable | Better |
| Use | Final evaluation | Model selection, tuning |

**Q: When NOT to use cross-validation?**
A:
- Very large datasets (train-test sufficient, CV too expensive)
- Time series (use time-based splits instead)
- When order matters (use specialized CV)

### Stratified K-Fold

**For classification**: Maintains class distribution in each fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Q: Why stratified?**
A: Regular K-Fold might create folds with very few samples of minority class. Stratified ensures each fold has same class distribution as original data.

### Leave-One-Out Cross-Validation (LOOCV)

**Special case**: K = n (each sample is a fold)

**Pros**:
- Maximum data for training
- Deterministic (no randomness)

**Cons**:
- Computationally expensive
- High variance in estimates
- Not practical for large datasets

**Q: LOOCV when?**
A: Very small datasets (<100 samples) where every data point matters.

### Time Series Cross-Validation

**TimeSeriesSplit**: Forward chaining, respects temporal order

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

**Q: Why different for time series?**
A: Cannot use future data to predict past! Standard K-Fold would leak future information. TimeSeriesSplit ensures training data always before validation data.

---

## Hyperparameter Tuning

### Grid Search

**Exhaustive search** over specified parameter grid

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

**Interview Q&A**

**Q: What is Grid Search?**
A: Exhaustive search trying all combinations of hyperparameters. For 3 max_depth values and 3 min_samples_split values = 3×3=9 combinations.

**Q: Complexity?**
A: O(P₁ × P₂ × ... × Pₙ × K) where Pᵢ = # values for parameter i, K = CV folds
- Can be very expensive!

**Q: Limitations?**
A:
1. Exponential growth with parameters (curse of dimensionality)
2. Only tries specified values (might miss optimal)
3. Computationally expensive
4. Equal spacing may not be optimal

### Random Search

**Randomly sample** from parameter distributions

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'n_estimators': randint(100, 500)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions,
    n_iter=50,  # Number of combinations to try
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

**Interview Q&A**

**Q: Random Search vs Grid Search?**
A:
| Aspect | Grid Search | Random Search |
|--------|-------------|---------------|
| Coverage | All combinations | Random subset |
| Efficiency | Exhaustive | More efficient |
| Speed | Slower | Faster |
| Good for | Few parameters | Many parameters |
| Optimal guarantee | Yes (in grid) | No |

**Q: Why Random Search can be better?**
A:
- Often finds good parameters faster
- Explores wider range of values
- Better for high-dimensional spaces
- Research shows: often 60 random tries ≈ full grid search

**Q: When to use which?**
A:
- **Grid Search**: 2-3 parameters, know good ranges, want exhaustive
- **Random Search**: Many parameters, wide ranges, limited compute

### Bayesian Optimization

**Smart search** using probabilistic model (not in sklearn, but important to know)

**How it works**:
1. Build probabilistic model of objective function
2. Use model to select most promising parameters
3. Evaluate, update model
4. Repeat

**Libraries**: Optuna, Hyperopt, scikit-optimize

**Q: Advantage over Random/Grid?**
A: Learns from previous trials, focuses search on promising regions. More efficient for expensive models.

---

## Nested Cross-Validation

**Problem**: Using same data for hyperparameter tuning and evaluation → overfitting!

**Solution**: Nested CV
- **Outer loop**: Model evaluation (test set simulation)
- **Inner loop**: Hyperparameter tuning

```python
from sklearn.model_selection import cross_val_score

# Inner CV for hyperparameter tuning (Grid/Random Search)
# Outer CV for final evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=inner_cv
)

# Outer CV evaluates the entire GridSearch process
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
```

**Q: Why nested CV?**
A: Prevents overfitting to validation set. Inner CV finds best hyperparameters, outer CV gives unbiased estimate of how well that process works.

**Q: When to use?**
A: When reporting final model performance. Not needed for just finding hyperparameters.

---

## Validation Strategies Summary

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Train-Test Split** | Large datasets, final eval | Fast, simple | Wastes data, high variance |
| **K-Fold CV** | Model selection, small-medium data | Efficient, low variance | K times slower |
| **Stratified K-Fold** | Classification, imbalanced | Maintains class distribution | Classification only |
| **LOOCV** | Very small datasets | Maximum data usage | Expensive, high variance |
| **TimeSeriesSplit** | Time series | Respects temporal order | Less data for training |
| **Nested CV** | Unbiased hyperparameter tuning eval | Unbiased estimate | Computationally expensive |

---

## Bias-Variance Tradeoff

**Bias**: Error from wrong assumptions (underfitting)
**Variance**: Error from sensitivity to training data (overfitting)

```
Total Error = Bias² + Variance + Irreducible Error
```

### Interview Q&A

**Q: Explain bias-variance tradeoff**
A:
- **High Bias, Low Variance**: Simple model (linear regression), underfits
- **Low Bias, High Variance**: Complex model (deep decision tree), overfits
- **Goal**: Balance both (e.g., regularized models, ensemble methods)

**Q: How to detect high bias vs high variance?**
A:
- **High Bias**: Both train and test error high, similar
- **High Variance**: Train error low, test error high (large gap)

**Q: Learning curves?**
A: Plot train/validation error vs training set size:
- **High Bias**: Both errors plateau high, close together
- **High Variance**: Large gap between train/validation, gap persists

**Q: How to reduce bias?**
A:
- More complex model
- More features
- Less regularization
- Train longer (neural networks)

**Q: How to reduce variance?**
A:
- Simpler model
- More training data
- Regularization
- Ensemble methods
- Early stopping
- Dropout (neural networks)

---

## Common Pitfalls

1. ❌ Using test set for hyperparameter tuning
2. ❌ Not using stratified CV for imbalanced classification
3. ❌ Fitting preprocessors on all data before split (data leakage!)
4. ❌ Using regular K-Fold for time series
5. ❌ Forgetting to set random_state (not reproducible)
6. ❌ Using LOOCV on large datasets
7. ❌ Not using nested CV when reporting tuned model performance

---

## Best Practices

1. ✅ Always split data FIRST, then preprocess
2. ✅ Use pipelines to prevent data leakage
3. ✅ Use stratified CV for classification
4. ✅ Use TimeSeriesSplit for time series
5. ✅ Start with Random Search, refine with Grid Search
6. ✅ Use nested CV for unbiased performance estimates
7. ✅ Set random_state for reproducibility
8. ✅ Monitor both train and validation scores (detect over/underfitting)

---

## Code Pattern: Complete Workflow

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Create pipeline (prevents data leakage)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# 3. Define parameter grid
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5]
}

# 4. Grid search with stratified CV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1
)

# 5. Fit
grid_search.fit(X_train, y_train)

# 6. Evaluate on test set (only once!)
test_score = grid_search.score(X_test, y_test)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {test_score:.3f}")
```

---

## One-Minute Summary

"**Model selection** chooses best model and hyperparameters. **Train-test split** (80-20) evaluates generalization, use `stratify=y` for classification. **Cross-validation** provides robust estimate by training on K-1 folds, validating on 1, repeating K times (typically K=5). More reliable than single split. **Stratified K-Fold** for classification, **TimeSeriesSplit** for time series. **Grid Search** tries all hyperparameter combinations (exhaustive, slow). **Random Search** samples randomly (faster, good for many parameters). Use **nested CV** for unbiased tuning evaluation. **Bias-variance tradeoff**: simple models have high bias (underfit), complex models have high variance (overfit). Always split first, use pipelines to prevent data leakage. Set random_state for reproducibility."
