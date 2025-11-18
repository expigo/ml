# Feature Engineering - Interview Preparation

## Core Concept

**Feature Engineering**: Creating, transforming, and selecting features to improve model performance.

"Applied machine learning is basically feature engineering" - Andrew Ng

---

## Feature Scaling

### Standardization (Z-score Normalization)
```
X_scaled = (X - μ) / σ

Mean = 0, Std = 1
Range: (-∞, +∞)
```

### Normalization (Min-Max Scaling)
```
X_norm = (X - min) / (max - min)

Range: [0, 1]
```

### When to Scale

**Must Scale**:
- KNN, SVM, K-Means (distance-based)
- Neural Networks
- Linear/Logistic Regression (for comparable coefficients)
- PCA (variance-based)
- Gradient Descent (faster convergence)

**Don't Need to Scale**:
- Tree-based (Decision Tree, Random Forest, XGBoost)
- Naive Bayes

### Interview Q&A

**Q: Standardization vs Normalization?**
A:
| Aspect | Standardization | Normalization |
|--------|----------------|---------------|
| Formula | (x-μ)/σ | (x-min)/(max-min) |
| Range | (-∞, +∞) | [0, 1] |
| Outliers | Less sensitive | Very sensitive |
| Use when | Normal distribution | Bounded distribution |
| Default choice | Yes | For bounded ranges |

**Q: Fit scaler on train or test?**
A: **ONLY on train!** Fit on train, transform both train and test. Otherwise data leakage.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # NO fit!
```

---

## Encoding Categorical Variables

### One-Hot Encoding
Convert categorical to binary columns

**Example**: Color = {Red, Blue, Green}
```
Red   → [1, 0, 0]
Blue  → [0, 1, 0]
Green → [0, 0, 1]
```

**Pros**: No ordinal assumption, works with all algorithms
**Cons**: High dimensionality (K categories → K columns), sparse

**When**: Nominal categories (no order), low cardinality (<10-20)

### Label Encoding
Convert categories to integers

**Example**: Size = {Small, Medium, Large} → {0, 1, 2}

**Pros**: No dimension increase
**Cons**: Implies order! (Small < Medium < Large)

**When**: Ordinal variables (has natural order), tree-based models

### Target Encoding (Mean Encoding)
Replace category with mean target value

**Pros**: Handles high cardinality, captures target relationship
**Cons**: Overfitting risk, data leakage risk

**Fix**: Use cross-validation, add smoothing

### Interview Q&A

**Q: One-hot encoding high cardinality feature (1000 categories)?**
A: Problems:
1. Creates 1000 columns (high dimension)
2. Sparse (mostly zeros)
3. Curse of dimensionality

**Solutions**:
1. Group rare categories into "Other"
2. Target encoding with cross-validation
3. Feature hashing
4. Embeddings (neural networks)

**Q: Drop first column in one-hot encoding?**
A: Yes, to avoid multicollinearity (dummy variable trap).
- K categories → K-1 columns
- Reference category = all zeros
- Important for linear models
- `drop='first'` in pandas get_dummies

**Q: Label encoding in linear models?**
A: Bad idea! Implies order.
- Category A=0, B=1, C=2 implies A < B < C
- Only use for tree-based models or ordinal variables

---

## Handling Missing Data

### Types of Missingness

**MCAR** (Missing Completely At Random): Missing is random
**MAR** (Missing At Random): Missing depends on observed data
**MNAR** (Missing Not At Random): Missing depends on unobserved data

### Strategies

#### 1. Deletion
- **Listwise**: Remove rows with any missing
- **Columnwise**: Remove features with too many missing

**When**: <5% missing, MCAR

#### 2. Imputation
- **Mean/Median/Mode**: Simple, fast
- **Forward/Backward Fill**: Time series
- **KNN Imputer**: Use similar samples
- **Model-based**: Predict missing values
- **MICE**: Multiple Imputation

**When**: >5% missing, MAR

#### 3. Add Indicator
Create binary "was_missing" feature

**When**: Missingness is informative

### Interview Q&A

**Q: Mean vs Median imputation?**
A:
- **Mean**: Sensitive to outliers, assumes normal
- **Median**: Robust to outliers, better default
- **Mode**: For categorical

**Q: Imputing before or after train-test split?**
A: **After!** Fit imputer on train, transform both. Otherwise data leakage.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

**Q: How to handle missing values in production?**
A: Save imputer/scaler objects, use same transformations on new data.

---

## Feature Creation

### Polynomial Features
Create interactions and powers

**Example**: X₁, X₂ → X₁, X₂, X₁², X₁X₂, X₂²

**When**: Capture non-linear relationships, interaction effects

**Warning**: Exponential growth (p features, degree d → (p+d choose d) features)

### Domain-Specific Features
Based on domain knowledge

**Examples**:
- **Date**: Year, month, day of week, is_weekend, hour
- **Text**: Length, word count, sentiment score
- **Geospatial**: Distance to landmarks, neighborhood
- **Ratios**: Revenue/Employees, BMI from height/weight

### Binning (Discretization)
Convert continuous to categorical

**Methods**:
- **Equal width**: Same bin width
- **Equal frequency**: Same # samples per bin
- **Custom**: Domain-driven

**When**: Capture non-linear effects, handle outliers

**Warning**: Loss of information

### Interview Q&A

**Q: When to create polynomial features?**
A:
1. Linear model + non-linear relationship
2. Small number of features (avoid explosion)
3. After checking for multicollinearity
4. Regularization (Ridge/Lasso) to handle many features

**Q: Feature engineering for time series?**
A:
- Lags: X_t-1, X_t-2, ...
- Rolling statistics: 7-day average, std
- Time indicators: hour, day_of_week, is_holiday
- Differences: X_t - X_t-1

---

## Feature Selection

Covered in detail in dimensionality reduction, brief recap:

### Filter Methods
- Correlation, Chi-square, Mutual Information
- Fast, before training

### Wrapper Methods
- RFE, Forward/Backward Selection
- Slow, uses model performance

### Embedded Methods
- Lasso, Tree importance
- During training, good balance

---

## Common Pitfalls

1. ❌ Scaling before train-test split (data leakage)
2. ❌ Label encoding nominal variables
3. ❌ Not dropping first column in one-hot (multicollinearity)
4. ❌ Imputing before splitting data
5. ❌ Creating features from test data
6. ❌ Not handling rare categories
7. ❌ Ignoring missing value patterns

---

## Best Practices

1. ✅ Always split data first, then transform
2. ✅ Fit on train only, transform train and test
3. ✅ Use pipelines to avoid data leakage
4. ✅ Save transformers for production
5. ✅ Check for data leakage
6. ✅ Start simple, add complexity gradually
7. ✅ Use domain knowledge
8. ✅ Validate feature importance

---

## Code Patterns

### Pipeline (Prevents Data Leakage)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Column Transformer (Different Transformations)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_features = ['age', 'income']
categorical_features = ['color', 'size']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

X_transformed = preprocessor.fit_transform(X_train)
```

### Complete Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model.fit(X_train, y_train)
```

---

## Decision Guide

```
Feature type?
  |
  ├─ Numeric
  │   ├─ Different scales? → Standardize/Normalize
  │   ├─ Missing values? → Impute (median)
  │   ├─ Skewed? → Log transform
  │   └─ Outliers? → Clip, robust scaling
  |
  ├─ Categorical
  │   ├─ Low cardinality (<10)? → One-Hot Encoding
  │   ├─ High cardinality (>50)? → Target Encoding, Hashing
  │   ├─ Ordinal? → Label Encoding
  │   └─ Missing values? → Treat as category or impute mode
  |
  └─ Date/Time
      └─ Extract: year, month, day, hour, day_of_week, is_weekend
```

---

## Interview Questions

**Q: How to detect data leakage?**
A:
1. Unrealistically high performance
2. Test accuracy > train accuracy
3. Feature importance shows suspicious features (e.g., ID)
4. Features that wouldn't be available at prediction time

**Prevention**:
- Split data first
- Use pipelines
- Time-based validation for time series
- Careful with target encoding

**Q: Feature engineering for imbalanced data?**
A:
1. Create features capturing minority class better
2. Interaction terms
3. Domain-specific features
4. SMOTE (synthetic oversampling)

**Q: How many features to create?**
A: Balance between:
- **Too few**: Underfitting, missing patterns
- **Too many**: Overfitting, curse of dimensionality

**Guidelines**:
- Start with domain knowledge
- Use regularization (Lasso)
- Cross-validation
- Feature selection methods
- Rule of thumb: n_samples >> n_features (10:1 ratio)

---

## One-Minute Summary

"Feature engineering transforms raw data into informative features. **Scaling**: Standardize (z-score) for most algorithms, normalize for bounded ranges. **Encoding**: One-hot for nominal (<10 categories), label for ordinal/trees, target for high cardinality (with CV). **Missing data**: Impute with median (robust), add missingness indicator if informative. **Feature creation**: Polynomial features for interactions, domain features from knowledge, date components for time. **Critical**: Split data first, fit on train only, use pipelines to prevent data leakage. Always standardize for distance-based/gradient descent methods, not needed for trees. Save all transformers for production deployment."
