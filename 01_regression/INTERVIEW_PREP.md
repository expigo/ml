# Linear Regression - Interview Preparation

Quick reference guide for interviews and rapid review.

---

## Core Concepts (Must Know)

### What is Linear Regression?
A supervised learning algorithm that models the relationship between dependent and independent variables using a linear function.

**Equation**: `Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε`

---

## Key Formulas

### Simple Linear Regression
```
Y = β₀ + β₁X + ε

β₁ = Cov(X, Y) / Var(X)
β₀ = ȳ - β₁x̄
```

### Multiple Linear Regression (Matrix Form)
```
Y = Xβ + ε

β = (XᵀX)⁻¹XᵀY  (Normal Equation)
```

### Cost Function (OLS)
```
J(β) = (1/2n) Σ(yᵢ - ŷᵢ)²
```

### Regularization
```
Ridge:       J(β) = MSE + λΣβⱼ²
Lasso:       J(β) = MSE + λΣ|βⱼ|
Elastic Net: J(β) = MSE + λ₁Σ|βⱼ| + λ₂Σβⱼ²
```

---

## Assumptions (LINE + M)

| Assumption | Check | Fix |
|------------|-------|-----|
| **Linearity** | Residual plots | Add polynomial features, transformations |
| **Independence** | Durbin-Watson test | Time series models |
| **Normality** | Q-Q plot, Shapiro-Wilk | Transform Y, robust regression |
| **Equal variance** | Residual plot | Weighted LS, transform Y |
| **No Multicollinearity** | VIF > 10 | Remove features, regularization, PCA |

**VIF**: `VIF_j = 1 / (1 - R²_j)`

---

## Evaluation Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **R²** | `1 - SSE/SST` | [0, 1] | % variance explained |
| **Adjusted R²** | `1 - (1-R²)(n-1)/(n-p-1)` | (-∞, 1] | Penalizes # features |
| **MSE** | `(1/n)Σ(yᵢ - ŷᵢ)²` | [0, ∞) | Avg squared error |
| **RMSE** | `√MSE` | [0, ∞) | Same units as Y |
| **MAE** | `(1/n)Σ|yᵢ - ŷᵢ|` | [0, ∞) | Robust to outliers |

---

## Ridge vs Lasso vs Elastic Net

| Feature | Ridge (L2) | Lasso (L1) | Elastic Net |
|---------|------------|------------|-------------|
| **Penalty** | `λΣβ²` | `λΣ|β|` | `λ₁Σ|β| + λ₂Σβ²` |
| **Solution** | Closed-form | Iterative | Iterative |
| **Coefficients** | Shrink toward 0 | Can be exactly 0 | Can be exactly 0 |
| **Feature Selection** | No | Yes | Yes |
| **Multicollinearity** | Handles well | Picks one arbitrarily | Handles well |
| **Use When** | Many correlated features | Few important features | Best of both |

**Key**: Higher λ = more regularization = simpler model

---

## Common Interview Questions

### 1. **Explain linear regression in simple terms**
**Answer**: Linear regression finds the best-fitting straight line through data points by minimizing the sum of squared differences between actual and predicted values. It assumes a linear relationship between features and target.

### 2. **What are the assumptions of linear regression?**
**Answer**:
- **Linearity**: Linear relationship between X and Y
- **Independence**: Observations are independent
- **Normality**: Residuals are normally distributed
- **Equal variance** (Homoscedasticity): Constant variance of residuals
- **No multicollinearity**: Features aren't highly correlated

### 3. **How do you check if assumptions are violated?**
**Answer**:
- Linearity: Residual plots (should be random)
- Normality: Q-Q plots, Shapiro-Wilk test
- Homoscedasticity: Residual vs fitted plot (horizontal band)
- Multicollinearity: VIF > 10 indicates problem

### 4. **What is multicollinearity and why is it a problem?**
**Answer**: When independent variables are highly correlated. Problems:
- Unstable coefficient estimates (high variance)
- Difficult to determine individual feature effects
- Coefficients can have wrong signs

**Solutions**: Remove correlated features, use PCA, apply regularization

### 5. **Explain R² and its limitations**
**Answer**: R² measures the proportion of variance in Y explained by the model (0-1 range).

**Limitations**:
- Always increases with more features (even irrelevant ones)
- Doesn't indicate if model is appropriate
- Can be high even if predictions are biased

**Solution**: Use Adjusted R² which penalizes additional features

### 6. **Ridge vs Lasso: When to use which?**
**Answer**:
- **Ridge**: When you believe most features are relevant, handles multicollinearity well
- **Lasso**: When you want feature selection (sparse model), only few features important
- **Elastic Net**: Combines both benefits, good default choice

### 7. **How to handle non-linear relationships?**
**Answer**:
- Add polynomial features (X²,  X³)
- Apply transformations (log, sqrt, reciprocal)
- Use interaction terms (X₁ × X₂)
- Consider non-linear models (trees, neural networks)

### 8. **What is the Normal Equation and when NOT to use it?**
**Answer**: `β = (XᵀX)⁻¹XᵀY` - closed-form solution for OLS.

**Don't use when**:
- Large datasets (O(n³) complexity)
- XᵀX is not invertible (multicollinearity)
- Need online learning

**Alternative**: Gradient descent

### 9. **Explain gradient descent for linear regression**
**Answer**: Iterative optimization algorithm:
```
Initialize β randomly
Repeat:
    β = β - α∇J(β)
    where ∇J(β) = (1/n)Xᵀ(Xβ - Y)
Until convergence
```
**α** is learning rate - too small = slow, too large = divergence

### 10. **How to detect and handle outliers?**
**Answer**:
**Detection**:
- Visual: Box plots, scatter plots
- Statistical: Z-score > 3, IQR method
- Leverage: Cook's distance, leverage values

**Handling**:
- Investigate first (data error vs real?)
- Remove if erroneous
- Transform variables
- Use robust regression (Huber, RANSAC)
- Winsorize (cap extreme values)

### 11. **Difference between prediction and confidence intervals?**
**Answer**:
- **Confidence Interval**: Range for the mean response (E[Y|X])
  - Narrower
  - "Where is the line?"

- **Prediction Interval**: Range for individual prediction
  - Wider (includes error variance)
  - "Where will next point fall?"

### 12. **What is heteroscedasticity?**
**Answer**: Non-constant variance of residuals across predicted values.

**Problems**:
- Biased standard errors
- Invalid confidence intervals and p-values

**Solutions**:
- Transform Y (log, sqrt)
- Weighted least squares
- Robust standard errors

### 13. **Explain overfitting and underfitting**
**Answer**:
- **Underfitting**: Model too simple, high bias, poor on train & test
  - Fix: Add features, increase complexity

- **Overfitting**: Model too complex, high variance, good on train, poor on test
  - Fix: Regularization, remove features, more data

**Detection**: Large gap between train and test performance

### 14. **How to choose regularization parameter λ?**
**Answer**:
- Cross-validation (most common)
- Information criteria (AIC, BIC)
- Regularization path analysis
- Grid search or random search

### 15. **Interpret regression coefficients**
**Answer**:
- **β₁ = 5**: One unit increase in X₁ is associated with 5 unit increase in Y, holding all other variables constant
- **Standardized data**: Coefficients represent relative importance
- **Interaction terms**: Effect of X₁ depends on value of X₂

---

## Code Snippets (Must Remember)

### Train Simple Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Extract parameters
intercept = model.intercept_
coefficients = model.coef_
```

### Train with Regularization
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Always standardize for regularization!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
```

### Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Evaluation
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
```

### Check VIF
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)
```

---

## Decision Tree

```
Need to predict continuous value?
    |
    ├─ Linear relationship?
    │   ├─ Yes → Linear Regression
    │   └─ No → Polynomial Regression or Non-linear model
    |
    ├─ Many features?
    │   ├─ p >> n → Lasso or Elastic Net (feature selection)
    │   └─ Multicollinearity → Ridge or Elastic Net
    |
    └─ Overfitting?
        ├─ Yes → Regularization (increase λ)
        └─ No → Standard OLS
```

---

## Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Normal Equation | O(np² + p³) | O(np) |
| Gradient Descent | O(npi) where i = iterations | O(np) |
| Prediction | O(p) | O(p) |

**Note**: For large n or p, gradient descent is preferred

---

## Advantages vs Disadvantages

### ✅ Advantages
- Fast training and prediction
- Interpretable (clear coefficient meanings)
- Works well with linearly separable data
- Provides confidence intervals
- Well-studied, reliable
- Low variance (stable)

### ❌ Disadvantages
- Assumes linear relationship
- Sensitive to outliers
- Requires assumptions (LINE)
- Poor with non-linear relationships
- Can't handle categorical targets
- High bias for complex problems

---

## Quick Facts

- **Invented**: 1805 by Legendre, popularized by Gauss
- **Type**: Supervised, Regression
- **Output**: Continuous values
- **Loss Function**: Mean Squared Error (L2 loss)
- **Optimization**: Normal Equation or Gradient Descent
- **Scikit-learn**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- **Complexity**: O(p) prediction, O(np² + p³) training
- **Interpretability**: High

---

## Common Pitfalls

1. ❌ Not checking assumptions
2. ❌ Forgetting to standardize before regularization
3. ❌ Using R² alone for model evaluation
4. ❌ Not splitting train/test data
5. ❌ Ignoring multicollinearity
6. ❌ Overfitting with polynomial features
7. ❌ Using Normal Equation with singular matrix
8. ❌ Extrapolating beyond training data range

---

## One-Minute Explanation

"Linear regression predicts a continuous target by finding the best-fitting line through data points. It minimizes the sum of squared errors using either a closed-form solution (Normal Equation) or gradient descent. The model assumes a linear relationship, independent observations, normally distributed residuals, and equal variance. Key metrics include R² (variance explained) and RMSE (prediction error). To prevent overfitting with many features, we use regularization: Ridge (L2) shrinks coefficients, Lasso (L1) can zero them out for feature selection, and Elastic Net combines both. Always check assumptions, use train-test split, and consider cross-validation for robust evaluation."

---

## Related Topics
- Logistic Regression (classification)
- Polynomial Regression (non-linear relationships)
- Generalized Linear Models (GLM)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1 + L2)
- Gradient Descent
- Normal Equation
