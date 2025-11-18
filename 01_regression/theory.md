# Linear Regression - Theoretical Foundation

## Table of Contents
1. [Introduction](#introduction)
2. [Simple Linear Regression](#simple-linear-regression)
3. [Multiple Linear Regression](#multiple-linear-regression)
4. [Mathematical Derivation](#mathematical-derivation)
5. [Assumptions](#assumptions)
6. [Regularization](#regularization)
7. [Model Evaluation](#model-evaluation)

---

## Introduction

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features) using a linear function.

**Key Concepts:**
- **Dependent Variable (Y)**: The variable we want to predict
- **Independent Variables (X)**: The features used for prediction
- **Coefficients (β)**: Parameters that define the linear relationship
- **Residuals (ε)**: The difference between observed and predicted values

---

## Simple Linear Regression

Models the relationship between one independent variable and one dependent variable.

### Equation

```
Y = β₀ + β₁X + ε
```

Where:
- **Y**: Dependent variable (target)
- **X**: Independent variable (feature)
- **β₀**: Intercept (value of Y when X = 0)
- **β₁**: Slope (change in Y for unit change in X)
- **ε**: Error term (residual)

### Geometric Interpretation

Simple linear regression finds the best-fitting straight line through the data points. The line minimizes the sum of squared vertical distances (residuals) from points to the line.

### Parameter Estimation

Using Ordinary Least Squares (OLS), we minimize:

```
SSE = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (β₀ + β₁xᵢ))²
```

**Closed-form solutions:**

```
β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²
   = Cov(X, Y) / Var(X)

β₀ = ȳ - β₁x̄
```

---

## Multiple Linear Regression

Extends simple linear regression to multiple independent variables.

### Equation

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

Where:
- **p**: Number of features
- **βⱼ**: Coefficient for feature j (holding all other features constant)

### Matrix Notation

```
Y = Xβ + ε
```

Where:
- **Y**: n×1 vector of responses
- **X**: n×(p+1) design matrix (includes intercept column)
- **β**: (p+1)×1 vector of coefficients
- **ε**: n×1 vector of errors

### Parameter Estimation (Matrix Form)

Minimize: `SSE = (Y - Xβ)ᵀ(Y - Xβ)`

**Normal Equation:**

```
β = (XᵀX)⁻¹XᵀY
```

**Gradient Descent Alternative:**

When XᵀX is not invertible or for large datasets:

```
β^(t+1) = β^(t) - α∇J(β^(t))

where ∇J(β) = -2XᵀY + 2XᵀXβ
```

---

## Mathematical Derivation

### Deriving the Normal Equation

1. Start with SSE: `L(β) = (Y - Xβ)ᵀ(Y - Xβ)`

2. Expand:
   ```
   L(β) = YᵀY - YᵀXβ - βᵀXᵀY + βᵀXᵀXβ
   ```

3. Take derivative with respect to β:
   ```
   ∂L/∂β = -2XᵀY + 2XᵀXβ
   ```

4. Set to zero and solve:
   ```
   -2XᵀY + 2XᵀXβ = 0
   XᵀXβ = XᵀY
   β = (XᵀX)⁻¹XᵀY
   ```

### Variance of Estimators

```
Var(β̂) = σ²(XᵀX)⁻¹
```

Where σ² is the variance of errors.

**Standard Errors:**

```
SE(β̂ⱼ) = √(σ̂²[(XᵀX)⁻¹]ⱼⱼ)

where σ̂² = SSE/(n-p-1)
```

---

## Assumptions

Linear regression relies on several key assumptions (LINE):

### 1. Linearity
The relationship between X and Y is linear.

**Check:** Residual plots, scatter plots
**Violation:** Try polynomial features, transformations

### 2. Independence
Observations are independent of each other.

**Check:** Durbin-Watson test, autocorrelation plots
**Violation:** Use time series models, add lagged variables

### 3. Normality
Residuals are normally distributed.

**Check:** Q-Q plots, Shapiro-Wilk test, histograms
**Violation:** Try transformations, use robust regression

### 4. Equal Variance (Homoscedasticity)
Residuals have constant variance across all levels of X.

**Check:** Residual plots, Breusch-Pagan test
**Violation:** Try weighted least squares, transformations

### 5. No Multicollinearity (Multiple Regression)
Independent variables are not highly correlated with each other.

**Check:** Variance Inflation Factor (VIF), correlation matrix
**Violation:** Remove features, use regularization, PCA

**VIF Formula:**

```
VIF_j = 1 / (1 - R²_j)
```

Where R²_j is from regressing X_j on all other predictors.

**Rule of thumb:** VIF > 10 indicates problematic multicollinearity

---

## Regularization

Regularization adds a penalty term to prevent overfitting and handle multicollinearity.

### Ridge Regression (L2 Regularization)

**Objective:**

```
minimize: SSE + λΣβⱼ²
```

**Solution:**

```
β_ridge = (XᵀX + λI)⁻¹XᵀY
```

**Properties:**
- Shrinks coefficients toward zero (but not exactly zero)
- Good when many features are correlated
- λ controls amount of shrinkage (larger λ = more shrinkage)
- Improves stability of (XᵀX)⁻¹

### Lasso Regression (L1 Regularization)

**Objective:**

```
minimize: SSE + λΣ|βⱼ|
```

**Properties:**
- Can shrink coefficients exactly to zero (feature selection)
- Produces sparse models
- No closed-form solution (solved via coordinate descent)
- Better when only few features are truly important

### Elastic Net

Combines L1 and L2 penalties:

**Objective:**

```
minimize: SSE + λ₁Σ|βⱼ| + λ₂Σβⱼ²
```

Or equivalently:

```
minimize: SSE + λ(αΣ|βⱼ| + (1-α)Σβⱼ²)
```

**Properties:**
- α ∈ [0, 1] controls L1 vs L2 balance
- α = 1: Pure Lasso
- α = 0: Pure Ridge
- Combines benefits of both

### Choosing Regularization Parameter (λ)

1. **Cross-Validation**: Try multiple λ values, select one with best CV score
2. **Information Criteria**: AIC, BIC
3. **Regularization Path**: Plot coefficients vs λ

---

## Model Evaluation

### R-Squared (Coefficient of Determination)

```
R² = 1 - (SSE/SST)
   = 1 - (Σ(yᵢ - ŷᵢ)²)/(Σ(yᵢ - ȳ)²)
```

**Interpretation:**
- Proportion of variance in Y explained by model
- Range: [0, 1] (can be negative for very poor models)
- R² = 0.75 means model explains 75% of variance

**Limitation:** Always increases when adding features (even irrelevant ones)

### Adjusted R-Squared

Penalizes for number of predictors:

```
R²_adj = 1 - (1 - R²)(n - 1)/(n - p - 1)
```

Where:
- n: number of observations
- p: number of predictors

**Use:** Compare models with different numbers of features

### Mean Squared Error (MSE)

```
MSE = (1/n)Σ(yᵢ - ŷᵢ)²
```

**Properties:**
- Average squared difference between actual and predicted
- Same units as Y²
- Sensitive to outliers (squares large errors)

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE
```

**Properties:**
- Same units as Y
- Easier to interpret than MSE
- Common metric for regression

### Mean Absolute Error (MAE)

```
MAE = (1/n)Σ|yᵢ - ŷᵢ|
```

**Properties:**
- More robust to outliers than MSE
- Same units as Y
- Linear penalty for errors

### Comparison of Metrics

| Metric | Range | Units | Outlier Sensitivity | Interpretation |
|--------|-------|-------|---------------------|----------------|
| R² | [0, 1] | Unitless | Medium | % variance explained |
| MSE | [0, ∞) | Y² | High | Avg squared error |
| RMSE | [0, ∞) | Y | High | Avg error magnitude |
| MAE | [0, ∞) | Y | Low | Avg absolute error |

---

## Statistical Inference

### Hypothesis Testing for Coefficients

**Null Hypothesis:** H₀: βⱼ = 0 (feature j has no effect)

**Test Statistic:**

```
t = β̂ⱼ / SE(β̂ⱼ)
```

Under H₀, t follows t-distribution with (n-p-1) degrees of freedom.

**P-value:** Probability of observing such extreme t-value if H₀ is true

**Decision:** Reject H₀ if p-value < α (typically 0.05)

### Confidence Intervals

```
CI = β̂ⱼ ± t_(α/2, n-p-1) × SE(β̂ⱼ)
```

95% CI means: if we repeat sampling many times, 95% of CIs would contain true βⱼ

### F-Test for Overall Significance

Tests if at least one predictor is significant:

**Null Hypothesis:** H₀: β₁ = β₂ = ... = βₚ = 0

**Test Statistic:**

```
F = (SSR/p) / (SSE/(n-p-1))
```

Where:
- SSR: Sum of Squares Regression = Σ(ŷᵢ - ȳ)²
- SSE: Sum of Squares Error = Σ(yᵢ - ŷᵢ)²

---

## Practical Considerations

### When to Use Linear Regression

✅ **Good for:**
- Continuous target variable
- Linear or approximately linear relationships
- Interpretability is important
- Fast training and prediction needed
- Understanding feature importance

❌ **Not ideal for:**
- Highly non-linear relationships
- Target variable is categorical
- Severe outliers in data
- High-dimensional data (p >> n) without regularization

### Feature Engineering for Linear Regression

1. **Polynomial Features**: Capture non-linear relationships
   - X, X², X³

2. **Interaction Terms**: Capture combined effects
   - X₁ × X₂

3. **Transformations**: Handle skewed distributions
   - log(X), √X, 1/X

4. **Binning**: Convert continuous to categorical

### Diagnostics and Remedies

| Problem | Diagnostic | Solution |
|---------|-----------|----------|
| Non-linearity | Residual plots | Add polynomial terms, transformations |
| Heteroscedasticity | Residual vs fitted plot | Weighted LS, transform Y |
| Non-normal residuals | Q-Q plot | Transform Y, robust regression |
| Outliers | Leverage plots, Cook's distance | Remove or investigate, robust methods |
| Multicollinearity | VIF > 10 | Remove features, regularization, PCA |

---

## Summary

Linear regression is a fundamental technique that:
- Models linear relationships between features and target
- Has closed-form solution via Normal Equation
- Requires several assumptions (LINE)
- Can be extended with regularization (Ridge, Lasso, Elastic Net)
- Provides interpretable coefficients
- Serves as foundation for many advanced methods

**Next Steps:**
- See `examples.ipynb` for practical implementations
- See `INTERVIEW_PREP.md` for quick reference and common questions
