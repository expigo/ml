# Classification - Theoretical Foundation

## Table of Contents
1. [Introduction](#introduction)
2. [Logistic Regression](#logistic-regression)
3. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors)
4. [Decision Trees](#decision-trees)
5. [Random Forests](#random-forests)
6. [Support Vector Machines (SVM)](#support-vector-machines)
7. [Naive Bayes](#naive-bayes)
8. [Evaluation Metrics](#evaluation-metrics)

---

## Introduction

Classification is a supervised learning task where the goal is to predict discrete class labels (categories) rather than continuous values.

**Types**:
- **Binary Classification**: Two classes (e.g., spam/not spam, diseased/healthy)
- **Multi-class Classification**: More than two classes (e.g., digit recognition 0-9)
- **Multi-label Classification**: Multiple labels per instance

---

## Logistic Regression

Despite its name, logistic regression is a **classification** algorithm.

### Binary Logistic Regression

#### Model

Instead of predicting Y directly, we predict the probability P(Y=1|X):

```
P(Y=1|X) = σ(z) = 1 / (1 + e^(-z))

where z = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
```

**σ(z)** is the **sigmoid (logistic) function**

#### Sigmoid Function Properties

```
σ(z) = 1 / (1 + e^(-z))
```

- Maps any real number to [0, 1]
- σ(0) = 0.5
- σ(∞) = 1
- σ(-∞) = 0
- Derivative: σ'(z) = σ(z)(1 - σ(z))

#### Decision Boundary

Classify as:
- Class 1 if P(Y=1|X) ≥ threshold (typically 0.5)
- Class 0 if P(Y=1|X) < threshold

The decision boundary is where P(Y=1|X) = 0.5, i.e., where z = 0:

```
β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ = 0
```

This is a **linear** decision boundary in feature space!

#### Loss Function: Log Loss (Cross-Entropy)

Cannot use MSE because it's non-convex for logistic regression.

```
L(β) = -1/n Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

where ŷᵢ = σ(β₀ + β₁x₁ᵢ + ... + βₚxₚᵢ)
```

**Interpretation**:
- If y=1: Loss = -log(ŷ) → minimized when ŷ→1
- If y=0: Loss = -log(1-ŷ) → minimized when ŷ→0

#### Optimization

No closed-form solution! Use iterative methods:

**Gradient Descent:**

```
∇L(β) = 1/n Xᵀ(σ(Xβ) - Y)

β^(t+1) = β^(t) - α∇L(β^(t))
```

**Newton's Method** (faster convergence):

```
β^(t+1) = β^(t) - H⁻¹∇L(β^(t))

where H is the Hessian matrix
```

#### Regularization

**L2 (Ridge)**:
```
L(β) = -log-likelihood + λΣβⱼ²
```

**L1 (Lasso)**:
```
L(β) = -log-likelihood + λΣ|βⱼ|
```

### Multi-class Logistic Regression

**One-vs-Rest (OvR)**: Train K binary classifiers, pick class with highest probability

**Multinomial (Softmax)**: Generalizes sigmoid to K classes

```
P(Y=k|X) = exp(z_k) / Σexp(z_j)

where z_k = β₀^(k) + β₁^(k)X₁ + ... + βₚ^(k)Xₚ
```

### Assumptions

1. **Binary or ordinal outcome**
2. **Independence of observations**
3. **Little to no multicollinearity** among features
4. **Linearity** between log-odds and features
5. **Large sample size** (rule of thumb: 10-20 events per predictor)

### Interpretation of Coefficients

**Odds Ratio**:

```
OR = exp(βⱼ)
```

- βⱼ > 0: Odds increase by factor exp(βⱼ) for unit increase in Xⱼ
- βⱼ < 0: Odds decrease
- βⱼ = 0: No effect

**Example**: If β₁ = 0.5, then OR = exp(0.5) ≈ 1.65
- One unit increase in X₁ multiplies odds of Y=1 by 1.65

---

## K-Nearest Neighbors (KNN)

A **non-parametric**, **instance-based** learning algorithm.

### Algorithm

1. Choose number of neighbors K
2. Calculate distance from test point to all training points
3. Select K nearest neighbors
4. Assign class by majority vote

### Distance Metrics

**Euclidean Distance** (most common):
```
d(x, x') = √(Σ(xᵢ - x'ᵢ)²)
```

**Manhattan Distance**:
```
d(x, x') = Σ|xᵢ - x'ᵢ|
```

**Minkowski Distance** (generalization):
```
d(x, x') = (Σ|xᵢ - x'ᵢ|^p)^(1/p)
```
- p=1: Manhattan
- p=2: Euclidean

### Choosing K

- **K too small**: Sensitive to noise, overfitting, high variance
- **K too large**: Oversmoothing, underfitting, high bias
- **Optimal K**: Cross-validation
- **Rule of thumb**: K = √n or try odd values to avoid ties

### Weighted KNN

Instead of equal votes, weight by inverse distance:

```
weight_i = 1 / distance_i
```

Closer neighbors have more influence.

### Pros and Cons

**✅ Advantages**:
- Simple, intuitive
- No training phase (lazy learning)
- Non-parametric (no assumptions about data distribution)
- Works with multi-class naturally

**❌ Disadvantages**:
- Slow prediction (must compute distances to all points)
- Memory intensive (stores all training data)
- Sensitive to feature scaling (must standardize!)
- Curse of dimensionality (performance degrades in high dimensions)
- Sensitive to irrelevant features

### Important: Feature Scaling

KNN is **distance-based**, so features must be on same scale!

Always use StandardScaler or MinMaxScaler before KNN.

---

## Decision Trees

A tree-structured classifier that recursively partitions feature space.

### Structure

- **Root Node**: Top of tree, represents entire dataset
- **Internal Nodes**: Decision points (feature + threshold)
- **Leaf Nodes**: Class predictions
- **Branches**: Outcomes of decisions

### How Trees are Built

**Recursive Binary Splitting**:

1. Start with all data at root
2. Find best feature and split point
3. Split data into two groups
4. Repeat for each group
5. Stop when stopping criterion met

### Splitting Criteria

#### For Classification:

**Gini Impurity** (CART algorithm):

```
Gini(node) = 1 - Σpᵢ²

where pᵢ = proportion of class i in node
```

- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity (binary, 50-50 split)

**Information Gain (ID3, C4.5 algorithms)**:

Based on entropy:

```
Entropy(node) = -Σpᵢ log₂(pᵢ)

Information Gain = Entropy(parent) - Σ(nⱼ/n)Entropy(childⱼ)
```

Choose split that maximizes information gain (reduces entropy most).

**Comparison**:
- Gini: Slightly faster (no logarithm)
- Entropy: More theoretically motivated
- In practice: Similar results

### Stopping Criteria

- Maximum depth reached
- Minimum samples per node
- Minimum samples per leaf
- No information gain from split
- All samples in node are same class

### Pruning

**Problem**: Trees tend to overfit

**Solution**: Pruning

**Pre-pruning**: Stop growing early (max_depth, min_samples_split)

**Post-pruning**: Grow full tree, then remove branches
- Cost-Complexity Pruning (alpha parameter)
- Remove subtrees that don't reduce error significantly

### Feature Importance

```
Importance(feature) = Σ(weighted reduction in impurity from splits using feature)
```

Normalized to sum to 1.

### Pros and Cons

**✅ Advantages**:
- Highly interpretable
- No feature scaling needed
- Handles non-linear relationships
- Automatic feature selection
- Handles missing values (some implementations)
- Works with mixed data types

**❌ Disadvantages**:
- High variance (small data changes → big tree changes)
- Overfits easily
- Unstable
- Biased toward features with more levels
- Greedy algorithm (not guaranteed optimal)

---

## Random Forests

An **ensemble** of decision trees using **bagging** and **feature randomness**.

### Algorithm

1. **Bootstrap Sampling**: Create B bootstrap samples from training data
2. **Random Feature Selection**: At each split, consider random subset of m features
3. **Build Trees**: Grow unpruned trees on each bootstrap sample
4. **Aggregate**: Majority vote for classification

### Key Parameters

- **n_estimators**: Number of trees (typically 100-500)
- **max_features**: # features to consider per split
  - Classification: √p (default)
  - Regression: p/3
- **max_depth**: Tree depth (often grown fully)
- **min_samples_split**: Minimum samples to split node

### Why Random Forests Work

**Bagging reduces variance**:
- Each tree sees different data (bootstrap)
- Trees are decorrelated
- Averaging reduces overfitting

**Feature randomness further decorrelates trees**:
- Even if one feature is very strong, not all trees will use it
- Allows other features to contribute

### Out-of-Bag (OOB) Error

Each tree is trained on ~63% of data (due to bootstrap sampling).

The remaining ~37% (out-of-bag samples) can be used for validation **without** needing separate validation set!

```
OOB Error = 1 - (correct OOB predictions / total samples)
```

### Feature Importance

Average importance across all trees:

```
Importance = (1/B) Σ importance_from_tree_i
```

More reliable than single tree importance.

### Pros and Cons

**✅ Advantages**:
- Excellent performance out-of-the-box
- Reduces overfitting (vs single tree)
- Handles high-dimensional data
- Feature importance
- OOB error for free validation
- Parallelizable

**❌ Disadvantages**:
- Less interpretable than single tree
- Slower training/prediction than single tree
- Larger model size
- Not great for extrapolation

---

## Support Vector Machines (SVM)

Finds optimal hyperplane that maximizes margin between classes.

### Linear SVM

#### Objective

Find hyperplane `w·x + b = 0` that maximizes margin.

**Margin**: Distance from hyperplane to nearest point of each class.

**Support Vectors**: Points on the margin boundary.

#### Mathematical Formulation

```
Minimize: (1/2)||w||²

Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i
```

This is a **convex quadratic programming** problem.

#### Soft Margin (for non-separable data)

Allows some misclassifications using **slack variables ξᵢ**:

```
Minimize: (1/2)||w||² + C Σξᵢ

Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

**C parameter**:
- Large C: Smaller margin, fewer misclassifications (overfitting risk)
- Small C: Larger margin, more misclassifications (underfitting risk)

### Kernel Trick

For non-linearly separable data, map to higher dimension using **kernel function**:

```
K(x, x') = φ(x) · φ(x')
```

We never explicitly compute φ(x), just K(x, x')!

#### Common Kernels

**Linear**:
```
K(x, x') = x · x'
```

**Polynomial**:
```
K(x, x') = (γx·x' + r)^d
```

**RBF (Radial Basis Function / Gaussian)**:
```
K(x, x') = exp(-γ||x - x'||²)
```

Most popular! γ controls influence:
- Large γ: Narrow influence (overfitting risk)
- Small γ: Wide influence (underfitting risk)

**Sigmoid**:
```
K(x, x') = tanh(γx·x' + r)
```

### Multi-class SVM

**One-vs-Rest**: K binary SVM classifiers

**One-vs-One**: K(K-1)/2 pairwise classifiers, majority vote

### Pros and Cons

**✅ Advantages**:
- Effective in high dimensions
- Memory efficient (only stores support vectors)
- Versatile (different kernels)
- Works well for clear margin of separation

**❌ Disadvantages**:
- Slow training for large datasets (O(n²) to O(n³))
- Sensitive to feature scaling
- No probabilistic interpretation (by default)
- Hard to interpret
- Choosing kernel and parameters is tricky

---

## Naive Bayes

Probabilistic classifier based on **Bayes' Theorem** with **naive independence assumption**.

### Bayes' Theorem

```
P(Y|X) = P(X|Y) P(Y) / P(X)
```

Where:
- **P(Y|X)**: Posterior probability
- **P(X|Y)**: Likelihood
- **P(Y)**: Prior probability
- **P(X)**: Evidence (normalizing constant)

### Naive Assumption

Features are **conditionally independent** given class:

```
P(X₁, X₂, ..., Xₚ | Y) = P(X₁|Y) P(X₂|Y) ... P(Xₚ|Y)
```

This is rarely true in practice, but often works surprisingly well!

### Classification Rule

```
ŷ = argmax_y P(Y=y) Π P(Xᵢ|Y=y)
```

Pick class with highest posterior probability.

### Types of Naive Bayes

#### Gaussian Naive Bayes

For continuous features, assume Gaussian distribution:

```
P(Xᵢ|Y=y) = (1/√(2πσ²_y)) exp(-(Xᵢ - μ_y)²/(2σ²_y))
```

Parameters μ and σ² estimated from training data.

#### Multinomial Naive Bayes

For count data (e.g., word counts in text):

```
P(Xᵢ|Y=y) = θ_yi
```

Where θ_yi is estimated by counting.

Common in **text classification**.

#### Bernoulli Naive Bayes

For binary features:

```
P(Xᵢ|Y=y) = P(Xᵢ=1|Y=y)^Xᵢ × (1 - P(Xᵢ=1|Y=y))^(1-Xᵢ)
```

### Laplace Smoothing

Problem: If a feature value never occurs with a class, P(Xᵢ|Y) = 0, making entire product 0.

**Solution**: Add-one smoothing

```
P(Xᵢ=k|Y=y) = (count(Xᵢ=k, Y=y) + α) / (count(Y=y) + αK)
```

Where α = smoothing parameter (typically 1), K = # of classes.

### Pros and Cons

**✅ Advantages**:
- Fast training and prediction
- Works well with small datasets
- Handles high dimensions well
- Probabilistic predictions
- Works well for text classification

**❌ Disadvantages**:
- Independence assumption rarely holds
- Can be outperformed by more sophisticated models
- Sensitive to feature engineering
- Not good for capturing feature interactions

---

## Evaluation Metrics

### Confusion Matrix

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Can be misleading with imbalanced classes!

**Precision** (Positive Predictive Value):
```
Precision = TP / (TP + FP)
```
- "Of all predicted positives, how many are actually positive?"
- Important when FP is costly

**Recall** (Sensitivity, True Positive Rate):
```
Recall = TP / (TP + FN)
```
- "Of all actual positives, how many did we catch?"
- Important when FN is costly

**F1 Score** (Harmonic mean of precision and recall):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balances precision and recall
- Good for imbalanced datasets

**Specificity** (True Negative Rate):
```
Specificity = TN / (TN + FP)
```

### ROC Curve and AUC

**ROC Curve**: Plot of TPR (Recall) vs FPR at various thresholds

```
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

**AUC** (Area Under ROC Curve):
- Range: [0, 1]
- 0.5: Random classifier
- 1.0: Perfect classifier
- Threshold-independent metric

**When to use**:
- Imbalanced datasets
- Need to compare models
- Don't know optimal threshold yet

### Precision-Recall Curve

Alternative to ROC for highly imbalanced datasets.

**Average Precision**: Area under PR curve

---

## Comparison of Classifiers

| Algorithm | Type | Interpretability | Speed (Train) | Speed (Predict) | Feature Scaling | Overfitting Risk |
|-----------|------|------------------|---------------|-----------------|-----------------|------------------|
| **Logistic Regression** | Linear | High | Fast | Fast | Yes | Low-Medium |
| **KNN** | Instance-based | Medium | None (lazy) | Slow | Yes | High (small K) |
| **Decision Tree** | Tree-based | Very High | Medium | Fast | No | Very High |
| **Random Forest** | Ensemble | Low | Slow | Medium | No | Low |
| **SVM** | Kernel-based | Low | Slow | Fast | Yes | Medium |
| **Naive Bayes** | Probabilistic | Medium | Very Fast | Very Fast | No | Low |

---

## Decision Guide

**Use Logistic Regression when**:
- Need interpretable model
- Linear decision boundary
- Need probability estimates
- Baseline model

**Use KNN when**:
- Small dataset
- Non-linear decision boundary
- No training time available
- Concept: similarity-based classification

**Use Decision Tree when**:
- Need highly interpretable model
- Mixed feature types
- Don't want feature scaling
- Feature interactions important

**Use Random Forest when**:
- Want best out-of-box performance
- Can sacrifice interpretability
- Have sufficient data
- Need feature importance

**Use SVM when**:
- High-dimensional data
- Clear margin of separation
- Small to medium dataset
- Can tune hyperparameters carefully

**Use Naive Bayes when**:
- Text classification
- Need very fast training/prediction
- Small dataset
- Features are relatively independent

---

## Summary

Classification algorithms predict discrete class labels. Each has tradeoffs:

- **Logistic Regression**: Interpretable, linear, needs scaled features
- **KNN**: Simple, non-parametric, slow prediction
- **Decision Trees**: Interpretable, overfits, no scaling needed
- **Random Forests**: High performance, ensemble, less interpretable
- **SVM**: Powerful, kernel trick, slow training
- **Naive Bayes**: Fast, probabilistic, independence assumption

Choose based on:
- Dataset size and dimensionality
- Need for interpretability
- Training/prediction speed requirements
- Linear vs non-linear relationships
- Performance requirements
