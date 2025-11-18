# Dimensionality Reduction - Interview Preparation

## Why Dimensionality Reduction?

1. **Curse of dimensionality**: High-dim data → sparse, distances meaningless
2. **Visualization**: Reduce to 2D/3D for plotting
3. **Computational efficiency**: Faster training, less memory
4. **Remove noise**: Keep signal, discard noise
5. **Multicollinearity**: Remove correlated features

---

## Principal Component Analysis (PCA)

### How it Works
1. Standardize features
2. Compute covariance matrix
3. Find eigenvectors (principal components) and eigenvalues
4. Sort by eigenvalues (variance explained)
5. Project data onto top K components

### Key Concepts
```
PC1 = direction of maximum variance
PC2 = orthogonal to PC1, next most variance
...

Explained Variance Ratio = λᵢ / Σλⱼ
Cumulative Variance: Keep PCs until 90-95% variance explained
```

### Interview Q&A

**Q: What is PCA?**
A: Unsupervised linear dimensionality reduction. Finds orthogonal directions (principal components) that maximize variance. Projects data onto these directions.

**Q: How to choose number of components?**
A:
1. Scree plot (elbow method)
2. Cumulative explained variance (e.g., 95%)
3. Cross-validation (for supervised task)
4. Domain knowledge

**Q: PCA assumptions?**
A:
1. Linearity (linear combinations of features)
2. Large variances = important (need to standardize!)
3. Components are orthogonal
4. Data is continuous

**Q: When to use PCA?**
A:
- High-dimensional data
- Features are correlated
- Need interpretability (PCs are combinations of features)
- Preprocessing before other algorithms

**Q: When NOT to use PCA?**
A:
- Features are already uncorrelated
- Non-linear relationships (use t-SNE, kernel PCA)
- Need exact features (PCA creates new features)
- Small datasets

**Q: Standardize before PCA?**
A: YES! PCA is variance-based. Features with larger scales dominate. Always use StandardScaler.

**Q: PCA vs Feature Selection?**
A:
- **PCA**: Creates new features (linear combinations)
- **Feature Selection**: Keeps original features
- PCA better for correlated features
- Feature Selection better for interpretability

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### How it Works
1. Compute pairwise similarities in high-dim (Gaussian)
2. Initialize random low-dim embedding
3. Compute pairwise similarities in low-dim (t-distribution)
4. Minimize KL divergence between distributions (gradient descent)

### Key Properties
- **Non-linear**: Captures complex structures
- **Preserves local structure**: Similar points stay close
- **Stochastic**: Different runs give different results
- **Slow**: O(n²), not suitable for large n

### Interview Q&A

**Q: PCA vs t-SNE?**
A:
| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Speed | Fast O(nd²) | Slow O(n²) |
| Deterministic | Yes | No (random init) |
| Interpretability | PCs have meaning | No interpretation |
| Use | Preprocessing, feature reduction | Visualization only |
| Global structure | Preserves | May distort |
| Local structure | May lose | Preserves |

**Q: When to use t-SNE?**
A: **Only for final visualization** of high-dimensional data. NOT for preprocessing or prediction.

**Q: t-SNE hyperparameters?**
A:
- **perplexity** (5-50): Balances local vs global structure. Higher = more global.
- **learning_rate** (10-1000): Too low = slow, too high = diverge
- **n_iter** (≥250): More iterations often better

**Q: Why not use t-SNE for ML pipeline?**
A:
1. Can't transform new data (no explicit mapping)
2. Stochastic (not reproducible without seed)
3. Slow
4. Distances in t-SNE space don't have meaning
5. May distort global structure

---

## Linear Discriminant Analysis (LDA)

### How it Works
**Supervised** dimensionality reduction:
1. Compute class means
2. Find directions that:
   - Maximize between-class variance
   - Minimize within-class variance
3. Project onto these directions

### Key Properties
```
Max components = min(n_features, n_classes - 1)

For binary classification: Max 1 component
For 3 classes: Max 2 components
```

### Interview Q&A

**Q: PCA vs LDA?**
A:
| Aspect | PCA | LDA |
|--------|-----|-----|
| Type | Unsupervised | Supervised |
| Goal | Maximize variance | Maximize class separation |
| Uses labels | No | Yes |
| Max components | n_features | n_classes - 1 |
| For classification | Suboptimal | Better |

**Q: When to use LDA?**
A:
- Classification task
- Want to visualize class separation
- Reduce dimensions before classifier
- Features normally distributed (assumes Gaussian)

**Q: LDA assumptions?**
A:
1. Features normally distributed
2. Classes have same covariance matrix
3. Features are independent

---

## Feature Selection Methods

### Filter Methods
Select features before training
- **Correlation**: Remove highly correlated
- **Variance**: Remove low-variance
- **Chi-square**: For categorical features
- **Mutual Information**: General dependency

**Pros**: Fast, model-agnostic
**Cons**: Ignores feature interactions

### Wrapper Methods
Use model performance to select features
- **RFE (Recursive Feature Elimination)**: Iteratively remove worst features
- **Forward Selection**: Start empty, add best
- **Backward Elimination**: Start full, remove worst

**Pros**: Considers feature interactions
**Cons**: Slow, model-dependent, overfitting risk

### Embedded Methods
Feature selection during training
- **Lasso (L1)**: Shrinks coefficients to zero
- **Tree-based importance**: Random Forest, XGBoost
- **ElasticNet**: L1 + L2

**Pros**: Fast, considers interactions, less overfitting than wrapper
**Cons**: Model-specific

### Interview Q&A

**Q: How to remove correlated features?**
A:
1. Compute correlation matrix
2. For pairs with |corr| > threshold (e.g., 0.9):
   - Keep feature with higher correlation to target
   - Or keep first one
3. Or use VIF > 10

**Q: Feature selection vs dimensionality reduction?**
A:
- **Feature Selection**: Keep subset of original features (interpretable)
- **Dimensionality Reduction**: Create new features (lose interpretability)

**Q: How to select features for linear model?**
A:
1. Lasso (L1 regularization) - automatic, sparse
2. Correlation with target
3. VIF for multicollinearity
4. Domain knowledge

---

## Code Snippets

### PCA
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always standardize first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative variance:", pca.explained_variance_ratio_.cumsum())

# Scree plot
import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.show()
```

### t-SNE
```python
from sklearn.manifold import TSNE

# t-SNE (for visualization only!)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)  # Slow!

# Visualize
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.show()
```

### LDA
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LDA(n_components=2)  # Max = n_classes - 1
X_lda = lda.fit_transform(X_scaled, y)  # Needs labels!

# Visualize class separation
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
```

### Feature Selection (Variance Threshold)
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

### RFE
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

print("Selected features:", rfe.support_)
print("Feature ranking:", rfe.ranking_)
```

### Lasso Feature Selection
```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Features with non-zero coefficients
selected = np.abs(lasso.coef_) > 1e-5
print(f"Selected {selected.sum()} out of {len(selected)} features")
```

---

## Decision Guide

```
Goal?
  |
  ├─ Visualization (2D/3D)?
  │   ├─ Linear ok? → PCA
  │   └─ Need non-linear? → t-SNE
  |
  ├─ Preprocessing for prediction?
  │   ├─ Unsupervised task? → PCA
  │   ├─ Classification? → LDA or PCA
  │   └─ Regression? → PCA
  |
  ├─ Feature selection (keep original features)?
  │   ├─ Fast, simple? → Filter methods
  │   ├─ Best performance? → Wrapper (RFE)
  │   └─ During training? → Embedded (Lasso, Trees)
  |
  └─ Remove correlated features?
      └─ Correlation matrix or VIF > 10
```

---

## Common Pitfalls

1. ❌ Not standardizing before PCA
2. ❌ Using t-SNE for anything except visualization
3. ❌ Fitting PCA on test data
4. ❌ Interpreting t-SNE distances literally
5. ❌ Using LDA without checking assumptions
6. ❌ Feature selection on entire dataset (should be in CV)

---

## One-Minute Summary

"Dimensionality reduction decreases features while preserving information. **PCA** finds orthogonal directions of maximum variance, linear, fast, interpretable (explained variance), use for preprocessing. **t-SNE** preserves local structure, non-linear, slow, stochastic, use ONLY for visualization. **LDA** maximizes class separation, supervised, max n_classes-1 components, good for classification. **Feature Selection**: Filter (fast, before training), Wrapper (RFE, slow, uses model), Embedded (Lasso, during training). Always standardize before PCA/LDA. Choose components by explained variance (95%) or cross-validation. PCA for preprocessing, t-SNE for final visualization, LDA for classification tasks."
