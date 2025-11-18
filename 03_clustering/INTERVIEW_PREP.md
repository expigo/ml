# Clustering - Interview Preparation

## Quick Comparison

| Algorithm | K Required | Shape | Outliers | Speed | Best For |
|-----------|------------|-------|----------|-------|----------|
| **K-Means** | Yes | Spherical | Sensitive | O(nKT) | Large datasets, spherical clusters |
| **Hierarchical** | No | Any | Sensitive | O(n³) | Small datasets, dendrograms |
| **DBSCAN** | No | Arbitrary | Robust | O(n log n) | Arbitrary shapes, noise |
| **GMM** | Yes | Elliptical | Medium | O(nK²T) | Soft clustering, probabilities |

---

## K-Means

### Core Concepts
```
Minimize: Σᵏᵢ₌₁ Σₓ∈Cᵢ ||x - μᵢ||²

Algorithm:
1. Initialize K centers
2. Assign points to nearest center
3. Update centers = mean of assigned points
4. Repeat until convergence
```

### Interview Q&A

**Q: How to choose K?**
A:
1. Elbow method (plot WCSS vs K)
2. Silhouette score
3. Gap statistic
4. Domain knowledge

**Q: K-Means++?**
A: Better initialization. Choose first center randomly, then choose subsequent centers with probability ∝ D(x)² (distance to nearest center). Spreads centers out, faster convergence.

**Q: Why standardize features?**
A: K-Means uses Euclidean distance. Features with larger scales dominate. Always use StandardScaler.

**Q: Limitations?**
A:
- Assumes spherical clusters
- Sensitive to outliers
- Must specify K
- Can get stuck in local minima
- Linear cluster boundaries only

**Q: Time complexity?**
A: O(nKdT) where n=samples, K=clusters, d=dimensions, T=iterations. Typically T is small (~10-100).

---

## Hierarchical Clustering

### Linkage Methods

**Single**: min distance → elongated clusters, "chaining"
**Complete**: max distance → compact clusters
**Average**: avg distance → balanced (most common)
**Ward**: minimize variance → balanced clusters

### Interview Q&A

**Q: Agglomerative vs Divisive?**
A:
- Agglomerative (bottom-up): Start with n clusters, merge. More common.
- Divisive (top-down): Start with 1 cluster, split. Rarely used.

**Q: How to determine K from dendrogram?**
A: Cut at height where largest vertical distance without crossing horizontal line. This gives natural number of clusters.

**Q: Pros/Cons vs K-Means?**
A:
- Pros: No K needed, dendrogram, deterministic
- Cons: Slow O(n³), can't handle large data, can't undo merges

**Q: When to use?**
A: Small datasets (<1000 points), need hierarchy, explore different K values.

---

## DBSCAN

### Key Parameters

**ε (epsilon)**: Neighborhood radius
**MinPts**: Minimum points for core point

### Core Concepts

**Core point**: ≥ MinPts within ε
**Border point**: Not core, but within ε of core
**Noise**: Neither core nor border

### Interview Q&A

**Q: Advantages over K-Means?**
A:
1. Finds arbitrary-shaped clusters
2. Identifies outliers (noise)
3. Don't need to specify K
4. Robust to outliers

**Q: How to choose parameters?**
A:
- **ε**: K-distance plot (plot distance to Kth nearest neighbor, look for elbow)
- **MinPts**: Rule of thumb = dimensions + 1, typically 4-10

**Q: Limitations?**
A:
1. Sensitive to ε and MinPts
2. Struggles with varying densities
3. High-dimensional data (curse of dimensionality)
4. Can't cluster data with large density differences

**Q: Time complexity?**
A: O(n log n) with spatial index (KD-tree), O(n²) otherwise

**Q: DBSCAN vs K-Means?**
A:
| Aspect | K-Means | DBSCAN |
|--------|---------|---------|
| Shape | Spherical | Arbitrary |
| K | Must specify | Automatic |
| Outliers | Sensitive | Identifies |
| Speed | Faster | Slower |

---

## Gaussian Mixture Models (GMM)

### Model
```
P(x) = Σᵏᵢ₌₁ πᵢ 𝒩(x | μᵢ, Σᵢ)

EM Algorithm:
E-step: Calculate P(cluster | data)
M-step: Update parameters (π, μ, Σ)
```

### Interview Q&A

**Q: GMM vs K-Means?**
A:
- GMM: Soft clustering (probabilities), elliptical, probabilistic
- K-Means: Hard clustering (labels), spherical, distance-based
- GMM is generalization of K-Means!

**Q: What is EM algorithm?**
A: Expectation-Maximization. Iterative:
- E-step: Calculate posterior probabilities (which cluster?)
- M-step: Update parameters based on weighted samples
- Repeat until convergence

**Q: How to choose K?**
A: BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion). Lower is better. Penalizes complexity.

**Q: Covariance types?**
A:
- **Full**: Each cluster has full covariance matrix (most flexible)
- **Tied**: All clusters share same covariance
- **Diag**: Diagonal covariance (features independent within cluster)
- **Spherical**: Single variance per cluster (like K-Means)

---

## Evaluation Metrics

### Without Ground Truth

**Silhouette Score**:
```
s = (b - a) / max(a, b)
where a = avg intra-cluster distance
      b = avg nearest-cluster distance
Range: [-1, 1], higher is better
```

**Davies-Bouldin Index**: Lower is better (avg similarity between each cluster and most similar one)

**Calinski-Harabasz**: Higher is better (ratio of between/within cluster variance)

### With Ground Truth

**Adjusted Rand Index (ARI)**: [-1, 1], 1 = perfect match
**Normalized Mutual Information (NMI)**: [0, 1], 1 = perfect match

### Interview Q&A

**Q: Why not use accuracy?**
A: Clustering is unsupervised! Cluster labels are arbitrary (cluster 0 vs 1 has no meaning). Need metrics that compare similarity regardless of label names.

**Q: Silhouette interpretation?**
A:
- 1: Point well matched to cluster, far from neighbors
- 0: Point on border between clusters
- -1: Point probably in wrong cluster

---

## Common Interview Questions

**Q: Clustering vs Classification?**
A:
- Clustering: Unsupervised, no labels, find structure
- Classification: Supervised, has labels, predict class

**Q: How to handle categorical features?**
A:
- One-hot encoding (but increases dimensions)
- K-modes (like K-Means for categorical data)
- Gower distance (mixed types)

**Q: Curse of dimensionality?**
A: In high dimensions:
- All points become equidistant
- Clusters become less meaningful
- Distance metrics break down
Solution: PCA, feature selection, specialized algorithms

**Q: How to validate clustering?**
A:
1. Internal metrics (silhouette, DB index)
2. External metrics if ground truth available (ARI, NMI)
3. Visual inspection (2D projections)
4. Domain expert evaluation
5. Stability analysis (run multiple times)

**Q: What if clusters have different sizes/densities?**
A:
- K-Means: Struggles (assumes equal size)
- Hierarchical: Can handle
- DBSCAN: Struggles with varying density
- GMM: Can handle (different covariances)

**Q: How many clusters in practice?**
A: Depends on application:
- Customer segmentation: 3-10
- Image segmentation: Many
- Document clustering: Varies
Use elbow method + domain knowledge

---

## Code Snippets

### K-Means
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Always scale!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# Elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

### Hierarchical
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

# Dendrogram
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# -1 indicates noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

### GMM
```python
from sklearn.mixture import GaussianMixture

# GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)  # Soft clustering!

# BIC for model selection
bics = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
```

---

## Decision Guide

```
Clustering Task
  |
  ├─ Know K?
  │   ├─ Yes → K-Means (spherical) or GMM (elliptical)
  │   └─ No → DBSCAN or Hierarchical
  |
  ├─ Cluster shape?
  │   ├─ Spherical → K-Means
  │   ├─ Elliptical → GMM
  │   └─ Arbitrary → DBSCAN
  |
  ├─ Dataset size?
  │   ├─ Small (<1000) → Hierarchical
  │   └─ Large → K-Means or DBSCAN
  |
  ├─ Need probabilities?
  │   └─ Yes → GMM
  |
  └─ Have outliers?
      └─ Yes → DBSCAN
```

---

## Common Pitfalls

1. ❌ Not scaling features (K-Means, DBSCAN, Hierarchical)
2. ❌ Using K-Means for non-spherical clusters
3. ❌ Not considering outliers
4. ❌ Choosing K arbitrarily without validation
5. ❌ Using high-dimensional data without dimensionality reduction
6. ❌ Interpreting cluster labels as having inherent meaning
7. ❌ Not validating clustering results

---

## One-Minute Summary

"Clustering groups similar data points without labels. **K-Means** minimizes within-cluster variance, fast but assumes spherical clusters and requires K. **Hierarchical** builds dendrogram, no K needed, slow O(n³), good for small data. **DBSCAN** finds arbitrary-shaped clusters based on density, robust to outliers, doesn't need K but sensitive to parameters. **GMM** models data as mixture of Gaussians, soft clustering with probabilities, handles elliptical clusters. Evaluate using silhouette score (internal) or ARI/NMI (external). Always scale features for distance-based methods. Choose algorithm based on cluster shape, dataset size, and whether K is known."
