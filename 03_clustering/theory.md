# Clustering - Theoretical Foundation

## Introduction

Clustering is an **unsupervised learning** technique that groups similar data points together without labeled data.

**Goal**: Maximize intra-cluster similarity and minimize inter-cluster similarity.

**Applications**: Customer segmentation, image segmentation, anomaly detection, document organization

---

## K-Means Clustering

### Algorithm

1. Initialize K cluster centers (randomly or using K-Means++)
2. **Assignment**: Assign each point to nearest center
3. **Update**: Recalculate centers as mean of assigned points
4. Repeat 2-3 until convergence (centers don't change)

### Mathematical Formulation

**Objective**: Minimize within-cluster sum of squares (WCSS)

```
J = Σᵏᵢ₌₁ Σₓ∈Cᵢ ||x - μᵢ||²

where μᵢ = cluster center, Cᵢ = cluster i
```

### K-Means++ Initialization

Better initialization than random:
1. Choose first center randomly
2. For each remaining center:
   - Choose point with probability proportional to D(x)²
   - D(x) = distance to nearest existing center
3. Encourages spread-out initial centers

### Choosing K

**Elbow Method**: Plot WCSS vs K, look for "elbow"
- WCSS always decreases with more clusters
- Elbow = point where decrease slows significantly

**Silhouette Score**: Measures how similar point is to its own cluster vs other clusters
```
s = (b - a) / max(a, b)

where a = avg distance to points in same cluster
      b = avg distance to points in nearest cluster

Range: [-1, 1], higher is better
```

**Gap Statistic**: Compare WCSS to expected WCSS under null reference distribution

### Pros & Cons

**✅ Advantages**:
- Simple, fast: O(nKT) where T = iterations
- Scales well to large datasets
- Works well for spherical clusters

**❌ Disadvantages**:
- Must specify K beforehand
- Sensitive to initialization (use K-Means++)
- Assumes spherical clusters of similar size
- Sensitive to outliers
- Only finds linear cluster boundaries

---

## Hierarchical Clustering

Builds hierarchy of clusters (dendrogram).

### Types

**Agglomerative** (Bottom-up):
1. Start: Each point is own cluster
2. Repeatedly merge closest clusters
3. Stop: Single cluster or desired K clusters

**Divisive** (Top-down):
1. Start: All points in one cluster
2. Repeatedly split clusters
3. Stop: Each point is own cluster or desired K clusters

### Linkage Methods

How to measure distance between clusters?

**Single Linkage**: Minimum distance between any two points
```
d(C₁, C₂) = min{d(x, y) : x ∈ C₁, y ∈ C₂}
```
- Produces elongated clusters
- Sensitive to noise/outliers ("chaining")

**Complete Linkage**: Maximum distance between any two points
```
d(C₁, C₂) = max{d(x, y) : x ∈ C₁, y ∈ C₂}
```
- Produces compact clusters
- Less sensitive to outliers

**Average Linkage**: Average distance between all pairs
```
d(C₁, C₂) = avg{d(x, y) : x ∈ C₁, y ∈ C₂}
```
- Compromise between single and complete
- Most commonly used

**Ward's Method**: Minimizes within-cluster variance
- Merge clusters that increase total within-cluster variance least
- Produces balanced clusters
- Popular choice

### Dendrogram

Tree diagram showing hierarchy of clusters.
- Cut at different heights → different number of clusters
- Height = distance at which clusters merge

### Pros & Cons

**✅ Advantages**:
- Don't need to specify K beforehand
- Dendogram provides visualization
- Can capture hierarchy
- Deterministic (no random initialization)

**❌ Disadvantages**:
- Slow: O(n³) time, O(n²) space
- Can't undo previous merges (greedy)
- Sensitive to noise and outliers
- Not suitable for large datasets

---

## DBSCAN (Density-Based Spatial Clustering)

Clusters based on density, can find arbitrary-shaped clusters.

### Concepts

**Core Point**: Has at least MinPts points within radius ε (including itself)

**Border Point**: Not core, but within ε of a core point

**Noise**: Neither core nor border

**Density-reachable**: Point p is density-reachable from q if there's chain of core points from q to p

### Algorithm

1. For each unvisited point:
   - Mark as visited
   - Find neighbors within ε
   - If neighbors < MinPts: mark as noise
   - Else: Start new cluster, expand cluster recursively

### Parameters

**ε (epsilon)**: Maximum distance for neighborhood
- Too small: Everything is noise
- Too large: Everything in one cluster

**MinPts**: Minimum points to form dense region
- Rule of thumb: MinPts ≥ dimensions + 1
- Typically: 4-10

**Choosing parameters**:
- K-distance plot: Plot distance to Kth nearest neighbor
- Look for "elbow" → good ε value

### Pros & Cons

**✅ Advantages**:
- Can find arbitrary-shaped clusters
- Identifies outliers (noise)
- Don't need to specify K
- Robust to outliers

**❌ Disadvantages**:
- Sensitive to parameters (ε, MinPts)
- Struggles with varying densities
- High-dimensional data (curse of dimensionality)
- O(n log n) with spatial index, O(n²) otherwise

---

## Gaussian Mixture Models (GMM)

Probabilistic model: Data generated from mixture of K Gaussian distributions.

### Model

```
P(x) = Σᵏᵢ₌₁ πᵢ 𝒩(x | μᵢ, Σᵢ)

where:
- πᵢ = mixing coefficient (weight) for component i, Σπᵢ = 1
- μᵢ = mean of component i
- Σᵢ = covariance matrix of component i
```

### Expectation-Maximization (EM) Algorithm

Iterative algorithm to find parameters:

**E-Step** (Expectation):
Calculate probability that point x belongs to cluster k:
```
γ(zₖ) = P(cluster k | x) = πₖ𝒩(x|μₖ,Σₖ) / Σⱼπⱼ𝒩(x|μⱼ,Σⱼ)
```

**M-Step** (Maximization):
Update parameters using weighted samples:
```
πₖ = (1/n)Σγ(zₖ)
μₖ = Σγ(zₖ)x / Σγ(zₖ)
Σₖ = Σγ(zₖ)(x-μₖ)(x-μₖ)ᵀ / Σγ(zₖ)
```

Repeat until convergence.

### GMM vs K-Means

| Aspect | K-Means | GMM |
|--------|---------|-----|
| Assignment | Hard (one cluster) | Soft (probabilities) |
| Cluster shape | Spherical | Elliptical (any covariance) |
| Model | Distance-based | Probabilistic |
| Output | Cluster labels | Probabilities |

### Pros & Cons

**✅ Advantages**:
- Soft clustering (probabilities)
- Can model elliptical clusters
- Probabilistic framework
- Can use BIC/AIC for model selection

**❌ Disadvantages**:
- Must specify K
- Sensitive to initialization
- Can converge to local optima
- Computationally expensive
- Assumes Gaussian distributions

---

## Clustering Evaluation Metrics

### Internal Metrics (No ground truth needed)

**Silhouette Score**:
```
s = (b - a) / max(a, b)
Range: [-1, 1], higher is better
```

**Davies-Bouldin Index**:
```
DB = (1/K) Σᵏᵢ₌₁ max_{j≠i} (σᵢ + σⱼ) / d(cᵢ, cⱼ)

Lower is better
```

**Calinski-Harabasz Index**:
```
CH = (Between-cluster variance / Within-cluster variance) × ((n-K)/(K-1))

Higher is better
```

### External Metrics (Requires ground truth)

**Adjusted Rand Index (ARI)**:
- Measures similarity between true and predicted clusters
- Range: [-1, 1], 1 = perfect match, 0 = random

**Normalized Mutual Information (NMI)**:
- Information theoretic measure
- Range: [0, 1], 1 = perfect match

**Purity**:
- Fraction of points correctly assigned to majority class in cluster
- Range: [0, 1], higher is better

---

## Algorithm Comparison

| Algorithm | Shape | K Required | Outliers | Speed | Use When |
|-----------|-------|------------|----------|-------|----------|
| **K-Means** | Spherical | Yes | Sensitive | Fast O(nKT) | Large data, spherical clusters |
| **Hierarchical** | Any | No | Sensitive | Slow O(n³) | Small data, need hierarchy |
| **DBSCAN** | Arbitrary | No | Robust | Medium O(n log n) | Arbitrary shapes, has outliers |
| **GMM** | Elliptical | Yes | Medium | Slow O(nK²T) | Need probabilities, elliptical |

---

## Summary

- **K-Means**: Fast, simple, spherical clusters, must specify K
- **Hierarchical**: Dendrogram, no K needed, slow, small datasets
- **DBSCAN**: Arbitrary shapes, outlier detection, parameter sensitive
- **GMM**: Probabilistic, soft clustering, elliptical clusters
