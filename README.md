# Data Science & Machine Learning Comprehensive Library

A complete, hands-on guide to Data Science and Machine Learning with theory, practical examples, and interview preparation materials.

## Overview

This repository is designed for learners at all levels - from beginners to those preparing for data science interviews. Each topic includes:
- **Theoretical Foundation**: Mathematical concepts and intuition
- **Step-by-Step Examples**: Detailed implementations with explanations
- **Interview Prep**: Quick reference sheets with key concepts and common questions

## Table of Contents

### 1. Regression
- **01_regression/**
  - Linear Regression (Simple & Multiple)
  - Polynomial Regression
  - Regularization (Ridge, Lasso, Elastic Net)
  - Assumptions and Diagnostics
  - [Interview Prep Summary](01_regression/INTERVIEW_PREP.md)

### 2. Classification
- **02_classification/**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - Naive Bayes
  - [Interview Prep Summary](02_classification/INTERVIEW_PREP.md)

### 3. Clustering
- **03_clustering/**
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN
  - Gaussian Mixture Models (GMM)
  - Clustering Evaluation Metrics
  - [Interview Prep Summary](03_clustering/INTERVIEW_PREP.md)

### 4. Dimensionality Reduction
- **04_dimensionality_reduction/**
  - Principal Component Analysis (PCA)
  - t-SNE
  - UMAP (Uniform Manifold Approximation and Projection)
  - Linear Discriminant Analysis (LDA)
  - Feature Selection Methods
  - [Interview Prep Summary](04_dimensionality_reduction/INTERVIEW_PREP.md)

### 5. Feature Engineering
- **05_feature_engineering/**
  - Feature Scaling (Normalization, Standardization)
  - Encoding Categorical Variables
  - Handling Missing Data
  - Feature Creation & Transformation
  - Feature Selection Techniques
  - [Interview Prep Summary](05_feature_engineering/INTERVIEW_PREP.md)

### 6. Model Selection & Evaluation
- **06_model_selection/**
  - Train-Test Split
  - Cross-Validation (K-Fold, Stratified, Leave-One-Out)
  - Hyperparameter Tuning (Grid Search, Random Search)
  - Performance Metrics
  - Bias-Variance Tradeoff
  - [Interview Prep Summary](06_model_selection/INTERVIEW_PREP.md)

### 7. Ensemble Methods
- **07_ensemble_methods/**
  - Bagging
  - Boosting (AdaBoost, Gradient Boosting, XGBoost)
  - Stacking
  - Voting Classifiers
  - [Interview Prep Summary](07_ensemble_methods/INTERVIEW_PREP.md)

### 8. Hypothesis Testing
- **08_hypothesis_testing/**
  - Fundamentals (Null/Alternative Hypotheses, p-values, significance levels)
  - t-tests (one-sample, two-sample, paired)
  - Chi-Square Tests
  - ANOVA (One-Way, Two-Way)
  - Non-parametric Tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
  - [Interview Prep Summary](08_hypothesis_testing/INTERVIEW_PREP.md)

### 9. Multiple Testing
- **09_multiple_testing/**
  - Family-Wise Error Rate (FWER)
  - Bonferroni Correction
  - False Discovery Rate (FDR)
  - Benjamini-Hochberg Procedure
  - Permutation Tests
  - [Interview Prep Summary](09_multiple_testing/INTERVIEW_PREP.md)

### 10. Evaluation Metrics
- **10_evaluation_metrics/**
  - Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Regression Metrics (MSE, RMSE, MAE, R², Adjusted R²)
  - Clustering Metrics (Silhouette Score, Davies-Bouldin Index)
  - [Interview Prep Summary](10_evaluation_metrics/INTERVIEW_PREP.md)

## Datasets

The `datasets/` directory contains sample datasets used throughout the examples:
- Synthetic datasets for demonstration
- Classic ML datasets (Iris, Wine, Boston Housing, etc.)
- Custom datasets for specific scenarios

## Interview Preparation

The `interview_prep/` directory contains:
- **QUICK_REFERENCE.md**: One-page cheat sheets for all topics
- **COMMON_QUESTIONS.md**: Frequently asked interview questions with answers
- **CODING_CHALLENGES.md**: Common coding problems and solutions

## How to Use This Repository

### For Learning:
1. Start with the theory documentation in each topic folder
2. Follow along with the step-by-step examples in Jupyter notebooks
3. Experiment with the code and try variations
4. Review the interview prep materials to solidify understanding

### For Interview Prep:
1. Review the INTERVIEW_PREP.md file in each topic folder
2. Study the quick reference guide
3. Practice coding challenges
4. Review common questions and formulate your own answers

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Clone the repository
git clone <repository-url>
cd ml

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e .

# Generate sample datasets
python datasets/generate_datasets.py

# Launch Jupyter Notebook
jupyter notebook
```

### Alternative: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Launch Jupyter Notebook
jupyter notebook
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- Jupyter
- XGBoost
- LightGBM
- UMAP-learn

See `pyproject.toml` for complete list with versions.

## Project Structure

```
ml/
├── README.md
├── pyproject.toml
├── datasets/
│   ├── generate_datasets.py
│   └── ...
├── 01_regression/
│   ├── theory.md
│   ├── examples.ipynb
│   └── INTERVIEW_PREP.md
├── 02_classification/
│   ├── theory.md
│   ├── examples.ipynb
│   └── INTERVIEW_PREP.md
├── ... (other topics)
└── interview_prep/
    ├── QUICK_REFERENCE.md
    ├── COMMON_QUESTIONS.md
    └── CODING_CHALLENGES.md
```

## Contributing

This is a learning resource. Feel free to:
- Add more examples
- Improve explanations
- Fix errors or typos
- Suggest additional topics

## License

This project is for educational purposes.

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [StatQuest YouTube Channel](https://www.youtube.com/user/joshstarmer)
- [Introduction to Statistical Learning](https://www.statlearning.com/)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

---

**Happy Learning!** 🚀
