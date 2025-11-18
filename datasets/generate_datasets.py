"""
Generate sample datasets for all examples in the ML library.
This script creates synthetic and real-world datasets used throughout the tutorials.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_regression, make_classification, make_blobs,
    make_moons, make_circles, load_iris, load_wine,
    load_breast_cancer, load_diabetes
)
import os


def generate_linear_regression_data():
    """Generate data for linear regression examples."""
    print("Generating linear regression datasets...")

    # Simple linear regression
    np.random.seed(42)
    X_simple = np.random.uniform(0, 10, 100).reshape(-1, 1)
    y_simple = 3 * X_simple.ravel() + 7 + np.random.normal(0, 2, 100)

    df_simple = pd.DataFrame({
        'X': X_simple.ravel(),
        'y': y_simple
    })
    df_simple.to_csv('linear_regression_simple.csv', index=False)

    # Multiple linear regression
    X_multi, y_multi = make_regression(
        n_samples=200, n_features=5, n_informative=3,
        noise=10, random_state=42
    )
    df_multi = pd.DataFrame(
        X_multi,
        columns=[f'feature_{i}' for i in range(X_multi.shape[1])]
    )
    df_multi['target'] = y_multi
    df_multi.to_csv('linear_regression_multiple.csv', index=False)

    # Polynomial regression (non-linear relationship)
    X_poly = np.random.uniform(-3, 3, 100).reshape(-1, 1)
    y_poly = 0.5 * X_poly.ravel()**2 + 2 * X_poly.ravel() + 1 + np.random.normal(0, 1, 100)

    df_poly = pd.DataFrame({
        'X': X_poly.ravel(),
        'y': y_poly
    })
    df_poly.to_csv('polynomial_regression.csv', index=False)

    print("  ✓ Linear regression datasets created")


def generate_classification_data():
    """Generate data for classification examples."""
    print("Generating classification datasets...")

    # Binary classification
    X_binary, y_binary = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1,
        random_state=42
    )
    df_binary = pd.DataFrame(X_binary, columns=['feature_1', 'feature_2'])
    df_binary['target'] = y_binary
    df_binary.to_csv('classification_binary.csv', index=False)

    # Multi-class classification
    X_multi, y_multi = make_classification(
        n_samples=400, n_features=4, n_redundant=0,
        n_informative=3, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    df_multi = pd.DataFrame(
        X_multi,
        columns=[f'feature_{i}' for i in range(X_multi.shape[1])]
    )
    df_multi['target'] = y_multi
    df_multi.to_csv('classification_multiclass.csv', index=False)

    # Non-linear classification (moons)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
    df_moons = pd.DataFrame(X_moons, columns=['feature_1', 'feature_2'])
    df_moons['target'] = y_moons
    df_moons.to_csv('classification_moons.csv', index=False)

    # Non-linear classification (circles)
    X_circles, y_circles = make_circles(
        n_samples=300, noise=0.1, factor=0.5, random_state=42
    )
    df_circles = pd.DataFrame(X_circles, columns=['feature_1', 'feature_2'])
    df_circles['target'] = y_circles
    df_circles.to_csv('classification_circles.csv', index=False)

    print("  ✓ Classification datasets created")


def generate_clustering_data():
    """Generate data for clustering examples."""
    print("Generating clustering datasets...")

    # Well-separated clusters
    X_separated, y_separated = make_blobs(
        n_samples=300, n_features=2, centers=4,
        cluster_std=0.6, random_state=42
    )
    df_separated = pd.DataFrame(X_separated, columns=['feature_1', 'feature_2'])
    df_separated['true_label'] = y_separated
    df_separated.to_csv('clustering_separated.csv', index=False)

    # Overlapping clusters
    X_overlap, y_overlap = make_blobs(
        n_samples=400, n_features=2, centers=3,
        cluster_std=1.5, random_state=42
    )
    df_overlap = pd.DataFrame(X_overlap, columns=['feature_1', 'feature_2'])
    df_overlap['true_label'] = y_overlap
    df_overlap.to_csv('clustering_overlap.csv', index=False)

    # Anisotropic clusters
    X_aniso, y_aniso = make_blobs(
        n_samples=300, n_features=2, centers=3, random_state=42
    )
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    df_aniso = pd.DataFrame(X_aniso, columns=['feature_1', 'feature_2'])
    df_aniso['true_label'] = y_aniso
    df_aniso.to_csv('clustering_anisotropic.csv', index=False)

    print("  ✓ Clustering datasets created")


def save_sklearn_datasets():
    """Save classic sklearn datasets."""
    print("Saving sklearn datasets...")

    # Iris dataset
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    df_iris['target_name'] = df_iris['target'].map({
        0: 'setosa', 1: 'versicolor', 2: 'virginica'
    })
    df_iris.to_csv('iris.csv', index=False)

    # Wine dataset
    wine = load_wine()
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine['target'] = wine.target
    df_wine.to_csv('wine.csv', index=False)

    # Breast cancer dataset
    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df_cancer['target'] = cancer.target
    df_cancer.to_csv('breast_cancer.csv', index=False)

    # Diabetes dataset
    diabetes = load_diabetes()
    df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df_diabetes['target'] = diabetes.target
    df_diabetes.to_csv('diabetes.csv', index=False)

    print("  ✓ Sklearn datasets saved")


def generate_hypothesis_testing_data():
    """Generate data for hypothesis testing examples."""
    print("Generating hypothesis testing datasets...")

    np.random.seed(42)

    # Two sample t-test data
    group_a = np.random.normal(loc=10, scale=2, size=50)
    group_b = np.random.normal(loc=12, scale=2, size=50)
    df_ttest = pd.DataFrame({
        'value': np.concatenate([group_a, group_b]),
        'group': ['A'] * 50 + ['B'] * 50
    })
    df_ttest.to_csv('hypothesis_ttest.csv', index=False)

    # ANOVA data (3 groups)
    group_1 = np.random.normal(loc=10, scale=2, size=40)
    group_2 = np.random.normal(loc=12, scale=2, size=40)
    group_3 = np.random.normal(loc=11, scale=2, size=40)
    df_anova = pd.DataFrame({
        'value': np.concatenate([group_1, group_2, group_3]),
        'group': ['Group1'] * 40 + ['Group2'] * 40 + ['Group3'] * 40
    })
    df_anova.to_csv('hypothesis_anova.csv', index=False)

    # Chi-square test data (contingency table)
    categories = np.random.choice(['A', 'B', 'C'], size=200, p=[0.4, 0.35, 0.25])
    outcomes = []
    for cat in categories:
        if cat == 'A':
            outcomes.append(np.random.choice(['Success', 'Failure'], p=[0.6, 0.4]))
        elif cat == 'B':
            outcomes.append(np.random.choice(['Success', 'Failure'], p=[0.5, 0.5]))
        else:
            outcomes.append(np.random.choice(['Success', 'Failure'], p=[0.7, 0.3]))

    df_chisq = pd.DataFrame({
        'category': categories,
        'outcome': outcomes
    })
    df_chisq.to_csv('hypothesis_chisquare.csv', index=False)

    print("  ✓ Hypothesis testing datasets created")


def generate_feature_engineering_data():
    """Generate data for feature engineering examples."""
    print("Generating feature engineering datasets...")

    np.random.seed(42)
    n_samples = 200

    # Dataset with various feature types
    df_features = pd.DataFrame({
        'numeric_1': np.random.uniform(0, 100, n_samples),
        'numeric_2': np.random.normal(50, 15, n_samples),
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'categorical_2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'ordinal': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
        'binary': np.random.choice([0, 1], n_samples),
    })

    # Add some missing values
    missing_idx = np.random.choice(n_samples, size=20, replace=False)
    df_features.loc[missing_idx, 'numeric_1'] = np.nan
    missing_idx = np.random.choice(n_samples, size=15, replace=False)
    df_features.loc[missing_idx, 'categorical_1'] = np.nan

    # Create target variable
    df_features['target'] = (
        0.5 * df_features['numeric_1'].fillna(df_features['numeric_1'].mean()) +
        0.3 * df_features['numeric_2'] +
        np.random.normal(0, 10, n_samples)
    )

    df_features.to_csv('feature_engineering.csv', index=False)

    print("  ✓ Feature engineering datasets created")


def main():
    """Generate all datasets."""
    print("\n" + "="*60)
    print("Generating Datasets for ML Library")
    print("="*60 + "\n")

    # Change to datasets directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    generate_linear_regression_data()
    generate_classification_data()
    generate_clustering_data()
    save_sklearn_datasets()
    generate_hypothesis_testing_data()
    generate_feature_engineering_data()

    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
