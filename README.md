# Principal Component Analysis (PCA)

## Overview

Principal Component Analysis (PCA) is a dimensionality reduction technique used to simplify complex datasets while preserving important information. It achieves this by transforming the data into a new coordinate system, where the axes (principal components) are ordered by the amount of variance they explain in the original data.

This README provides a step-by-step guide to performing PCA using Python's `scikit-learn` library on the Iris dataset.

## Contents

1. Task 2: Load the Data and Libraries
2. Task 3: Visualize the Data
3. Task 4: Standardize the Data
4. Task 5: Compute the Eigenvectors and Eigenvalues
5. Task 6: Singular Value Decomposition (SVD)
6. Task 7: Picking Principal Components Using the Explained Variance
7. Task 8: Project Data Onto Lower-Dimensional Linear Subspace

## Task 2: Load the Data and Libraries

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
iris.columns=['sepal_length', 'sepal_width', 'petal_lenth', 'petal_width', 'species']
iris.dropna(how="all", inplace=True)
```

## Task 3: Visualize the Data

```python
sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width, hue=iris.species, style=iris.species);
```

## Task 4: Standardize the Data

```python
from sklearn.preprocessing import StandardScaler
X = iris.iloc[:, 0:4].values
y = iris.species.values
X = StandardScaler().fit_transform(X)
```

## Task 5: Compute the Eigenvectors and Eigenvalues

```python
covariance_matrix = np.cov(X.T)
print("Covariance matrix:\n", covariance_matrix)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigenvectors:\n", eigen_vectors, "\n")
print("Eigenvalues:\n", eigen_values)
```

## Task 6: Singular Value Decomposition (SVD)

```python
eigen_vec_svd, _, _= np.linalg.svd(X.T)
print(eigen_vec_svd)
```

## Task 7: Picking Principal Components Using the Explained Variance

```python
for val in eigen_values:
    print(val)
variance_explained = [(i / sum(eigen_values)) * 100 for i in eigen_values]
print(variance_explained)
cumulative_variance_explained = np.cumsum(variance_explained)
print(cumulative_variance_explained)
sns.lineplot(x=[1,2,3,4], y=cumulative_variance_explained);
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs Number of components")
plt.show()
```

## Task 8: Project Data Onto Lower-Dimensional Linear Subspace

```python
projection_matrix = (eigen_vectors.T[:][:])[:2].T
print("Projection matrix: \n", projection_matrix)
X_pca = X.dot(projection_matrix)
for species in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
    sns.scatterplot(X_pca[y==species, 0], X_pca[y==species, 1])
```
