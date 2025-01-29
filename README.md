# Cluster Analysis: Create, Visualize, and Interpret Customer Segments

## Overview

This project explores methods for performing cluster analysis, visualizing clusters through dimensionality reduction, and interpreting clusters by analyzing impactful features. It provides practical implementations of k-Means and DBSCAN clustering techniques, evaluates cluster quality using Silhouette scores, and visualizes the results using PCA and t-SNE.

## Features

- **Clustering Algorithms:** Implementation of k-Means and DBSCAN for customer segmentation.
- **Dimensionality Reduction:** Use of Principal Component Analysis (PCA) and t-SNE for cluster visualization.
- **Cluster Evaluation:** Calculation of Silhouette scores to assess clustering quality.
- **Data Preprocessing:** Normalization techniques for improved clustering performance.
- **Interactive Visualizations:** 2D and 3D plots for better interpretability of clusters.

## Installation

To run this project, ensure you have Python installed along with the required dependencies.

```sh
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Usage

1. **Prepare the Data:** Load and preprocess customer data.
2. **Apply Clustering Algorithms:** Implement k-Means and DBSCAN to segment the data.
3. **Evaluate Clusters:** Compute Silhouette scores to measure cluster separability.
4. **Visualize Results:** Use PCA and t-SNE for intuitive cluster visualization.

### Example Code

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np

# Load and normalize data
df = ...  # Load your dataset here
normalized_vectors = preprocessing.normalize(df)

# Apply k-Means Clustering
kmeans = KMeans(n_clusters=4).fit(normalized_vectors)

# Plot Inertia to determine optimal clusters
scores = [KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```

## Results

- **Optimal Cluster Selection:** The elbow method helps determine the best k value.
- **Performance Evaluation:** Silhouette scores indicate clustering effectiveness.
- **Visual Insights:** PCA and t-SNE projections provide an intuitive understanding of customer groups.


---

Happy clustering! 🚀

