import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
data = load_wine()
X = data['data']
y = data['target']

# Normalize the data using z-score normalization
X = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data projected onto the first two principle components
colors = ['red', 'green', 'blue']
for target, color in zip(np.unique(y), colors):
    mask = y == target
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f'Class {target + 1}')
plt.legend()
plt.show()
