# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Generate or Load Customer Dataset
# For demonstration, here's a sample dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualIncome': [15, 16, 17, 18, 19, 30, 32, 33, 35, 36],
    'SpendingScore': [39, 81, 6, 77, 40, 60, 55, 47, 52, 80]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Sample Dataset:")
print(df)

# 2. Data Preprocessing
# Select only relevant features for clustering
X = df[['AnnualIncome', 'SpendingScore']]

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means Clustering
# Determine the optimal number of clusters using the Elbow Method
inertia = []  # List to store inertia for different K values

for k in range(1, 11):  # Test for K=1 to K=10
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Inertia: sum of squared distances to the closest centroid

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# From the elbow curve, let's assume the optimal K is 3 (adjust based on your dataset)
optimal_k = 3

# 4. Fit K-Means with Optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', data=df, hue='Cluster', palette='Set1', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],  # Re-scale centers
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            color='black', marker='X', s=200, label='Centroids')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# 6. Display Final Results
print("Customer Segments with Clusters:")
print(df)
