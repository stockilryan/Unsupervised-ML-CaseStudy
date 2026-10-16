import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import textwrap

df = pd.read_csv("survey.csv")

print(df.shape)
print(df.isnull().sum().sum())

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())


    else:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
"""Generate values for missing data points
   I used the median of a column for numerical values and the mode for categorial"""

df = pd.get_dummies(df)

print(df.shape)
print(df.isnull().sum().sum())

selector = VarianceThreshold(threshold=0.01) # removes features with very low variance
selector.fit(df)

cols_to_keep = df.columns[selector.get_support()]
df_filtered = df[cols_to_keep]

print("Shape after VarianceThreshold:", df_filtered.shape)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_filtered)

print("Scaled features shape:", scaled_features.shape)

pca = PCA(n_components = 0.95)
pca_features = pca.fit_transform(scaled_features)

print("Shape after PCA:", pca_features.shape)

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker = 'o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

kmeans_final = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
final_clusters = kmeans_final.fit_predict(pca_features)
df_filtered['Cluster'] = final_clusters
print("Cluster distribution:")
print(df_filtered['Cluster'].value_counts())

scaled_df = pd.DataFrame(scaled_features, columns=df_filtered.columns.drop('Cluster', errors='ignore'))
scaled_df['Cluster'] = final_clusters

# Calculate Silhouette Score for your clusters
sil_score = silhouette_score(pca_features, final_clusters)
print(f"Silhouette Score: {sil_score:.3f}")

cluster_summary = df_filtered.groupby('Cluster').mean(numeric_only=True)
print("Cluster Summary / Means:")
print(cluster_summary)
print("\nNumber of respondents per cluster:")
print(df_filtered['Cluster'].value_counts())

plt.figure(figsize=(9, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=final_clusters, cmap='viridis', alpha=0.7, edgecolor='k')
plt.title('K-Means Clusters of Mental Health Survey (PCA-Reduced Space)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Group')
plt.grid(True)
plt.show()




# 1. Calculate cluster means and overall dataset mean
cluster_means = df_filtered.groupby('Cluster').mean(numeric_only=True)
overall_mean = df_filtered.drop(columns=['Cluster']).mean(numeric_only=True)

# 2. Find the top 6 most varying features across clusters
feature_variance = cluster_means.var().sort_values(ascending=False)
top_features = feature_variance.head(6).index

# 3. Create subplots comparing cluster means against the overall mean
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
axes = axes.flatten()

cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']
colors = ['#008080', '#ffd700', '#800080']

for i, feature in enumerate(top_features):
    wrapped_title = "\n".join(textwrap.wrap(feature, width=45))
    values = cluster_means[feature].values
    mean_val = overall_mean[feature]
    
    ax = axes[i]
    
    # Plot horizontal bars for the clusters
    bars = ax.barh(cluster_labels, values, color=colors, edgecolor='k', height=0.5)
    
    # Add a vertical dashed red line representing the overall dataset mean for comparison
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Overall Mean ({mean_val:.2f})')
    
    ax.set_title(wrapped_title, fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('Mean Value', fontsize=9)
    
    # Dynamic scaling for safety
    max_v = max(max(values), mean_val)
    ax.set_xlim(0, max_v * 1.25 if max_v > 1 else 1.05)
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=8)

plt.subplots_adjust(hspace=0.7, wspace=0.3)
plt.suptitle('Cluster Means vs. Overall Dataset Baseline', fontsize=14, fontweight='bold', y=0.98)

# Save the updated chart
plt.savefig('cluster_means_comparison.png', dpi=300, bbox_inches='tight')
print("Updated chart with overall mean baseline saved as 'cluster_means_comparison.png'!")

plt.show()