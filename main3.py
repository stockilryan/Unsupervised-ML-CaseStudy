import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import textwrap


def fill_NAN(filepath):
    """x"""
    df = pd.read_csv(filepath)
    print("Shape before", df.shape)
    print("Missing data points:", df.isnull().sum().sum())

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def one_hot_encoding(df):
    """xxx"""
    df = pd.get_dummies(df)
    print("Shape after filling missing values:", df.shape)
    print("Missing data points after:", df.isnull().sum().sum())
    return df

def varianceT_Sscaler(df):
    """x"""
    feature_selector = VarianceThreshold(threshold=0.01)
    feature_selector.fit(df)
    columns_selected = df.columns[feature_selector.get_support()]
    df_filtered = df[columns_selected]

    print("Shape after Variance Threshold:", df_filtered.shape)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_filtered)
    print("Shape after scaling features:", scaled_features.shape)
    return scaled_features, df_filtered

def p_c_a(scaled_features, n_components):
    """x"""
    pca = PCA(n_components)
    pca_features = pca.fit_transform(scaled_features)
    print("Shape after PCA:", pca_features.shape)
    return pca_features

def elbow_method(inertia, k_range, pca_features):
    """x"""
    for k in k_range:
        k_means = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        k_means.fit(pca_features)
        inertia.append(k_means.inertia_)

    plt.figure(figsize = (8, 5))
    plt.plot(k_range, inertia, marker = 'o', linestyle = '-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for k')
    plt.grid(True)
    ##plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    return k_means

def sil_sc(pca_features, clusters):
    """x"""
    sil_score = silhouette_score(pca_features, clusters)
    print(f"Silhouette score: {sil_score:.3f}")
    return sil_score

def k_means(pca_features, df_filtered, scaled_features):
    """x"""
    k_means_after = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
    clusters = k_means_after.fit_predict(pca_features)
    df_filtered['Cluster'] = clusters
    print("Cluster distribution:", df_filtered['Cluster'].value_counts())
    scaled_df = pd.DataFrame(scaled_features, columns = df_filtered.columns.drop('Cluster', errors = 'ignore'))
    scaled_df['Cluster'] = clusters

    plt.figure(figsize=(9, 6))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis', alpha=0.7, edgecolor='k')
    plt.title('K-Means Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Group')
    plt.grid(True)
    ##plt.savefig('kmeans_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    return df_filtered, clusters

def cluster_comparison(df_filtered):
    """x"""
    cluster_means = df_filtered.groupby('Cluster').mean(numeric_only = True)
    total_mean = df_filtered.drop(columns = ['Cluster']).mean(numeric_only = True)
    feature_variance = cluster_means.var().sort_values(ascending = False)
    top6_features = feature_variance.head(6).index
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (14, 14))
    axes = axes.flatten()
    cluster_lables = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    colours = ['#008080', '#ffd700', '#800080']

    for i, feature in enumerate(top6_features):
        wrapped_title = "\n".join(textwrap.wrap(feature, width=45))
        values = cluster_means[feature].values
        mean_value = total_mean[feature]
        ax = axes[i]
        bars = ax.barh(cluster_lables, values, color = colours, edgecolor = 'k', height = 0.5)
        ax.axvline(mean_value, color = 'red', linestyle = '--', linewidth = 2, label = f'Overall Mean ({mean_value:.2f})')
        ax.set_title(wrapped_title, fontsize = 9, fontweight = 'bold', pad = 12)
        ax.set_xlabel('Mean Value', fontsize = 9)
        max_v = max(max(values), mean_value)
        ax.set_xlim(0, max_v * 1.25 if max_v > 1 else 1.05)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.7)
        ax.legend(loc = 'lower right', fontsize = 8)
        plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
        plt.suptitle('Cluster Means vs. Overall Baseline', fontsize = 14, fontweight = 'bold', y = 0.98)
    ##plt.savefig("cluster_means_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    return fig

if __name__ == "__main__":
    filepath = "survey.csv"
    n_components = 0.95
    inertia = []
    k_range = range(1, 11)

    df_clean = fill_NAN(filepath)
    df_encode = one_hot_encoding(df_clean)
    scaled_features, df_filtered = varianceT_Sscaler(df_encode)
    pca_features = p_c_a(scaled_features, n_components)
    elbow_method(inertia, k_range, pca_features)   
    df_filtered, clusters = k_means(pca_features, df_filtered, scaled_features)
    sil_sc(pca_features, clusters)
    cluster_comparison(df_filtered)