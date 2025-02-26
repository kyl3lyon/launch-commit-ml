# -- Imports --
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples

import umap.umap_ as umap
import seaborn as sns
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# -- Functions --
def scale_data(data: pd.DataFrame, predictor_columns: List[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = data.copy()
    scaled_data[predictor_columns] = scaler.fit_transform(data[predictor_columns])
    return scaled_data

def find_optimal_clusters(data: pd.DataFrame, max_clusters: int = 10) -> None:
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def perform_clustering(data: pd.DataFrame, n_clusters: int = None, method: str = 'kmeans', **kwargs) -> pd.DataFrame:
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif method == 'dbscan':
        clusterer = DBSCAN(**kwargs)
    elif method == 'optics':
        clusterer = OPTICS(**kwargs)
    elif method == 'hdbscan':
        clusterer = HDBSCAN(**kwargs)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42, **kwargs)
        embedding = reducer.fit_transform(data)
        clusterer = HDBSCAN(min_cluster_size=15, **kwargs)
        data['cluster'] = clusterer.fit_predict(embedding)
        return data
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans', 'hierarchical', 'dbscan', 'optics', 'hdbscan', or 'umap'.")
    
    data['cluster'] = clusterer.fit_predict(data)
    return data

def visualize_clusters(data: pd.DataFrame, predictor_columns: List[str], cluster_column: str = 'cluster', target: pd.Series = None):
    import plotly.express as px
    from plotly.offline import plot

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data[predictor_columns])
    
    # Add small random jitter to reduce perfect overlaps
    jitter_amount = 0.01
    pca_result += np.random.normal(0, jitter_amount, pca_result.shape)
    
    # Create a DataFrame with PCA results
    df_plot = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
    df_plot['Cluster'] = data[cluster_column]
    
    # Add predictor columns to df_plot for hover data
    for col in predictor_columns:
        df_plot[col] = data[col]
    
    if target is not None:
        df_plot['Target'] = target
        df_plot['Target_Label'] = df_plot['Target'].map({0: 'All Clear', 1: 'Delay', 2: 'Scrub'})
        
        color_map = {'All Clear': 'green', 'Delay': 'orange', 'Scrub': 'red'}
        symbol_map = {'All Clear': 'circle', 'Delay': 'diamond', 'Scrub': 'x'}
        
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            color='Target_Label', symbol='Cluster',
                            color_discrete_map=color_map,
                            symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x'],
                            hover_data=predictor_columns,
                            labels={'Cluster': 'Cluster', 'Target_Label': 'Target Class'},
                            title='Cluster and Class Visualization using PCA (3D)')
        
        fig.update_traces(marker=dict(size=3))
    else:
        fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
                            color='Cluster',
                            hover_data=predictor_columns,
                            labels={'Cluster': 'Cluster'},
                            title='Cluster Visualization using PCA (3D)')
        
        fig.update_traces(marker=dict(size=3))
    
    fig.update_layout(scene=dict(xaxis_title='First Principal Component',
                                 yaxis_title='Second Principal Component',
                                 zaxis_title='Third Principal Component'),
                      legend_title_text='Legend',
                      legend=dict(x=1.05, y=0.5))  # Move legend outside of plot area
    
    # Save the plot as an HTML file
    plot(fig, filename='cluster_visualization.html', auto_open=False)
    print("Interactive plot saved as 'cluster_visualization.html'. Please open this file in a web browser to view.")

def generate_dendrogram(data: pd.DataFrame, predictor_columns: List[str], max_leaf_nodes: int = 30) -> None:
    linkage_matrix = linkage(data[predictor_columns], method='ward')
    
    plt.figure(figsize=(20, 12))
    dendrogram(linkage_matrix, 
               truncate_mode='lastp',  # Show only the last p merged clusters
               p=max_leaf_nodes,       # Show up to max_leaf_nodes leaf nodes
               leaf_font_size=10,
               leaf_rotation=90,
               show_contracted=True)   # Show contracted nodes as ellipsis
    plt.title('Hierarchical Clustering Dendrogram (Truncated)', fontsize=16)
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def analyze_cluster_characteristics(data: pd.DataFrame, cluster_column: str = 'cluster') -> None:
    cluster_stats = data.groupby(cluster_column).agg(['mean', 'std'])
    print("Cluster Characteristics:")
    print(cluster_stats)
    
    # Optionally, you can return the cluster_stats DataFrame
    return cluster_stats

def augment_training_data(X_train: pd.DataFrame, 
                          y_train: pd.Series, 
                          all_weather_data: pd.DataFrame, 
                          predictor_columns: List[str],
                          target_clusters: List[int],
                          n_clusters: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Augments the training dataset by sampling data points from specified clusters 
    in the all_weather_data DataFrame.

    Args:
        X_train (pd.DataFrame): The original training features.
        y_train (pd.Series): The original training labels.
        all_weather_data (pd.DataFrame): The unlabeled weather data for sampling.
        predictor_columns (List[str]): List of predictor columns used for clustering.
        target_clusters (List[int]):  List of cluster IDs to sample from.
        n_clusters (int): The number of clusters used in the clustering process.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The augmented training features and labels.
    """
    # Select only the predictor columns from all_weather_data
    # to ensure consistency with the training data features.
    all_weather_data_subset = all_weather_data[predictor_columns]

    # Drop rows with NaN values
    all_weather_data_subset = all_weather_data_subset.dropna()

    # Scale all_weather_data using the SAME scaler fitted on X_train
    all_weather_scaled = scale_data(all_weather_data_subset, predictor_columns)

    # Fit KMeans model using X_train_scaled and n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scale_data(X_train[predictor_columns], predictor_columns))

    # Predict cluster labels for all_weather_scaled
    all_weather_data_subset['cluster'] = kmeans.predict(all_weather_scaled)

    augmented_data = []
    for cluster_id in target_clusters:
        cluster_data = all_weather_data_subset[all_weather_data_subset['cluster'] == cluster_id]
        sample_size = len(X_train[X_train['cluster'] == cluster_id])  # Match original size
        cluster_sample = cluster_data.sample(n=sample_size, random_state=42)
        augmented_data.append(cluster_sample)

    # Simplified concatenation
    X_train_augmented = pd.concat([X_train] + augmented_data, ignore_index=True)
    y_train_augmented = pd.concat([y_train, pd.Series([1] * len(pd.concat(augmented_data)))], ignore_index=True)

    return X_train_augmented, y_train_augmented



# -- New Stuff --
def visualize_clusters_umap(
    data: pd.DataFrame,
    predictor_columns: List[str],
    cluster_column: str = 'cluster',
    target: pd.Series = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean'
):
    """
    Visualizes clusters and target classes using UMAP for dimensionality reduction.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    embedding = reducer.fit_transform(data[predictor_columns])

    plt.figure(figsize=(14, 10))
    
    if target is not None:
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Cluster': data[cluster_column],
            'Target': target
        })
        # Use seaborn for better visualization
        sns.scatterplot(
            data=plot_df,
            x='UMAP1',
            y='UMAP2',
            hue='Target',
            style='Cluster',
            palette=['green', 'yellow', 'red'],
            s=70,
            alpha=0.8,
            edgecolor='k'
        )
        
        # Add tighter circles around clusters
        for cluster in plot_df['Cluster'].unique():
            cluster_points = plot_df[plot_df['Cluster'] == cluster]
            center = cluster_points[['UMAP1', 'UMAP2']].mean()
            radius = np.percentile(np.sqrt(np.sum((cluster_points[['UMAP1', 'UMAP2']] - center)**2, axis=1)), 90)
            circle = plt.Circle((center['UMAP1'], center['UMAP2']), radius, fill=False, linestyle='--', color='black')
            plt.gca().add_artist(circle)
        
        plt.title('UMAP Projection of Clusters and Target Classes')
        plt.legend(title='Target Classes', loc='best')
    else:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=data[cluster_column], cmap='Spectral', s=50)
        plt.colorbar(scatter, label='Cluster')
        
        # Add tighter circles around clusters
        for cluster in np.unique(data[cluster_column]):
            cluster_points = embedding[data[cluster_column] == cluster]
            center = np.mean(cluster_points, axis=0)
            radius = np.percentile(np.sqrt(np.sum((cluster_points - center)**2, axis=1)), 90)
            circle = plt.Circle((center[0], center[1]), radius, fill=False, linestyle='--', color='black')
            plt.gca().add_artist(circle)
        
        plt.title('UMAP Projection of Clusters')
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.show()

def compute_cluster_centroids(data: pd.DataFrame, cluster_column: str, predictor_columns: List[str]) -> pd.DataFrame:
    centroids = data.groupby(cluster_column)[predictor_columns].mean()
    return centroids

def plot_feature_distributions(data: pd.DataFrame, predictor_columns: List[str], cluster_column: str):
    for feature in predictor_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_column, y=feature, data=data)
        plt.title(f'Distribution of {feature} Across Clusters')
        plt.show()

def cluster_target_crosstab(data: pd.DataFrame, cluster_column: str, target: pd.Series):
    """
    Plots the proportion of target classes within each cluster.
    """
    crosstab = pd.crosstab(data[cluster_column], target, normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Proportion of Target Classes within Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Target Class')
    plt.show()
    return crosstab

def silhouette_analysis(data: pd.DataFrame, predictor_columns: List[str], clusters: np.ndarray):
    silhouette_avg = silhouette_score(data[predictor_columns], clusters)
    print(f"Average Silhouette Score: {silhouette_avg:.3f}")

    sample_silhouette_values = silhouette_samples(data[predictor_columns], clusters)

    plt.figure(figsize=(10, 6))
    y_lower = 10
    for i in range(len(np.unique(clusters))):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(np.unique(clusters)))
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    plt.title('Silhouette Plot for Various Clusters')
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()

def cluster_target_confusion_matrix(data: pd.DataFrame, cluster_column: str, target: pd.Series):
    cm = confusion_matrix(target, data[cluster_column])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Cluster')
    plt.ylabel('Target Class')
    plt.title('Confusion Matrix between Clusters and Target Classes')
    plt.show()
    return cm