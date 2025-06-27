"""
*Dataset Link:- https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
*Description:
This data set is created only for the learning purpose of the customer segmentation concepts , also known as market basket analysis . I will demonstrate this by using unsupervised ML technique (KMeans Clustering Algorithm) in the simplest form.
"""
# K-Means Clustering for Customer Segmentation
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# Set the aesthetics for visualizations
plt.style.use('dark_background')
colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#6A0572', '#AB83A1']
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.facecolor'] = '#222831'
plt.rcParams['figure.facecolor'] = '#222831'
plt.rcParams['text.color'] = '#EEEEEE'
plt.rcParams['axes.labelcolor'] = '#EEEEEE'
plt.rcParams['xtick.color'] = '#EEEEEE'
plt.rcParams['ytick.color'] = '#EEEEEE'


# Load the dataset
def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nChecking for missing values:")
    print(df.isnull().sum())

    return df


# Exploratory Data Analysis
def perform_eda(df):
    # Gender Distribution
    plt.figure(figsize=(10, 6))
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=colors[:2], textprops={'color': 'white'}, startangle=90)
    plt.title('Gender Distribution', fontsize=15)
    plt.tight_layout()
    #plt.savefig('gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Age Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Age'], bins=20, kde=True, color=colors[0])
    plt.title('Age Distribution', fontsize=15)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Annual Income Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color=colors[1])
    plt.title('Annual Income Distribution', fontsize=15)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('income_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Spending Score Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color=colors[2])
    plt.title('Spending Score Distribution', fontsize=15)
    plt.xlabel('Spending Score (1-100)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('spending_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Pairwise relationships
    plt.figure(figsize=(15, 10))
    sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', palette=colors[:2], plot_kws={'alpha': 0.7})
    plt.suptitle('Pairwise Relationships', fontsize=20, y=1.02)
    plt.tight_layout()
    #plt.savefig('pairwise_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={'shrink': .8}, vmin=-1, vmax=1)
    plt.title('Correlation Matrix', fontsize=15)
    plt.tight_layout()
    #plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("EDA completed and visualizations saved.")
    return df


# Preprocess data for clustering
def preprocess_data(df):
    # Select features for clustering
    X = df.drop(['CustomerID', 'Gender'], axis=1).values

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# Determine optimal number of clusters using Elbow Method
def find_optimal_clusters_elbow(X_scaled):
    wcss = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Visualize the Elbow Method
    plt.figure(figsize=(12, 6))
    plt.plot(K_range, wcss, marker='o', linestyle='-', color=colors[0], linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal k', fontsize=15)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()

    return wcss


# Determine optimal number of clusters using Silhouette Analysis
def find_optimal_clusters_silhouette(X_scaled):
    silhouette_scores = []
    K_range = range(2, 11)  # Silhouette score is not defined for k=1

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

    # Visualize the Silhouette Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color=colors[1], linewidth=2, markersize=8)
    plt.title('Silhouette Analysis for Optimal k', fontsize=15)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Find the k with maximum silhouette score
    optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k, silhouette_scores


# Perform K-means clustering with the optimal number of clusters
def perform_kmeans(X_scaled, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    return kmeans, cluster_labels, centroids


# Visualize the clusters
def visualize_clusters(df, cluster_labels, kmeans, optimal_k, scaler):
    df_with_clusters = df.copy()
    X_scaled, _ = preprocess_data(df_with_clusters)
    df_with_clusters['Cluster'] = cluster_labels

    # Get cluster centers in original scale
    original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # 2D Visualizations for pairs of features
    feature_pairs = [
        ('Annual Income (k$)', 'Spending Score (1-100)'),
        ('Age', 'Spending Score (1-100)'),
        ('Age', 'Annual Income (k$)')
    ]

    for feature1, feature2 in feature_pairs:
        plt.figure(figsize=(12, 8))
        for i in range(optimal_k):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
            plt.scatter(cluster_data[feature1], cluster_data[feature2],
                        c=colors[i % len(colors)], label=f'Cluster {i + 1}', alpha=0.7, s=80)

        # Plot the centroids
        centroid_idx = 0  # Index of centroid in original_centroids array
        if feature1 == 'Age':
            centroid_x_idx = 0
        else:  # Annual Income
            centroid_x_idx = 1

        if feature2 == 'Spending Score (1-100)':
            centroid_y_idx = 2
        else:  # Age or Annual Income
            centroid_y_idx = 0 if feature2 == 'Age' else 1

        for i in range(optimal_k):
            plt.scatter(original_centroids[i, centroid_x_idx], original_centroids[i, centroid_y_idx],
                        c='white', s=200, alpha=1, marker='X', edgecolors='black')

        plt.title(f'Clusters by {feature1} vs {feature2}', fontsize=15)
        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        #plt.savefig(f'clusters_{feature1.split()[0].lower()}_{feature2.split()[0].lower()}.png',
         #           dpi=300, bbox_inches='tight')
        plt.show()

    # 3D Visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        ax.scatter(cluster_data['Age'],
                   cluster_data['Annual Income (k$)'],
                   cluster_data['Spending Score (1-100)'],
                   c=colors[i % len(colors)], label=f'Cluster {i + 1}', alpha=0.7, s=60)

    # Plot the centroids
    for i in range(optimal_k):
        ax.scatter(original_centroids[i, 0], original_centroids[i, 1], original_centroids[i, 2],
                   c='white', s=200, alpha=1, marker='X', edgecolors='black')

    ax.set_title('3D Visualization of Clusters', fontsize=15)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Annual Income (k$)', fontsize=12)
    ax.set_zlabel('Spending Score (1-100)', fontsize=12)
    ax.legend()
    plt.tight_layout()
   # plt.savefig('3d_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

    # PCA for visualization if more than 3 features
    if X_scaled.shape[1] > 3:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        for i in range(optimal_k):
            plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1],
                        c=colors[i % len(colors)], label=f'Cluster {i + 1}', alpha=0.7, s=80)

        plt.title('Clusters after PCA', fontsize=15)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       # plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()

    return df_with_clusters


# Analyze and interpret clusters
def analyze_clusters(df_with_clusters, optimal_k):
    # Cluster profiles
    cluster_profiles = pd.DataFrame()

    for i in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        profile = {
            'Cluster Size': len(cluster_data),
            'Cluster %': len(cluster_data) / len(df_with_clusters) * 100,
            'Gender Ratio (F:M)': f"{len(cluster_data[cluster_data['Gender'] == 'Female'])}/{len(cluster_data[cluster_data['Gender'] == 'Male'])}",
            'Avg Age': cluster_data['Age'].mean(),
            'Avg Annual Income': cluster_data['Annual Income (k$)'].mean(),
            'Avg Spending Score': cluster_data['Spending Score (1-100)'].mean()
        }
        cluster_profiles[f'Cluster {i + 1}'] = pd.Series(profile)

    print("\nCluster Profiles:")
    print(cluster_profiles.round(2))

    # Visualize cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = [len(df_with_clusters[df_with_clusters['Cluster'] == i]) for i in range(optimal_k)]
    plt.pie(cluster_sizes, labels=[f'Cluster {i + 1}' for i in range(optimal_k)],
            autopct='%1.1f%%', startangle=90, colors=colors[:optimal_k], textprops={'color': 'white'})
    plt.title('Cluster Sizes', fontsize=15)
    plt.tight_layout()
   # plt.savefig('cluster_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Visualize cluster profiles
    metrics = ['Avg Age', 'Avg Annual Income', 'Avg Spending Score']
    cluster_means = pd.DataFrame()

    for i in range(optimal_k):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        cluster_means[f'Cluster {i + 1}'] = [
            cluster_data['Age'].mean(),
            cluster_data['Annual Income (k$)'].mean(),
            cluster_data['Spending Score (1-100)'].mean()
        ]

    cluster_means.index = metrics

    plt.figure(figsize=(14, 8))
    cluster_means.T.plot(kind='bar', ax=plt.gca(), color=colors[:len(metrics)])
    plt.title('Comparison of Cluster Characteristics', fontsize=15)
    plt.xlabel('Cluster')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title='Metrics')
    plt.tight_layout()
   # plt.savefig('cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Radar chart for cluster profiles (normalized)
    normalized_means = cluster_means.copy()
    for idx in normalized_means.index:
        max_val = normalized_means.loc[idx].max()
        min_val = normalized_means.loc[idx].min()
        normalized_means.loc[idx] = (normalized_means.loc[idx] - min_val) / (
                    max_val - min_val) if max_val > min_val else normalized_means.loc[idx]

    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor('#222831')

    for i in range(optimal_k):
        values = normalized_means[f'Cluster {i + 1}'].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i + 1}', color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.set_title('Cluster Profiles (Normalized)', fontsize=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
   # plt.savefig('radar_cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cluster_profiles


# Main function
def main() -> dict:
    print("Starting Customer Segmentation Analysis...")
    # Step 1: Load and explore data
    df = load_and_explore_data('archive/Mall_Customers.csv')
    # Step 2: Perform Exploratory Data Analysis
    df = perform_eda(df)
    # Step 3: Preprocess data
    X_scaled, scaler = preprocess_data(df)
    # Step 4: Find optimal number of clusters
    wcss = find_optimal_clusters_elbow(X_scaled)
    optimal_k, silhouette_scores = find_optimal_clusters_silhouette(X_scaled)
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
    # Step 5: Perform K-means clustering
    kmeans, cluster_labels, centroids = perform_kmeans(X_scaled, optimal_k)
    # Step 6: Visualize the clusters
    df_with_clusters = visualize_clusters(df, cluster_labels, kmeans, optimal_k, scaler)
    # Step 7: Analyze and interpret clusters
    cluster_profiles = analyze_clusters(df_with_clusters, optimal_k)
    print("\nCustomer Segmentation Analysis completed successfully!")
    print("All visualizations have been saved to the current directory.")
    # Return results for further use if needed
    return {
        'df': df,
        'df_with_clusters': df_with_clusters,
        'kmeans_model': kmeans,
        'optimal_k': optimal_k,
        'cluster_profiles': cluster_profiles
    }


if __name__ == "__main__":
    results = main()
