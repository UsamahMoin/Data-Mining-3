# Data Mining Project 3

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import os

def setup_centroids(dataset, num_clusters, random_seed=0):
    np.random.seed(random_seed)
    centroids = [dataset[np.random.randint(dataset.shape[0])]]
    while len(centroids) < num_clusters:
        distances = np.array([min(np.linalg.norm(data_point - centroid) ** 2 for centroid in centroids) for data_point in dataset])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(dataset[j])
                break
    return np.array(centroids)

def compute_error(cluster_groups, centroids_list):
    total_error = 0.0
    for cluster, centroid in zip(cluster_groups, centroids_list):
        total_error += np.sum(np.linalg.norm(cluster - centroid, axis=1))
    return total_error

def assign_clusters(dataset, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for data_point in dataset:
        distances = [np.linalg.norm(data_point - centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(data_point)
    return clusters

def execute_k_means(dataset, num_clusters, iterations=20):
    centroids = setup_centroids(dataset, num_clusters)
    for _ in range(iterations):
        clusters = assign_clusters(dataset, centroids)
        updated_centroids = np.array([np.mean(clust, axis=0) if len(clust) > 0 else centroids[i] for i, clust in enumerate(clusters)])
        
        if np.all(centroids == updated_centroids):
            break
        centroids = updated_centroids

    error = compute_error([np.array(cluster) for cluster in clusters], centroids)
    return error, centroids

def display_error_graph(cluster_counts, error_values, graph_title):
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, error_values, marker='o')
    plt.title(graph_title)
    plt.xlabel('Clusters Count')
    plt.ylabel('Total Error')
    plt.xticks(cluster_counts)
    plt.grid(True)
    plt.show()

def process_dataset(dataset_filepath):
    dataset = pd.read_csv(dataset_filepath, delim_whitespace=True, header=None)
    dataset = dataset.iloc[:, :-1].values

    cluster_range = range(2, 11)
    errors = []

    for cluster_num in cluster_range:
        error, _ = execute_k_means(dataset, cluster_num)
        errors.append(error)
        print(f'For k = {cluster_num} After 20 iterations: Error = {error:.4f}')

    display_error_graph(cluster_range, errors, 'K-Means Clustering Error Analysis')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py UCI_datasets/<dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    if not os.path.isfile(dataset_path):
        print("Dataset file not found.")
        sys.exit(1)
    
    process_dataset(dataset_path)
