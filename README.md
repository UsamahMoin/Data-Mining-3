# K-Means Clustering Project

This project implements the K-Means clustering algorithm from scratch in Python. The code is designed to process a dataset, perform clustering, and visualize the clustering error for different numbers of clusters.

## Prerequisites

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Installation

To install the required packages, you can use pip:

pip install numpy pandas matplotlib


## Functions

### `setup_centroids(dataset, num_clusters, random_seed=0)`
Initializes centroids for K-Means using the k-means++ algorithm.

### `compute_error(cluster_groups, centroids_list)`
Computes the total error for the given clusters and centroids.

### `assign_clusters(dataset, centroids)`
Assigns each data point in the dataset to the nearest centroid.

### `execute_k_means(dataset, num_clusters, iterations=20)`
Performs the K-Means clustering algorithm and returns the final error and centroids.

### `display_error_graph(cluster_counts, error_values, graph_title)`
Displays a graph of the total error for different numbers of clusters.

### `process_dataset(dataset_filepath)`
Processes the dataset, performs K-Means clustering for a range of cluster counts, and displays the error graph.

## Main Execution

The script is designed to be run from the command line. It expects a single argument: the path to the dataset file. The dataset file should be in a format compatible with Pandas `read_csv` function with whitespace as the delimiter and no header.
