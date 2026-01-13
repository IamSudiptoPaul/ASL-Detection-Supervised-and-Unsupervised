# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Assignment On Supervised And Unsupervised Learning Project
Coursework 002 for: CMP-7058A Artificial Intelligence

Feature Extraction

@author: C102 ( 100525654 , 100525448, 100538928 )
@date:   11/1/2026

"""

from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from unsupervised_kmeans import Kmeans, read_and_prepare_data
from unsupervised_top_down import TopDownClustering
from unsupervised_bottom_up import BottomUpClustering
import matplotlib.pyplot as plt

def clusters_to_labels(clusters, data_size):
    labels = [0] * data_size
    for cluster_idx, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = cluster_idx
    return labels

# Main Function
def main():
    raw_features = pd.read_csv('preprocessed_extracted_features.csv')
    data = read_and_prepare_data(raw_features)

    # Scores For Graphing
    kmeans_score_list = []
    topdown_score_list = []
    bottomup_score_list = []

    # Run 10 Times
    for i in range(10):
        print("Iternation: ", i)

        # For K-means
        kmean = Kmeans(10, 100)
        kmean.simulate(data)
        kmeans_labels = clusters_to_labels(kmean.clusters, len(data))
        kmeans_score = round(silhouette_score(data, kmeans_labels), 4)
        print(f"K-means Silhouette Score: {kmeans_score}")
        kmeans_score_list.append(kmeans_score)

        # For Top-Down
        topDown = TopDownClustering(num_clusters=10)
        topDown.simulate(data)
        topdown_labels = clusters_to_labels(topDown.clusters, len(data))
        topdown_score =  round(silhouette_score(data, topdown_labels), 4)
        print(f"Top-Down Silhouette Score: {topdown_score}")
        topdown_score_list.append(topdown_score)

        # For Bottom-Up
        bottomUp = BottomUpClustering(num_clusters=10)
        bottomUp.simulate(data)
        bottomup_labels = clusters_to_labels(bottomUp.clusters, len(data))
        bottomup_score =  round(silhouette_score(data, bottomup_labels), 4)
        print(f"Bottom-Up Silhouette Score: {bottomup_score}")
        bottomup_score_list.append(bottomup_score)

    # Plot Graph
    iterations = range(1, 11)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, kmeans_score_list, marker='o', label='K-means')
    plt.plot(iterations, topdown_score_list, marker='s', label='Top-Down')
    plt.plot(iterations, bottomup_score_list, marker='^', label='Bottom-Up')
    plt.xlabel('Iteration')
    plt.ylabel('Silhouette Scores')
    plt.title('Silhouette Score Comparison Betweek KMeans, TopDown, BottomUp')
    plt.legend()
    plt.grid(True)
    plt.xticks(iterations)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()