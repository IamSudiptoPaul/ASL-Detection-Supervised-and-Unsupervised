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

import pandas as pd
import math
import numpy as np
from unsupervised_kmeans import Kmeans, read_and_prepare_data, e_distance

class BottomUpClustering:
    def __init__(self, num_clusters=10, max_iters=100):
        self.n_clusters = num_clusters
        self.max_iters = max_iters
        self.clusters = []
        self.data = []

    def simulate(self, data):
        self.data = np.array(data)
        n_samples = len(self.data)
        self.clusters = [[i] for i in range(n_samples)]
        
        # 1. Initial Distance Matrix
        print(f"Calculating distance matrix...")
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            diff = self.data[i] - self.data[i+1:]
            dist_matrix[i, i+1:] = np.sqrt(np.sum(diff**2, axis=1))
            dist_matrix[i+1:, i] = dist_matrix[i, i+1:]
            
        np.fill_diagonal(dist_matrix, float('inf'))

        iteration = 0
        # Determine how many merges we actually need to reach n_clusters
        total_merges_needed = n_samples - self.n_clusters
        
        # We use a while loop that stops at n_clusters OR max_iters
        while len([c for c in self.clusters if len(c) > 0]) > self.n_clusters and iteration < self.max_iters:
            # Find closest pair
            idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            
            size1 = len(self.clusters[idx1])
            size2 = len(self.clusters[idx2])
            
            # The new distance to any cluster 'k' is the average of distances from idx1 and idx2
            # (size1 * d1 + size2 * d2) / (size1 + size2)
            for k in range(len(self.clusters)):
                if len(self.clusters[k]) > 0 and k != idx1 and k != idx2:
                    avg_dist = (size1 * dist_matrix[idx1, k] + size2 * dist_matrix[idx2, k]) / (size1 + size2)
                    dist_matrix[idx1, k] = dist_matrix[k, idx1] = avg_dist

            # Merge
            self.clusters[idx1] = self.clusters[idx1] + self.clusters[idx2]
            self.clusters[idx2] = [] 
            
            # Remove idx2 from consideration
            dist_matrix[idx2, :] = float('inf')
            dist_matrix[:, idx2] = float('inf')
            dist_matrix[idx1, idx1] = float('inf')
            
            iteration += 1
            if iteration % 500 == 0:
                active = len([c for c in self.clusters if len(c) > 0])
                print(f"Iteration {iteration}: {active} clusters remaining")
        
        self.clusters = [c for c in self.clusters if len(c) > 0]
        print(f"Bottom-Up Clustering Complete At Iteration: {iteration}")
        return self

def main():
    # Load Data
    raw_features = pd.read_csv('preprocessed_extracted_features.csv')
    labels = raw_features['Label'].tolist()
    data = read_and_prepare_data(raw_features)

    # Run Bottom-Up
    bottomUp = BottomUpClustering(num_clusters=10, max_iters=len(data)-10)
    bottomUp.simulate(data)

    # Output Format matching your example
    print(f"Number of clusters: {len(bottomUp.clusters)}")
    cluster_analysis_data = []
    
    for cluster in bottomUp.clusters:
        label_counts = {}
        for i in cluster:
            label = labels[i]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Print list of tuples
        print(sorted(label_counts.items(), key=lambda x: x[1]))
        
        # Collect for table
        size = len(cluster)
        max_label, max_count = max(label_counts.items(), key=lambda x: x[1])
        cluster_analysis_data.append({
            'size': size,
            'max_label': max_label,
            'max_count': max_count,
            'share': round((max_count / size) * 100, 2)
        })

    # Print Table
    print("\nCluster Analysis")
    print(f"{'Cluster':<10}{'Size':<10}{'Max Label':<12}{'Max Count':<10}{'Share (%)':<10}")
    print("-" * 52)
    for index, info in enumerate(cluster_analysis_data):
        print(f"{index:<10}{info['size']:<10}{info['max_label']:<12}{info['max_count']:<10}{info['share']:<10}")

if __name__ == "__main__":
    main()