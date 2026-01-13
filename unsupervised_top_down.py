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
from unsupervised_kmeans import Kmeans, read_and_prepare_data

# Top-Down Clustering
class TopDownClustering:
    def __init__(self, num_clusters = 10, max_iters = 100):
        self.n_clusters = num_clusters
        self.max_iters = max_iters
        self.clusters = []
        self.data = []

    # Simulate
    def simulate(self, data):
        self.data = data
        # Start With Root Cluster With All Indexes
        queue = [list(range(len(data)))]
        final_clusters = []

        # While There Is Queue, And Final Clusters And Queue Cluster Is Less Than Number Of Clusters
        while queue and len(final_clusters) + len(queue) < self.n_clusters:
            # Get First Cluster In Queue
            cluster = queue.pop(0)

            # If Cluster Has Less Than 2 Data, Too Small To Split And It Is In Final Clusters Stage. Skip To Next Loop.
            if len(cluster) < 2:
                final_clusters.append(cluster)
                continue

            # Run KMeans To Split Cluster Into 2
            subset = [data[i] for i in cluster]
            kmeans = Kmeans(K=2, max_iters=self.max_iters)
            kmeans.simulate(subset)

            # Map Back To Cluster's Indexes And Append Sub-Clusters
            for sub_cluster in kmeans.clusters:
                indexes = [cluster[i] for i in sub_cluster]
                if len(indexes) > 0:
                    queue.append(indexes)

        # Add remaining clusters from queue
        final_clusters.extend(queue)
        self.clusters = final_clusters
        return self

# Main
def main():
    # Read CSV Data And Drop Label Column
    raw_features = pd.read_csv('preprocessed_extracted_features.csv')
    labels = raw_features['Label'].tolist()
    data = read_and_prepare_data(raw_features)

    # Run top-down clustering
    topDown = TopDownClustering(num_clusters=10)
    topDown.simulate(data)

    # Print results
    print(f"Number of clusters: {len(topDown.clusters)}")
    cluster_analysis_data = []
    for cluster in topDown.clusters:
        cluster_info = {}
        label_counts = {}
        for i in cluster:
            label = labels[i]
            if label in label_counts:
                label_counts[label] = label_counts[label] + 1
            else:
                label_counts[label] = 1
        print(sorted(label_counts.items(), key=lambda x: x[1]))
        cluster_info['size'] = len(cluster)
        max_label, max_count = max(label_counts.items(), key=lambda x: x[1])
        cluster_info['max_label'] = max_label
        cluster_info['max_count'] = max_count
        cluster_info['share'] = round((max_count/len(cluster)) * 100, 2)
        cluster_analysis_data.append(cluster_info)

    # Print Cluster Information
    print("Cluster Analysis")
    print(f"{'Cluster':<10}{'Size':<10}{'Max Label':<12}{'Max Count':<10}{'Share (%)':<10}")
    print("-" * 52)
    for index, info in enumerate(cluster_analysis_data):
        print(f"{index:<10}{info['size']:<10}{info['max_label']:<12}{info['max_count']:<10}{info['share']:<10}")

if __name__ == "__main__":
    main()