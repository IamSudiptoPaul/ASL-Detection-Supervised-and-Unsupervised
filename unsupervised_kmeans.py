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
import random
import matplotlib.pyplot as plt

# K-means Class
class Kmeans:
    # Initialization
    def __init__(self, K, max_iters = 100):
        self.max_iters = max_iters
        self.K = K
        self.centroids = []
        self.clusters = []
        self.data = []

    # Initialization: Get Random centroids For K clusters. 
    def randomize_centroids(self):
        self.centroids = random.sample(self.data, self.K)

    # Assign Clusters
    def assign_clusters(self):
        clusters = [[] for i in range(self.K)]
        for index, coordinates in enumerate(self.data):
            distances = [e_distance(coordinates, c) for c in self.centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(index)
        self.clusters = clusters

    # Update centroids 
    def update_centroids(self):
        new_centroids = []
        for cluster in self.clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(self.data))
            else:
                num_coordinates = len(self.data[0])
                new_centroid = []
                for i in range(num_coordinates):
                    total = 0
                    for j in cluster:
                        total += self.data[j][i]
                    mean = total/len(cluster)
                    new_centroid.append(mean)
                new_centroids.append(new_centroid)
        self.centroids = new_centroids

    # Simulate
    def simulate(self, data):
        self.data = data
        self.randomize_centroids()
        for i in range(self.max_iters):
            old_centroids = [c[:] for c in self.centroids]
            self.assign_clusters()
            self.update_centroids()

            if old_centroids == self.centroids:
                print("Centroids Doesn't Change. Clustering Is Set At Iteration: ", i)
                break

# Euclidean Distance - Formula From Slide Week 11
def e_distance(x, y):
    n = len(x)
    sum = 0
    for i in range(n):
        sum += (x[i]-y[i]) ** 2
    return math.sqrt(sum)

# Flatten The 21 Landmarks Of (x,y,z) And Have 63 Coordinates
def row_to_vector(row):
    vector = []
    for landmark in row:
        coordinates = landmark.split(',')
        for c in coordinates:
            vector.append(float(c))
    return vector

def read_and_prepare_data(raw_features):
    # Read CSV data and drop Label column   
    data = raw_features.drop(columns=['Label'])
    data = data.apply(row_to_vector, axis=1).tolist()
    return data

# Main Function
def main():
    # Define K Clusters Number
    K = 10

    # Read CSV data and drop Label column
    raw_features = pd.read_csv('preprocessed_extracted_features.csv')
    labels = raw_features['Label'].tolist()
    raw_features = read_and_prepare_data(raw_features)

    # Initialize And Simulate Kmeans
    kmean = Kmeans(10, 100)
    kmean.simulate(raw_features)

    # Print Predicted Cluster Information
    for cluster in kmean.clusters:
        label_counts = {}
        for i in cluster:
            label = labels[i]
            if label in label_counts:
                label_counts[label] = label_counts[label] + 1
            else:
                label_counts[label] = 1
        print(sorted(label_counts.items(), key=lambda x: x[1]))

if __name__ == "__main__":
    main()