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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, random
from collections import Counter, defaultdict
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, f1_score

# CORE UTILITIES
def e_distance(x, y):
    return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    labels = df['Label'].tolist()
    # Flatten landmarks: each row becomes a 63-element vector
    data = df.drop(columns=['Label']).apply(
        lambda row: [float(c) for lm in row for c in str(lm).split(',')], axis=1
    ).tolist()
    return np.array(data), labels, sorted(list(set(labels)))

# MODELS
class KmeansModel:
    def __init__(self, K=10, max_iters=50):
        self.K, self.max_iters = K, max_iters
        self.clusters, self.centroids = [], []

    def simulate(self, data):
        self.centroids = random.sample(data.tolist(), self.K)
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.K)]
            for idx, p in enumerate(data):
                dists = [e_distance(p, c) for c in self.centroids]
                clusters[dists.index(min(dists))].append(idx)
            
            old = [c[:] for c in self.centroids]
            self.centroids = [np.mean([data[i] for i in cls], axis=0).tolist() if cls else random.choice(data).tolist() for cls in clusters]
            self.clusters = clusters
            if old == self.centroids: break
        return self

def knn_predict(X_test, X_train, y_train, k=10):
    preds = []
    for test_p in X_test:
        dists = sorted([(e_distance(test_p, X_train[i]), y_train[i]) for i in range(len(X_train))])[:k]
        scores = defaultdict(float)
        for d, l in dists: scores[l] += 1.0 / (d + 1e-9)
        preds.append(max(scores, key=scores.get))
    return preds

# FINAL EVALUATION
def run_final_comparison(data, labels, unique_labels):
    # k-NN Logic (Supervised)
    split = int(len(data) * 0.8)
    indices = np.random.permutation(len(data))
    X_train, X_test = data[indices[:split]], data[indices[split:]]
    y_train, y_test = [labels[i] for i in indices[:split]], [labels[i] for i in indices[split:]]
    
    print("Simulating k-NN...")
    y_pred = knn_predict(X_test, X_train, y_train)
    knn_acc = sum(1 for i in range(len(y_test)) if y_test[i] == y_pred[i]) / len(y_test)
    knn_f1 = f1_score(y_test, y_pred, average='macro')

    # K-means Logic (Unsupervised)
    print("Simulating K-means...")
    km = KmeansModel(K=10).simulate(data)
    
    flat_clusters = np.zeros(len(data))
    for c_id, idxs in enumerate(km.clusters):
        for i in idxs: flat_clusters[i] = c_id
    
    purity = np.mean([Counter([labels[i] for i in cls]).most_common(1)[0][1]/len(cls) for cls in km.clusters if cls])
    sil = silhouette_score(data, flat_clusters)
    ari = adjusted_rand_score(labels, flat_clusters)

    # FINAL TERMINAL OUTPUT
    print("\n" + "="*60)
    print(f"{'FINAL MODEL COMPARISON SUMMARY':^60}")
    print("="*60)
    print(f"{'Metric':<30} | {'k-NN (Sup)':<12} | {'K-means (Un)':<12}")
    print("-" * 60)
    print(f"{'Primary (Accuracy/Purity)':<30} | {knn_acc:.4f}     | {purity:.4f}")
    print(f"{'F1-Score (Macro Average)':<30} | {knn_f1:.4f}     | {'N/A':<12}")
    print(f"{'Silhouette (Separation)':<30} | {'N/A':<12} | {sil:.4f}")
    print(f"{'ARI (Label Agreement)':<30} | {'N/A':<12} | {ari:.4f}")
    print("="*60)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # k-NN Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=unique_labels), annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=unique_labels, yticklabels=unique_labels)
    axes[0].set_title(f"k-NN: Acc {knn_acc:.2f} | F1 {knn_f1:.2f}")
    
    # K-means Cluster Heatmap
    cluster_dist = [[Counter([labels[i] for i in cls]).get(l, 0) for l in unique_labels] for cls in km.clusters]
    sns.heatmap(cluster_dist, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=unique_labels, yticklabels=[f"C{i}" for i in range(10)])
    axes[1].set_title(f"K-means: Purity {purity:.2f} | Sil {sil:.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data, labels, unique_labels = prepare_data('preprocessed_extracted_features.csv')
    run_final_comparison(data, labels, unique_labels)