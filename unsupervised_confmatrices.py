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
from sklearn.metrics import confusion_matrix
from collections import Counter

from unsupervised_kmeans import Kmeans, read_and_prepare_data
from unsupervised_bottom_up import BottomUpClustering
from unsupervised_top_down import TopDownClustering

def get_predicted_labels(clusters, actual_labels):
    predictions = [None] * len(actual_labels)
    for cluster in clusters:
        if not cluster: continue
        cluster_actuals = [actual_labels[i] for i in cluster]
        majority_label = Counter(cluster_actuals).most_common(1)[0][0]
        for i in cluster:
            predictions[i] = majority_label
    return predictions

def plot_confusion_matrix(ax, actual, predicted, title, labels):
    cm = confusion_matrix(actual, predicted, labels=labels)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", 
                    color="white" if cm[i, j] > thresh else "black")

if __name__ == "__main__":
    # Load Data
    raw_df = pd.read_csv('preprocessed_extracted_features.csv')
    actual_labels = raw_df['Label'].tolist()
    unique_labels = sorted(list(set(actual_labels)))
    data = read_and_prepare_data(raw_df)

    # Run All 3 Algorithms using the imported classes
    print("Running K-Means...")
    km = Kmeans(K=10)
    km.simulate(data)
    km_preds = get_predicted_labels(km.clusters, actual_labels)

    print("Running Bottom-Up...")
    bu = BottomUpClustering(num_clusters=10)
    bu.simulate(data)
    bu_preds = get_predicted_labels(bu.clusters, actual_labels)

    print("Running Top-Down...")
    td = TopDownClustering(num_clusters=10) 
    td.simulate(data)
    td_preds = get_predicted_labels(td.clusters, actual_labels)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    plot_confusion_matrix(axes[0], actual_labels, km_preds, "K-Means (k_means.py)", unique_labels)
    plot_confusion_matrix(axes[1], actual_labels, bu_preds, "Bottom-Up (unsupervised_bottom_up.py)", unique_labels)
    plot_confusion_matrix(axes[2], actual_labels, td_preds, "Top-Down (unsupervised_top_down.py)", unique_labels)

    plt.tight_layout()
    plt.show()