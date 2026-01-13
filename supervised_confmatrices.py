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

import supervised_knn
import supervised_decisiontree
import supervised_GaussianNaiveBayes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_supervised_comparison(results):
    """
    Generates an accuracy bar chart and side-by-side confusion matrices.
    """
    names = list(results.keys())
    accuracies = [results[name][0] for name in names]

    # 1. Bar Chart for Accuracy Comparison
    plt.figure(figsize=(10, 5))
    colors = ["#00f7fb", '#f28e2b', '#e15759']
    bars = plt.bar(names, accuracies, color=colors)
    plt.title("Classifier Accuracy Comparison", fontsize=14)
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.1)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

    # 2. Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, name in enumerate(names):
        acc, cm, labels = results[name]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=labels, yticklabels=labels, cbar=False)
        
        axes[i].set_title(f"{name}\n(Accuracy: {acc:.4f})", fontsize=12)
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig('supervised_comparison_plot.png')
    plt.show()

def main():
    results = {}

    # Gather data from your modules
    print("Evaluating kNN...")
    acc_knn, cm_knn, labels_knn = supervised_knn.evaluate()
    results["kNN"] = (acc_knn, cm_knn, labels_knn)

    print("Evaluating Decision Tree...")
    acc_dt, cm_dt, labels_dt = supervised_decisiontree.evaluate()
    results["DecisionTree"] = (acc_dt, cm_dt, labels_dt)

    print("Evaluating Gaussian NB...")
    acc_nb, cm_nb, labels_nb = supervised_GaussianNaiveBayes.evaluate()
    results["GaussianNB"] = (acc_nb, cm_nb, labels_nb)

    # Print Summary Table
    print("\n" + "="*40)
    print(f"{'Classifier':<15} {'Accuracy':<10}")
    print("-" * 40)
    # Sort results by accuracy (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for name, (acc, _, _) in sorted_results:
        print(f"{name:<15} {acc:.4f}")
    print("="*40)

    # Generate Visual Plots
    plot_supervised_comparison(results)

if __name__ == "__main__":
    main()