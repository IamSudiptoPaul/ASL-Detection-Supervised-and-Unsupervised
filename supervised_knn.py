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

import csv
import math
import random
from collections import defaultdict

# Load current CSV (Label + 21 cols, each is "x,y,z")
def load_22col_landmark_csv(csv_path, label_col="Label", expected_landmarks=21):
    X, y = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV missing header.")

        if label_col not in reader.fieldnames:
            raise ValueError(f"Label column '{label_col}' not found.")

        lm_cols = [f"lm{i}" for i in range(expected_landmarks)]
        for c in lm_cols:
            if c not in reader.fieldnames:
                raise ValueError(f"Expected column '{c}' not found. Found: {reader.fieldnames}")

        for row in reader:
            label = row[label_col]
            feats = []
            for c in lm_cols:
                s = row[c].strip()
                parts = s.split(",")
                if len(parts) != 3:
                    raise ValueError(f"Bad landmark format in col {c}: {s}")
                feats.extend([float(parts[0]), float(parts[1]), float(parts[2])])
            X.append(feats)
            y.append(label)

    labels_sorted = sorted(set(y))
    return X, y, labels_sorted

def stratified_split(X, y, test_ratio=0.2, seed=42):
    rnd = random.Random(seed)
    by_class = defaultdict(list)
    for xi, yi in zip(X, y):
        by_class[yi].append(xi)

    X_train, y_train, X_test, y_test = [], [], [], []
    for cls, items in by_class.items():
        rnd.shuffle(items)
        n = len(items)
        if n <= 1:
            X_train.extend(items)
            y_train.extend([cls] * len(items))
            continue

        n_test = int(round(n * test_ratio))
        n_test = max(1, min(n_test, n - 1))  
        test_items = items[:n_test]
        train_items = items[n_test:]

        X_test.extend(test_items)
        y_test.extend([cls] * len(test_items))
        X_train.extend(train_items)
        y_train.extend([cls] * len(train_items))

    # shuffle overall
    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    rnd.shuffle(train)
    rnd.shuffle(test)

    X_train, y_train = (list(t) for t in zip(*train)) if train else ([], [])
    X_test, y_test = (list(t) for t in zip(*test)) if test else ([], [])
    return X_train, y_train, X_test, y_test

def fit_standardizer(X_train):
    n = len(X_train)
    d = len(X_train[0]) if n else 0

    means = [0.0] * d
    for x in X_train:
        for j, v in enumerate(x):
            means[j] += v
    means = [m / n for m in means]

    vars_ = [0.0] * d
    for x in X_train:
        for j, v in enumerate(x):
            diff = v - means[j]
            vars_[j] += diff * diff
    vars_ = [v / n for v in vars_]
    stds = [math.sqrt(v) for v in vars_]
    stds = [s if s > 0.0 else 1.0 for s in stds]  
    return means, stds

def standardize(X, means, stds):
    out = []
    for x in X:
        out.append([(v - means[j]) / stds[j] for j, v in enumerate(x)])
    return out

# KNN 
def euclidean_distance_sq(a, b):
    s = 0.0
    for va, vb in zip(a, b):
        diff = va - vb
        s += diff * diff
    return s

def knn_predict_one(x, X_train, y_train, k=5, weighted=True):
    dists = []
    for xi, yi in zip(X_train, y_train):
        dsq = euclidean_distance_sq(x, xi)
        dists.append((dsq, yi))
    dists.sort(key=lambda t: t[0])
    neigh = dists[:k]

    if not weighted:
        counts = defaultdict(int)
        for _, lab in neigh:
            counts[lab] += 1
        return sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0]

    eps = 1e-9
    scores = defaultdict(float)
    for dsq, lab in neigh:
        w = 1.0 / (math.sqrt(dsq) + eps)
        scores[lab] += w
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))[0][0]

def knn_predict(X_test, X_train, y_train, k=5, weighted=True):
    return [knn_predict_one(x, X_train, y_train, k=k, weighted=weighted) for x in X_test]

def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m

def print_confusion_matrix(cm, labels):
    width = max(7, max(len(l) for l in labels) + 2)
    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print("".rjust(width) + "".join(l.rjust(width) for l in labels))
    for lab, row in zip(labels, cm):
        print(lab.rjust(width) + "".join(str(v).rjust(width) for v in row))

def print_per_class_metrics(cm, labels):
    n = len(labels)
    row_sum = [sum(cm[i]) for i in range(n)]
    col_sum = [sum(cm[i][j] for i in range(n)) for j in range(n)]

    print("\nPer-class metrics:")
    print("Label\tPrecision\tRecall\t\tF1\t\tSupport")
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = col_sum[i] - tp
        fn = row_sum[i] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        print(f"{lab}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{row_sum[i]}")

# Main
def evaluate(csv_path="preprocessed_extracted_features.csv", test_ratio=0.2, seed=42):
    csv_path = "preprocessed_extracted_features.csv"  
    X, y, labels = load_22col_landmark_csv(csv_path)

    X_train, y_train, X_test, y_test = stratified_split(X, y, test_ratio=0.2, seed=42)

    means, stds = fit_standardizer(X_train)
    X_train_s = standardize(X_train, means, stds)
    X_test_s = standardize(X_test, means, stds)

    k = 10
    weighted = True
    y_pred = knn_predict(X_test_s, X_train_s, y_train, k=k, weighted=weighted)

    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}  (k={k}, weighted={weighted})")

    cm = confusion_matrix(y_test, y_pred, labels)
    print_confusion_matrix(cm, labels)
    print_per_class_metrics(cm, labels)

    #quick k sweep
    print("\nK sweep:")
    for kk in [1, 3, 5, 7, 9]:
        yp = knn_predict(X_test_s, X_train_s, y_train, k=kk, weighted=True)
        print(f"k={kk} -> acc={accuracy(y_test, yp):.4f}")

    return acc, cm, labels

def main():
    acc, cm, labels = evaluate()
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
