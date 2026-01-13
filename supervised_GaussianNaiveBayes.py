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
from collections import Counter, defaultdict

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

    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    rnd.shuffle(train)
    rnd.shuffle(test)

    X_train, y_train = (list(t) for t in zip(*train)) if train else ([], [])
    X_test, y_test = (list(t) for t in zip(*test)) if test else ([], [])
    return X_train, y_train, X_test, y_test

class GaussianNBFromScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.labels = None
        self.prior = {}     # label -> prior prob
        self.mean = {}      # label -> [mean_j]
        self.var = {}       # label -> [var_j]

    def fit(self, X, y):
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length.")
        d = len(X[0])
        if any(len(row) != d for row in X):
            raise ValueError("All X rows must have the same length.")

        self.labels = sorted(set(y))
        n = len(y)
        counts = Counter(y)

        # Priors
        self.prior = {lab: counts[lab] / n for lab in self.labels}

        # Means and variances per class per feature
        for lab in self.labels:
            idxs = [i for i, yi in enumerate(y) if yi == lab]
            m = len(idxs)

            # mean
            mu = [0.0] * d
            for i in idxs:
                xi = X[i]
                for j in range(d):
                    mu[j] += xi[j]
            mu = [v / m for v in mu]

            # variance 
            var = [0.0] * d
            for i in idxs:
                xi = X[i]
                for j in range(d):
                    diff = xi[j] - mu[j]
                    var[j] += diff * diff
            var = [v / m for v in var]

  
            var = [v + self.var_smoothing for v in var]

            self.mean[lab] = mu
            self.var[lab] = var

        return self

    def _log_gaussian_pdf(self, x, mu, var):
        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mu) ** 2) / (2.0 * var)

    def predict_one(self, x):
        best_label = None
        best_logp = -float("inf")
        d = len(x)

        for lab in self.labels:
            logp = math.log(self.prior[lab]) if self.prior[lab] > 0 else -float("inf")
            mu = self.mean[lab]
            var = self.var[lab]
            for j in range(d):
                logp += self._log_gaussian_pdf(x[j], mu[j], var[j])

            if logp > best_logp:
                best_logp = logp
                best_label = lab

        return best_label

    def predict(self, X):
        return [self.predict_one(x) for x in X]

def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

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

def evaluate(csv_path="preprocessed_extracted_features.csv", test_ratio=0.2, seed=42):
    csv_path = "preprocessed_extracted_features.csv"
    test_ratio = 0.2
    seed = 42

    # Load
    X, y, labels = load_22col_landmark_csv(csv_path)

    # Split
    X_train, y_train, X_test, y_test = stratified_split(X, y, test_ratio=test_ratio, seed=seed)

    # Train GNB
    gnb = GaussianNBFromScratch(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = gnb.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Train size: {len(y_train)} | Test size: {len(y_test)} | Classes: {len(labels)}")

    cm = confusion_matrix(y_test, y_pred, labels)
    print_confusion_matrix(cm, labels)
    print_per_class_metrics(cm, labels)
    return acc, cm, labels

def main():
    acc, cm, labels = evaluate()
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
