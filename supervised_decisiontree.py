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

    # shuffle overall
    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))
    rnd.shuffle(train)
    rnd.shuffle(test)

    X_train, y_train = (list(t) for t in zip(*train)) if train else ([], [])
    X_test, y_test = (list(t) for t in zip(*test)) if test else ([], [])
    return X_train, y_train, X_test, y_test

# Standardization
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

# Decision Tree
class DecisionTreeClassifierScratch:
    class Node:
        __slots__ = ("is_leaf", "pred", "feature", "threshold", "left", "right")
        def __init__(self, is_leaf, pred=None, feature=None, threshold=None, left=None, right=None):
            self.is_leaf = is_leaf
            self.pred = pred
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  
        self.root = None

    def fit(self, X, y):
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length.")
        d = len(X[0])
        if any(len(row) != d for row in X):
            raise ValueError("All rows in X must have the same feature dimension.")
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        if self.root is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return [self._predict_one(row, self.root) for row in X]

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.pred

    @staticmethod
    def _majority_class(y):
        counts = Counter(y)
        # deterministic tie-break on label
        return sorted(counts.items(), key=lambda t: (-t[1], str(t[0])))[0][0]

    @staticmethod
    def _gini_from_counts(counts, n):
        if n == 0:
            return 0.0
        s = 1.0
        for c in counts.values():
            p = c / n
            s -= p * p
        return s

    def _build_tree(self, X, y, depth):
        pred = self._majority_class(y)

        # Stop rules
        if depth >= self.max_depth:
            return self.Node(is_leaf=True, pred=pred)
        if len(set(y)) == 1:
            return self.Node(is_leaf=True, pred=y[0])
        if len(y) < self.min_samples_split:
            return self.Node(is_leaf=True, pred=pred)

        best = self._best_split(X, y)
        if best is None:
            return self.Node(is_leaf=True, pred=pred)

        feat, thr, left_idx, right_idx = best
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return self.Node(is_leaf=True, pred=pred)

        X_left = [X[i] for i in left_idx]
        y_left = [y[i] for i in left_idx]
        X_right = [X[i] for i in right_idx]
        y_right = [y[i] for i in right_idx]

        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)

        return self.Node(
            is_leaf=False,
            pred=pred,
            feature=feat,
            threshold=thr,
            left=left_node,
            right=right_node
        )

    def _best_split(self, X, y):
        n = len(y)
        d = len(X[0])
        parent_counts = Counter(y)
        parent_gini = self._gini_from_counts(parent_counts, n)

        feat_indices = list(range(d))
        if self.max_features is not None and 0 < self.max_features < d:
            feat_indices = feat_indices[:self.max_features]  # deterministic subset

        best_gain = 0.0
        best = None

        for j in feat_indices:
            # Sort by feature value
            pairs = sorted(((X[i][j], y[i], i) for i in range(n)), key=lambda t: t[0])

            left_counts = Counter()
            right_counts = Counter(y)

            for k in range(0, n - 1):
                val_k, lab_k, idx_k = pairs[k]
                left_counts[lab_k] += 1
                right_counts[lab_k] -= 1
                if right_counts[lab_k] == 0:
                    del right_counts[lab_k]

                val_next = pairs[k + 1][0]
                if val_k == val_next:
                    continue 

                n_left = k + 1
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                g_left = self._gini_from_counts(left_counts, n_left)
                g_right = self._gini_from_counts(right_counts, n_right)
                weighted = (n_left / n) * g_left + (n_right / n) * g_right
                gain = parent_gini - weighted

                if gain > best_gain:
                    thr = (val_k + val_next) / 2.0
                    left_idx = [i for v, _, i in pairs if v <= thr]
                    right_idx = [i for v, _, i in pairs if v > thr]
                    best_gain = gain
                    best = (j, thr, left_idx, right_idx)

        return best

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
    # ---- Settings you may tune ----
    csv_path = "extracted_features.csv"  
    test_ratio = 0.2
    seed = 42

    # Decision tree hyperparameters
    max_depth = 12
    min_samples_split = 2
    min_samples_leaf = 3
    max_features = None  


    use_standardization = False

    # ---- Load & split ----
    X, y, labels = load_22col_landmark_csv(csv_path)
    X_train, y_train, X_test, y_test = stratified_split(X, y, test_ratio=test_ratio, seed=seed)

    # ---- Standardization----
    if use_standardization:
        means, stds = fit_standardizer(X_train)
        X_train = standardize(X_train, means, stds)
        X_test = standardize(X_test, means, stds)

    # ---- Train tree ----
    clf = DecisionTreeClassifierScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    clf.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"(max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, min_samples_split={min_samples_split}, max_features={max_features})")
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
