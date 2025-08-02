import numpy as np
from collections import Counter
import random

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _best_split(self, X, y):
        m, n = X.shape
        best_gain = -1
        split_idx, split_thresh = None, None
        for i in range(n):
            thresholds = np.unique(X[:, i])
            for thresh in thresholds:
                left_mask = X[:, i] <= thresh
                right_mask = X[:, i] > thresh
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_thresh = thresh
        return split_idx, split_thresh

    def _information_gain(self, parent, left, right):
        def gini(y):
            counts = np.bincount(y)
            probs = counts / len(y)
            return 1 - np.sum(probs**2)
        m = len(parent)
        return gini(parent) - len(left)/m * gini(left) - len(right)/m * gini(right)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth or len(np.unique(y)) == 1 or num_samples < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return Counter(y).most_common(1)[0][0]
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return (feat_idx, threshold, left_subtree, right_subtree)

    def _predict(self, inputs, node):
        if not isinstance(node, tuple):
            return node
        feat_idx, threshold, left, right = node
        if inputs[feat_idx] <= threshold:
            return self._predict(inputs, left)
        else:
            return self._predict(inputs, right)



class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_features(self, X):
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Select random subset of features for each tree
            feature_indices = np.random.choice(X.shape[1], self._get_features(X), replace=False)
            tree.feature_indices = feature_indices
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([
            tree.predict(X[:, tree.feature_indices]) for tree in self.trees
        ])
        tree_preds = tree_preds.T
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])


