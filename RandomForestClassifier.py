import numpy as np
from collections import Counter

import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini", max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split) or (len(set(y)) == 1):
            return self._most_common_label(y)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return self._most_common_label(y)

        left_idx = X[:, best_feat] < best_thresh
        right_idx = X[:, best_feat] >= best_thresh
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        n_features = X.shape[1]

        # 决定要使用多少个特征
        if self.max_features is None:
            features = range(n_features)
        else:
            max_feats = min(self.max_features, n_features)
            features = np.random.choice(n_features, max_feats, replace=False)

        best_gain = -1
        best_feat, best_thresh = None, None

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] < thresh
                right_idx = X[:, feat] >= thresh
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gain = self._gini_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, thresh
        return best_feat, best_thresh

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _gini_gain(self, y, left_idx, right_idx):
        n = len(y)
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        return self._gini(y) - (len(left_idx) / n) * gini_left - (len(right_idx) / n) * gini_right

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        return self._predict_one(x, left if x[feat] < thresh else right)


class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
