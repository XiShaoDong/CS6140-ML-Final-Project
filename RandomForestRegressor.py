class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="mse"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth) or (len(y) < self.min_samples_split):
            return np.mean(y)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return np.mean(y)

        left_idx = X[:, best_feat] < best_thresh
        right_idx = X[:, best_feat] >= best_thresh
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feat, best_thresh, left, right)

    def _best_split(self, X, y):
        best_loss = float('inf')
        best_feat, best_thresh = None, None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] < thresh
                right_idx = X[:, feat] >= thresh
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                loss = self._mse(y[left_idx]) * len(left_idx) + self._mse(y[right_idx]) * len(right_idx)
                if loss < best_loss:
                    best_loss = loss
                    best_feat, best_thresh = feat, thresh
        return best_feat, best_thresh

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        return self._predict_one(x, left if x[feat] < thresh else right)
    
class RandomForestRegressor:
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
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
