from collections import Counter
from c45.c45tree import C45Tree
from regression_tree.RegressionTree import RegressionTree
import numpy as np


class RandomForest:
    def __init__(self, tree_type: type(C45Tree) | type(RegressionTree), num_trees=25, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree_type = tree_type
        self.decision_trees = []

    @staticmethod
    def _sample(X, y):
        n_rows = len(X)
        samples = np.random.choice(n_rows, n_rows, replace=True)
        _X = []
        _y = []
        for sample in samples:
            _X.append(X[sample])
            _y.append(y[sample])
        return _X, _y

    def fit(self, X, y):
        self.decision_trees = []

        num_built = 0
        while num_built < self.num_trees:
            _X, _y = self._sample(X, y)
            clf = self.tree_type(_X, _y, max_depth=self.max_depth, min_sample_split=self.min_samples_split)
            self.decision_trees.append(clf)
            num_built += 1

    def predict(self, X):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))

        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
