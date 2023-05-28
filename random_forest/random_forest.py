import math
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

    def sample(self, X, y):
        xx = np.asarray(X)
        yy = np.asarray(y)
        n_rows = xx.shape[0]
        samples = np.random.choice(n_rows, n_rows, replace=True)
        xxx = xx[samples]
        yyy = yy[samples]
        return xxx, yyy

    def fit(self, X, y):
        self.decision_trees = []

        num_built = 0
        while num_built < self.num_trees:
            _X, _y = self.sample(X, y)
            clf = self.tree_type(_X, _y, max_depth=self.max_depth, min_sample_split=self.min_samples_split)
            self.decision_trees.append(clf)
            num_built += 1

    def predict(self, X):
        predictions = np.array([[tree.predict(X_row) for X_row in X] for tree in self.decision_trees])
        predictions = np.swapaxes(predictions, 0, 1)
        preds = np.array([self.most_common_label(p) for p in predictions])
        return preds

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
