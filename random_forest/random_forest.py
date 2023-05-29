import math
from collections import Counter
from c45.c45tree import C45Tree
from regression_tree.RegressionTree import RegressionTree
import numpy as np


class RandomForest:
    def __init__(self, tree_type: type(C45Tree) | type(RegressionTree), num_trees=25, max_depth=0xFFFFFFFF, min_sample_split=1):
        self.num_trees = num_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.tree_type = tree_type
        self.decision_trees = []

    def sample(self, X, y):
        n_rows = len(X)
        samples = np.random.choice(n_rows, n_rows, replace=True)
        x_samples = []
        y_samples = []
        for sample in samples:
            x_samples.append(X[sample])
            y_samples.append(y[sample])
        return x_samples, y_samples

    def fit(self, X, y):
        self.decision_trees = []

        num_built = 0
        while num_built < self.num_trees:
            x_sample, y_sample = self.sample(X, y)
            clf = self.tree_type(x_sample, y_sample, max_depth=self.max_depth, min_sample_split=self.min_sample_split)
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
