from collections import Counter
from typing import *
import numpy as np

class RegressionTree:
    # Creates and trains a regression tree using X = matrix of attributes, Y = vector of values (classes)
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.all_attrs = [set() for _ in range(len(x[0]))]
        self.all_classes = set(y)
        self.most_common_class = Counter(y).most_common(1)[0][0]
        for row in x:
            for i, a in enumerate(row):
                if self.Node.is_discrete(x, i):
                    self.all_attrs[i].add(a)

        self.tree = self.Node(self, x, y, list(range(len(x[0]))))

    def predict(self, x_row):
        node = self.tree
        while(True):
            if node.leaf:
                return node.leaf_value
            elif node.threshold is not None:
                node = node.children[0 if x_row[node.attr] <= node.threshold else 1]
                continue
            for c in node.children:
                if x_row[node.attr] == c.attr_value:
                    node = c
                    break
            else:
                return self.most_common_class

    def get_tree_logic(self) -> str:
        return self.__inner_get_tree_logic(0, self.tree)

    def __inner_get_tree_logic(self, indent: int, node: 'RegressionTree.Node') -> str:
        ind = " " * indent
        if node.leaf:
            return f"{ind}return '{node.leaf_value}'"
        discrete = node.threshold is None
        o = f"{ind}if attr[{node.attr}] {'==' if discrete else '<= ' + str(node.threshold)}"
        for i, c in enumerate(node.children):
            o += f"{(' ' + c.attr_value.__repr__()) if discrete else ''}:\n{self.__inner_get_tree_logic(indent + 2, c)}"
            if i < len(node.children) - 1:
                o += f"\n{ind}elif attr[{node.attr}] ==" if discrete else f"\n{ind}else"
        return o

    class Node:
        SubsetType = Dict[Any, Tuple[List[Any], List[List[Any]]]]  # in other words: Dict[the attribute value that data was split on, Tuple[Y values for the split, X values for the split]]

        def __init__(self, tree: 'RegressionTree', x, y, attrs: List[int], value=None) -> None:
            self.tree: RegressionTree = tree
            self.leaf_value = None  # the result of this node if a leaf node, None if failed node
            self.attr_value = value
            self.attr: int | None = None  # the attribute that this node decides on
            self.leaf: bool = True
            self.threshold: float | None = None  # if a floating attribute node
            self.children: List[RegressionTree.Node] = []
            self.y = y
            self.x = x
            self.attrs = attrs

            _, cvn = self.standard_deviation(y)

            if len(y) == 0:  # there are no instances having the current attribute value in the current subset
                self.leaf_value = self.tree.most_common_class
            elif len(y) < 4 or cvn < 0.05 or len(attrs) == 0:  # coefficient under threshold or no attributes left to decide on
                self.leaf_value = sum(y)/len(y) #assigning average value

            else:
                (best_attr, best_threshold, splits) = self.split(x, y, attrs)
                rest_attrs = attrs.copy()
                rest_attrs.remove(best_attr)
                self.leaf = False
                self.attr = best_attr
                self.threshold = best_threshold
                self.children = [RegressionTree.Node(self.tree, subset[1], subset[0], rest_attrs, split_attr) for
                                 split_attr, subset in splits.items()]

        def split(self, x, y, attrs) -> Tuple[int, float, SubsetType]:
            _, best_attr = self.standard_deviation_reduction(x, y, attrs)
            subsets: RegressionTree.Node.SubsetType = {a: ([], []) for a in self.tree.all_attrs[best_attr]}
            best_threshold = None

            if self.is_discrete(x, best_attr):
                for i, row in enumerate(x):
                    subsets[row[best_attr]][0].append(y[i])
                    subsets[row[best_attr]][1].append(row)
            else:
                sort_idx = np.argsort([row[best_attr] for row in x])
                for i in range(len(sort_idx) - 1):
                    if x[i][best_attr] != x[i + 1][best_attr]:
                        lesser = ([], [])
                        greater = ([], [])
                        best_threshold = (x[i][best_attr] + x[i + 1][best_attr]) / 2
                        for ri, row in enumerate(x):
                            which = (greater if row[best_attr] > best_threshold else lesser)
                            which[0].append(y[ri])
                            which[1].append(row)
                        subsets = {'<=': lesser, '>': greater}

            return (best_attr, best_threshold, subsets)

        @staticmethod
        def is_discrete(x, attr: int) -> bool:
            for row in x:
                if row[attr] is None:
                    continue
                else:
                    return not (type(row[attr]) == int or type(row[attr]) == float )
            return True

        @staticmethod
        def standard_deviation(y) -> Tuple[float, float]:
            n = len(y)
            if n>0:
                average = sum(y) / n
                sd = np.sqrt(sum((v - average) ** 2 for v in y) / n)  # standard deviation
                cv = sd / average  # coefficient of variation
                return sd, cv
            else:
                return np.inf, np.inf

        def standard_deviation_reduction(self, x, y, attrs) -> Tuple[float, int]:
            sdy, _ = self.standard_deviation(y)
            max_sdr = 0
            best_attr = None

            for attr in attrs:
                    discrete = self.is_discrete(x, attr)
                    sdr = sdy
                    for a in self.tree.all_attrs[attr]:
                        idx = []

                        if discrete:
                            for ri, row in enumerate(x):
                                if row[attr] == a:
                                    idx.append(ri)

                        else:
                            for ri, row in enumerate(x):
                                if row[attr] > a*0.9 and row[attr] < a*1.1:
                                    idx.append(ri)
                        if len(idx)>0:
                            s, _ = self.standard_deviation([y[i] for i in idx])
                            sdr -= s * len(idx) / len(x) #subtract stdev * probability of value

                    if sdr > max_sdr:
                        max_sdr = sdr
                        best_attr = attr



            return max_sdr, best_attr