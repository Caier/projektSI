from collections import Counter
from typing import *
import math
import numpy as np

#example call: C45Tree(x, y, max_depth=5, min_sample_split=5)
#the keyword arguments (hyperparameters) are not required to be set
class C45Tree:
    #Creates and trains a C4.5 decision tree using X = matrix of attributes, Y = vector of values (classes)
    def __init__(self, x, y, max_depth = 0xFFFFFFFF, min_sample_split = 1) -> None:
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.x = x
        self.y = y
        self.all_attrs = [set() for _ in range(len(x[0]))]
        self.all_classes = set(y)
        self.most_common_class = Counter(y).most_common(1)[0][0]
        for row in x:
            for i, a in enumerate(row):
                if self.Node.is_discrete(x, i):
                    self.all_attrs[i].add(a)

        self.tree = self.Node(self, x, y, list(range(len(x[0]))), 0)

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

    def __inner_get_tree_logic(self, indent: int, node: 'C45Tree.Node') -> str:
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
        SubsetType = Dict[Any, Tuple[List[Any], List[List[Any]]]] #in other words: Dict[the attribute value that data was split on, Tuple[Y values for the split, X values for the split]]

        def __init__(self, tree: 'C45Tree', x, y, attrs: List[int], depth: int, value = None) -> None:
            self.tree: C45Tree = tree
            self.leaf_value = None #the result of this node if a leaf node, None if failed node
            self.attr_value = value
            self.attr: int | None = None #the attribute that this node decides on
            self.leaf: bool = True
            self.threshold: float | None = None #if a floating attribute node
            self.children: List[C45Tree.Node] = []

            if len(y) == 0: #there are no instances having the current attribute value in the current subset 
                self.leaf_value = self.tree.most_common_class
            elif not any(v != y[0] for v in y): #all values are of same class
                self.leaf_value = y[0]
            elif len(attrs) == 0 or tree.min_sample_split > len(x) or tree.max_depth <= depth: #no attributes left to decide on or not enough samples or too deep
                self.leaf_value = Counter(y).most_common(1)[0][0]
            else:
                (best_attr, best_thresh, splits) = self.split(x, y, attrs)
                rest_attrs = attrs.copy()
                rest_attrs.remove(best_attr)
                self.leaf = False
                self.attr = best_attr
                self.threshold = best_thresh
                self.children = [C45Tree.Node(self.tree, subset[1], subset[0], rest_attrs, depth+1, split_attr) for split_attr, subset in splits.items()]

        def split(self, x, y, attrs) -> Tuple[int, float, SubsetType]:
            splitted: C45Tree.Node.SubsetType = {}
            best_attr: int = None
            best_thresh: float = None
            max_gain = -math.inf

            for attr in attrs:
                subsets: C45Tree.Node.SubsetType = { a: ([], []) for a in self.tree.all_attrs[attr] }
                if self.is_discrete(x, attr):
                    for i, row in enumerate(x):
                        subsets[row[attr]][0].append(y[i])
                        subsets[row[attr]][1].append(row)
                    gain = self.gain_ratio(y, subsets)
                    if gain > max_gain:
                        max_gain = gain
                        splitted = subsets
                        best_attr = attr
                        best_thresh = None
                else:
                    sort_idx = np.argsort([row[attr] for row in x])
                    for i in range(len(sort_idx) - 1):
                        if x[i][attr] != x[i+1][attr]:
                            lesseq = ([], []); greater = ([], [])
                            threshold = (x[i][attr] + x[i+1][attr]) / 2
                            for ri, row in enumerate(x):
                                which = (greater if row[attr] > threshold else lesseq)
                                which[0].append(y[ri])
                                which[1].append(row)
                            subsets = {'<=': lesseq, '>': greater }
                            gain = self.gain_ratio(y, subsets)
                            if gain > max_gain:
                                max_gain = gain
                                splitted = subsets
                                best_attr = attr
                                best_thresh = threshold

            return (best_attr, best_thresh, splitted)

        @staticmethod
        def is_discrete(x, attr: int) -> bool:
            for row in x:
                if row[attr] is None:
                    continue
                else: return not (type(row[attr]) == int or type(row[attr]) == float)
            return True

        @staticmethod
        def gain_ratio(y, splits: SubsetType) -> float:
            E_before_split = C45Tree.Node.entropy(y)
            E_after_split = sum((len(split[0]) / len(y)) * C45Tree.Node.entropy(split[0]) for split in splits.values())
            split_info = 0
            for split in splits.values():
                ratio = len(split[0]) / len(y)
                if ratio == 0 or math.isnan(ratio):
                    continue
                split_info += ratio * math.log2(ratio)
            if split_info == 0:
                return math.inf
            return (E_before_split - E_after_split) / -split_info

        @staticmethod
        def entropy(y) -> float:
            class_freq = [v / len(y) for v in Counter(y).values()]
            return -sum(v * math.log2(v) for v in class_freq)