from collections import Counter
from typing import *
import math

class C45Tree:
    #Creates and trains a C4.5 decision tree using X = matrix of attributes, Y = vector of values (classes)
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.tree = Node(x, y, list(range(len(x[0]))))

    def predict(self, x_row):
        node = self.tree
        iter = 0
        while(True):
            iter += 1
            if iter > 15000:
                breakpoint()
            if node.leaf:
                return node.leaf_value
            for c in node.children:
                if x_row[node.attr] == c.attr_value:
                    node = c
                    break
        
class Node:
    SubsetType = Dict[Any, Tuple[List[Any], List[List[Any]]]] #in other words: Dict[the attribute value that data was split on, Tuple[Y values for the split, X values for the split]]

    def __init__(self, x, y, attrs: List[int], value = None) -> None:
        self.leaf_value = None #the result of this node if a leaf node, None if failed node
        self.attr_value = value
        self.attr: int = None #the attribute that this node decides on
        self.leaf: bool = True
        self.threshold: float = None #if a floating attribute node
        self.children: List[Node] = []

        if len(y) == 0:
            pass
        elif not any(v != y[0] for v in y): #all values are of same class
            self.leaf_value = y[0]
        elif len(attrs) == 0: #no attributes left to decide on
            self.leaf_value = Counter(y).most_common(1)[0][0]
        else:
            (best_attr, best_thresh, splits) = self.split(x, y, attrs)
            rest_attrs = attrs.copy()
            rest_attrs.remove(best_attr)
            self.leaf = False
            self.attr = best_attr
            self.threshold = best_thresh
            self.children = [Node(subset[1], subset[0], rest_attrs, split_attr) for split_attr, subset in splits.items()]

    @staticmethod
    def split(x, y, attrs) -> Tuple[int, float, SubsetType]:
        splitted: Node.SubsetType = {}
        best_attr: int = None
        best_thresh: float = None
        max_gain = -math.inf

        for attr in attrs:
            subsets: Node.SubsetType = {}
            if Node.is_discrete(x, attr):
                for i, row in enumerate(x): #tutaj może dojść do sytuacji że w akurat tym podzbiorze X nie będzie wszystkich możliwych wartości atrybutu
                    if row[attr] not in subsets: 
                        subsets[row[attr]] = ([y[i]], [row])
                    else:
                        subsets[row[attr]][0].append(y[i])
                        subsets[row[attr]][1].append(row)
                gain = Node.gain_ratio(y, subsets)
                if gain > max_gain:
                    max_gain = gain
                    splitted = subsets
                    best_attr = attr
                    best_thresh = None
            else:
                raise NotImplementedError()

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
        E_before_split = Node.entropy(y)
        E_after_split = sum((len(split[0]) / len(y)) * Node.entropy(split[0]) for split in splits.values())
        split_info = 0
        for split in splits.values():
            ratio = len(split[0]) / len(y)
            split_info += ratio * math.log2(ratio)
        if split_info == 0:
            return math.inf
        return (E_before_split - E_after_split) / -split_info

    @staticmethod
    def entropy(y) -> float:
        class_freq = [v / len(y) for v in Counter(y).values()]
        return -sum(v * math.log2(v) for v in class_freq)
