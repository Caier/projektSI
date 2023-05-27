from c45.c45tree import C45Tree

#for our purposes the ID3 tree is just like the C4.5 tree but uses information gain instead of gain ratio

class ID3Tree(C45Tree):
    class Node(C45Tree.Node):
        def goodness_function(self, y, splits: C45Tree.Node.SubsetType) -> float:
            E_before_split = C45Tree.Node.entropy(y)
            E_after_split = sum((len(split[0]) / len(y)) * C45Tree.Node.entropy(split[0]) for split in splits.values())
            return E_before_split - E_after_split