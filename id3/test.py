from id3.id3tree import ID3Tree
import datasets.iris.prepareXY as iris
import datasets.grzybki.prepareXY as grzybki
import random

for dataset in (iris, grzybki):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.5 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test = [row for i, row in enumerate(X) if i not in train_idx]
    Y_test = [row for i, row in enumerate(Y) if i not in train_idx]

    c45 = ID3Tree(X_train, Y_train)

    correct = 0
    for x, y in zip(X_test, Y_test):
        if (v := c45.predict(x)) == y:
            correct += 1

    print(f"{dataset.__name__}: {correct / len(Y_test)}")
    #print(c45.get_tree_logic())