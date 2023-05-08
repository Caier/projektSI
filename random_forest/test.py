from random_forest import RandomForest
import datasets.iris.prepareXY as iris
import datasets.grzybki.prepareXY as grzybki
import random

for dataset in (grzybki, iris):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.1 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test = [row for i, row in enumerate(X) if i not in train_idx]
    Y_test = [row for i, row in enumerate(Y) if i not in train_idx]

    random_forest = RandomForest(X_train, Y_train)
    random_forest.fit()

    preds = random_forest.predict(X_test)
    correct = 0
    for i in range(len(preds)):
        if preds[i] == Y_test[i]:
            correct += 1

    print(dataset.__name__)
    print(correct / len(Y_test))