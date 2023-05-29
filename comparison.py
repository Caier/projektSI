import matplotlib.pyplot as plt
import random
from c45.c45tree import C45Tree
from id3.id3tree import ID3Tree
from random_forest.random_forest import RandomForest
import datasets.iris.prepareXY as iris
import datasets.grzybki.prepareXY as grzybki
import datasets.dry_bean.prepareXY as dry_bean
import datasets.titanic.prepareXY as titanic
import datasets.adult.prepareXY as adult
import datasets.zoo.prepareXY as zoo
import datasets.ionosphere.prepareXY as iono

# bean_data = ("Dry Beans", dry_bean.prepareXY())
# iono_data = ("Ionosphere", iono.prepareXY())
# zoo_data = ("Zoo", zoo.prepareXY())
# adult_data = ("Adult", adult.prepareXY(500))

def get_train_test(X, Y):
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test = [row for i, row in enumerate(X) if i not in train_idx]
    Y_test = [row for i, row in enumerate(Y) if i not in train_idx]
    return X_train, Y_train, X_test, Y_test

def plot(dataset):
    tries = 5
    c45Acc = 0; id3Acc = 0; rfAcc = 0
    for i in range(tries):
        print(f"iter: {i}")
        zX, zY, zXt, zYt = get_train_test(*dataset[1])
        zc45 = C45Tree(zX, zY)
        zid3 = ID3Tree(zX, zY)
        zc45Acc = 0; zid3Acc = 0; zrfAcc = 0
        for x, y in zip(zXt, zYt):
            if zc45.predict(x) == y:
                zc45Acc += 1
            if zid3.predict(x) == y:
                zid3Acc += 1
        zrf = RandomForest(C45Tree, num_trees=20, max_depth=10, min_sample_split=2)
        zrf.fit(zX, zY)
        zrfPred = zrf.predict(zXt)
        for i in range(len(zYt)):
            if zrfPred[i] == zYt[i]:
                zrfAcc += 1
        c45Acc += zc45Acc / len(zYt)
        id3Acc += zid3Acc / len(zYt)
        rfAcc += zrfAcc / len(zYt)
    c45Acc /= tries; id3Acc /= tries; rfAcc /= tries
    w = [c45Acc, id3Acc, rfAcc]
    plt.bar([1, 2, 3], w, width=1, color=["green", "blue", "orange"])
    print(w)
    plt.ylim(min(w) - 0.1, max(w))
    plt.ylabel("Dokładność")
    plt.xticks([1,2,3], ["C4.5", "ID3", "RF"])
    plt.title(f"Średnia dokładność każdej z metod dla zbioru {dataset[0]}")
    plt.show()

#plot(("Dry Beans", dry_bean.prepareXY()))
#plot(("Adult", adult.prepareXY(500)))
plot(("Iono", iono.prepareXY()))