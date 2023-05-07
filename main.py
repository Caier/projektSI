from c45.c45tree import C45Tree
import random
import pprint

pp = pprint.PrettyPrinter(2, 80)

with open("./agaricus-lepiota.data", 'r') as f:
    Y = []; X = []
    for line in f.readlines():
        data = line.strip().split(',')
        Y.append(data[0])
        X.append(data[1:])

while(True):
    train_idx = random.sample(range(len(Y)), int(0.1 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test = [row for i, row in enumerate(X) if i not in train_idx]
    Y_test = [row for i, row in enumerate(Y) if i not in train_idx]
    
    attrsTest = [set() for _ in range(len(X[0]))]
    for row in X_test:
        for i, v in enumerate(row):
            attrsTest[i].add(v)
    attrsTrain = [set() for _ in range(len(X[0]))]
    for row in X_train:
        for i, v in enumerate(row):
            attrsTrain[i].add(v)

    for i in range(len(attrsTest)):
        if any(v not in attrsTrain[i] for v in attrsTest[i]):
            print("Test dataset contains attributes that were not included in train data, reshuffling...")
            break
    else:
        #pp.pprint((attrsTest, attrsTrain))
        break

c45 = C45Tree(X_train, Y_train)

correct = 0
for x, y in zip(X_test, Y_test):
    if c45.predict(x) == y:
        correct += 1

print(correct / len(Y_test))