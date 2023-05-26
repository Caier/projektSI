import numpy as np

from regression_tree.RegressionTree import RegressionTree
from datasets.wine import prepareXY as wine
from datasets.automobile import prepareXY as automobile
import random

for dataset in (automobile, wine):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test =  [row for i, row in enumerate(X) if i not in train_idx]
    Y_test =  [row for i, row in enumerate(Y) if i not in train_idx]

    rt = RegressionTree(X_train, Y_train)

    SE = 0 #Standard error of regression
    for x, y in zip(X_test, Y_test):
        SE += (y - rt.predict(x))**2
    SE  = np.sqrt(SE/len(Y_test))/np.average(Y_test)


    # RSS = 0
    # SST = 0
    # for x, y in zip(X_test, Y_test):
    #     RSS += (y - rt.predict(x))**2
    #     SST += (y - np.average(Y_test))**2
    # R2 = 1 - RSS/SST

    print(f"{dataset.__name__}: {SE}")