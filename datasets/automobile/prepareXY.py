import pandas as pd
import numpy as np

def prepareXY():

    data = pd.read_csv("datasets/automobile/automobile.data").to_numpy()
    Y = []
    X = []
    for d in data:
        if d[-1] != '?':
            Y.append(int(d[-1]))
            X.append(d[:-1])


    return X, Y