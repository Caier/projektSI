import pandas as pd
import numpy as np


def prepareXY():

    data = pd.read_csv("datasets/machine/machine.data").to_numpy()[:,:-1]
    Y = data[:,-1]
    X = np.delete(data, -1, axis=1)
   


    return X, Y