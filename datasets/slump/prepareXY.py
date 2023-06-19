import pandas as pd
import numpy as np




def prepareXY():
    data = np.delete(pd.read_csv("datasets/slump/slump_test.data").to_numpy(),0,axis=1)
    Y = data[:,-1]
    X = np.delete(data, -1, axis=1)


    return X, Y