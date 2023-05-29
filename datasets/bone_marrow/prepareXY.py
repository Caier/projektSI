import pandas as pd
import numpy as np

def prepareXY():

    data = pd.read_csv("datasets/bone_marrow/bone_marrow.data").to_numpy()
    Y = data[:,-2]
    X = np.delete(data, -2, axis=1)
   

    return X, Y