import pandas as pd

def prepareXY():


    data = pd.read_csv("datasets/forestfires/forestfires.csv").to_numpy()

    Y = data[1:, -1].astype(int)
    X = data[1:, :-1]
    return X, Y
