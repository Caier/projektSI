import pathlib
import random

def prepareXY():
    with open(pathlib.Path(__file__).parent.joinpath('dataset.csv'), 'r') as f:
        Y = []; X = []
        for line in f.readlines()[1:]:
            data = line.strip().split(';')
            Y.append(data[-1])
            X.append([float(x) for x in data[:-1]])
        idx = random.sample(range(len(Y)), 2000)
        return ([x for i, x in enumerate(X) if i in idx], [y for i, y in enumerate(Y) if i in idx])