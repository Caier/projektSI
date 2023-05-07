import pathlib

def prepareXY():
    with open(pathlib.Path(__file__).parent.joinpath('agaricus-lepiota.data'), 'r') as f:
        Y = []; X = []
        for line in f.readlines():
            data = line.strip().split(',')
            Y.append(data[0])
            X.append(data[1:])
        return (X, Y)