import pathlib

def prepareXY():
    with open(pathlib.Path(__file__).parent.joinpath('ionosphere.data'), 'r') as train:
        Y = []; X = []
        for line in train.readlines():
            data = line.strip().split(',')
            Y.append(data[-1])
            row = [float(x) for x in data[:-1]]
            X.append(row)
        return (X, Y)