import pathlib

def prepareXY():
    with open(pathlib.Path(__file__).parent.joinpath('iris.data'), 'r') as f:
        data = f.readlines()
        X = []; Y = []
        for line in data:
            row = line.strip().split(',')
            Y.append(row[-1])
            X.append([float(a) for a in row[:-1]])
        return (X, Y)