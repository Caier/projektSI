import pathlib

def prepareXY():
    with open(pathlib.Path(__file__).parent.joinpath('train.csv'), 'r') as train:
        Y = []; X = []
        for line in train.readlines()[1:]:
            data = line.strip().split(',')
            Y.append(data[1])
            tf = lambda x: None if x == '' else float(x)
            row = [data[2], data[4], tf(data[5]), tf(data[6]), tf(data[7]), tf(data[9]), data[10], data[11]]
            X.append(row)
        return (X, Y)