import pathlib

def prepareXY(n: int = 1000):
    with open(pathlib.Path(__file__).parent.joinpath('adult.data'), 'r') as train:
        Y = []; X = []; attrs = [{} for _ in range(14)]
        for line in train.readlines()[:n]:
            data = line.strip().split(', ')
            Y.append(data[-1])
            row = data[:-1]
            for i in (0, 2, 4, 10, 11, 12):
                row[i] = float(row[i])
            for i in (1, 3, 5, 6, 7, 8, 9, 13):
                if row[i] in attrs[i]:
                    row[i] = attrs[i][row[i]]
                else:
                    attrs[i][row[i]] = str(len(attrs[i].keys()))
                    row[i] = attrs[i][row[i]]
            X.append(row)
        return (X, Y)