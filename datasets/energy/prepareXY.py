import pandas as pd
import numpy as np




def prepareXY():

    data = pd.read_csv("datasets/energy/energydata_complete.csv").to_numpy()

    Y = data[:,1]
    X = np.delete(data, 1, axis=1)

    newX0 = np.zeros([len(X), 6])

    for i, dt in enumerate(X[:,0]):
        dt = str.split(dt, sep="-")
        dt2 = str.split(dt[2], sep=":")
        dt3 = str.split(dt2[0], sep=" ")
        final = dt[:2]+dt3+dt2[1:]
        final = np.array([int(j) for j in final])
        newX0[i] = final
        
        

    X = np.delete(X, 0, axis=1)

    for i in range(6):
        X = np.insert(X, i, newX0[:,i], axis=1)

    
    return X, Y