import numpy as np
import random

import sys
sys.path.append('../projektSI')  

from regression_tree.RegressionTree import RegressionTree
from datasets.wine import prepareXY as wine
from datasets.automobile import prepareXY as automobile
from datasets.energy import prepareXY as energy
from datasets.machine import prepareXY as machine
from datasets.bone_marrow import prepareXY as bone_marrow
from datasets.slump import prepareXY as slump





  


for dataset in (automobile, wine, energy, machine, bone_marrow, slump):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test =  [row for i, row in enumerate(X) if i not in train_idx]
    Y_test =  [row for i, row in enumerate(Y) if i not in train_idx]

    rt = RegressionTree(X_train, Y_train)

    SE = 0 #standard error of regression
    MSE = 0 #mean square error
    MAE = 0 #mean absolute error
   
    for x, y in zip(X_test, Y_test):
        prediction = rt.predict(x)
        
        MAE += abs(y - prediction)
        MSE += (y - prediction)**2
      
    MSE /= len(Y_test)
    MAE /= len(Y_test)
    RMSE = np.sqrt(MSE)



    print(f"{dataset.__name__}: MAE: {MAE}")

   



