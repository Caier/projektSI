import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor 

import sys
sys.path.append('../projektSI')  

from regression_tree.RegressionTree import RegressionTree
from datasets.wine import prepareXY as wine
from datasets.automobile import prepareXY as automobile
from datasets.energy import prepareXY as energy
from datasets.machine import prepareXY as machine
from datasets.bone_marrow import prepareXY as bone_marrow
from datasets.slump import prepareXY as slump

for dataset in (automobile, bone_marrow, energy, machine, slump, wine):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test =  [row for i, row in enumerate(X) if i not in train_idx]
    Y_test =  [row for i, row in enumerate(Y) if i not in train_idx]

    rt = RegressionTree(X_train, Y_train)

    MSE = 0 #mean square error
    MAE = 0 #mean absolute error
   
    for x, y in zip(X_test, Y_test):
        prediction = rt.predict(x)
        
        MAE += abs(y - prediction)
        MSE += (y - prediction)**2
      
    MSE /= len(Y_test)
    MAE /= len(Y_test)
    RMSE = np.sqrt(MSE)

    print(f"{dataset.__name__}: MAE: {MAE} RMSE: {RMSE}")



#comparison with sklearn 
print("COMPARISON WITH SCIKIT-LEARN")
for dataset in (energy, slump, wine):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test =  [row for i, row in enumerate(X) if i not in train_idx]
    Y_test =  [row for i, row in enumerate(Y) if i not in train_idx]

    rt = RegressionTree(X_train, Y_train)
    ct = DecisionTreeRegressor(random_state=0) #compared tree
    ct.fit(X_train, Y_train)
    Cpredictions = ct.predict(X_test)

    rMSE = 0
    rMAE = 0
    cMSE = 0
    cMAE = 0 

    for x, y, c in zip(X_test, Y_test, Cpredictions):
        Rprediction = rt.predict(x)
              
        rMAE += abs(y - Rprediction)
        rMSE += (y - Rprediction)**2         
        cMAE += abs(y - c)
        cMSE += (y - c)**2 
        
    cMSE /= len(Y_test)
    cMAE /= len(Y_test)
    rMSE /= len(Y_test)
    rMAE /= len(Y_test)


    print(f"{dataset.__name__}:\nErrors:\nMAE: {rMAE}\nRMSE: {np.sqrt(rMSE)}\nSKlearn: \nMAE: {cMAE}\nRMSE: {np.sqrt(cMSE)}")








   




