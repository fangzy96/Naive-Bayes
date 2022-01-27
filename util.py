import numpy as np
from numba import jit

@jit(nopython=True, fastmath = True)
def confusion_matrix(y: np.ndarray, y_hat: np.ndarray):
    # to calculate TP, FN, FP, TN
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(0, len(y)-1):
        if y_hat[i] == 1:
            if y[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y[i] == 0:
                TN += 1
            else:
                FN += 1

    return TP, FP, FN, TN


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!
    Args:
        y: True labels.
        y_hat: Predicted labels.
    Returns: Accuracy
    """
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    n = y.size # (TP + FP + TN + FN)

    return (y == y_hat).sum() / n # (TP + TN)/(TP + FP + TN + FN)


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    TP = 0
    FP = 0

    for i in range(0, len(y)):
        if y[i]==1 and y_hat[i]==1:
            TP += 1
        elif y[i]==0 and y_hat[i]==1:
            FP += 1

    return TP / (TP + FP)

@jit(nopython=True, fastmath = True)
def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    TP = 0
    FN = 0

    for i in range(0, len(y)):
        if y[i]==1 and y_hat[i]==1:
            TP += 1
        elif y[i]==1 and y_hat[i]==0:
            FN += 1

    return TP / (TP + FN)

@jit(nopython=True, fastmath = True)
def specificity(y: np.ndarray, y_hat: np.ndarray) -> float:
    # y: True labels.
    # y_hat: Predicted labels.
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    TN = 0
    FP = 0

    for i in range(0, len(y)):
        if y[i] == 0 and y_hat[i] == 0:
            TN += 1
        elif y[i] == 0 and y_hat[i] == 1:
            FP += 1

    return TN / (TN + FP)

@jit(nopython=True, fastmath = True)
def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray):
    # y: True labels.
    # p_y_hat: Confidence on +
    if y.size != p_y_hat.size:
        raise ValueError('y and p_y_hat must be the same shape/size!')

    n = len(y)
    roc_pairs = np.zeros((n+1,2))

    # to calculate the pairs
    for i in range(0,n):
        y_hat = np.zeros(n)
        y_hat[p_y_hat>=p_y_hat[i]] = 1 # each confidence is seen as a decision value
        TP, FP, FN, TN = confusion_matrix(y, y_hat)
        True_rate = TP/(TP+FN)
        False_rate = FP/(FP+TN)
        roc_pairs[i+1, 0] = False_rate
        roc_pairs[i+1, 1] = True_rate

    return roc_pairs


# to calculate the AUC depending on pairs from ROC
def auc(roc_pairs: np.ndarray) -> float:
    # Compute the area under the ROC graph curve
    roc_pairs = np.sort(roc_pairs,axis=0)
    n = roc_pairs.shape[0]
    auc_area = 0
    # to calculate the integral
    for i in range(1,n):
        auc_area += 0.5 * (roc_pairs[i,0]-roc_pairs[i-1,0])*(roc_pairs[i,1] + roc_pairs[i-1,1])

    return auc_area


def cv_split(X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False):
    y_R = np.transpose(y)
    Data = np.column_stack((X,y_R))
    Data = Data[Data[:,-1].argsort()] # Sort Data by the last column
    Rows = len(X) #Calculate how many rows there is in the Array
    #For loop, I want the index of the row when I can split 0 and 1
    result_x =[]
    result_y =[]
    for i in range(Rows-1):
        if Data[i,-1] != Data[i+1, -1]:
            index =i
    #index is the index of the last 0 row
    #split the Data Array by the index row, into DataArray0 and DataArray1, 0 and 1 are labels
    Data_0 = Data[0:index+1,:]
    Data_1 = Data[index+1:Rows,:]

    Ran_Data0 = np.random.permutation(Data_0)
    Ran_Data1 = np.random.permutation(Data_1)
    #Split the Data0 array and Data1 array into 5 parts and merge them separately
    Split_D0 = np.array_split(Ran_Data0,folds,axis = 0)
    Split_D1 = np.array_split(Ran_Data1,folds,axis = 0)
    #Split is completed
    for i in range(folds):
        a = np.vstack((Split_D0[i], Split_D1[i]))
        result_x.append(a[:,0:len(X[0])])
        result_y.append(a[:,-1])

    return result_x,result_y
