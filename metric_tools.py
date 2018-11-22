import pandas as pd
import numpy as np


def tpr_weight_funtion(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    y_true = y_true.reshape(y_true.shape[0], 1)
    y_predict = y_predict.reshape(y_predict.shape[0], 1)

    y = np.concatenate((y_true, y_predict), axis=1)

    pos_all = np.argwhere(y_true == 1).shape[0]

    sort_index = np.argsort(y[:, 1])

    y = y[sort_index]

    tpr_fpr_list = []
    for i in range(y.shape[0]-1, -1, -1):
        tp = np.argwhere(y[i:, 0] == 1).shape[0]
        fn = np.argwhere(y[:i, 0] == 1).shape[0]
        fp = np.argwhere(y[i:, 0] == 0).shape[0]
        tn = np.argwhere(y[:i, 0] == 0).shape[0]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_fpr_list.append([tpr, fpr])
        if fpr > 0.05:
            break
    fpr001 = 1
    fpr005 = 1
    fpr01 = 1

    tpr1 = tpr2 = tpr3 = 0
    for tpr_fpr in tpr_fpr_list:
        if abs(tpr_fpr[1] - 0.001) < fpr001:
            tpr1 = tpr_fpr[0]
            fpr001 = abs(tpr_fpr[1] - 0.001)
        if abs(tpr_fpr[1] - 0.005) < fpr005:
            tpr2 = tpr_fpr[0]
            fpr005 = abs(tpr_fpr[1] - 0.005)
        if abs(tpr_fpr[1] - 0.01) < fpr01:
            tpr3 = tpr_fpr[0]
            fpr01 = abs(tpr_fpr[1] - 0.01)
    return tpr1 * 0.4 + tpr2 * 0.3 + tpr3 * 0.3


if __name__ == '__main__':
    y_predict = np.random.random(100)
    y_predict.sort()
    y_true = np.ones(100)
    y_true[:50] = 0

    tt = tpr_weight_funtion(y_true, y_predict)
    print(tt)
