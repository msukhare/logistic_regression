# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    appli_logistic_reg.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/18 10:10:30 by msukhare          #+#    #+#              #
#    Updated: 2018/10/18 15:03:11 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np
import csv

def get_max_min(X, j, max_or_min):
    to_ret = X[0][j]
    for i in range(int(X.shape[0])):
        if (max_or_min == 1 and to_ret < X[i][j]):
            to_ret = X[i][j]
        if (max_or_min == 0 and to_ret > X[i][j]):
            to_ret = X[i][j]
    return (to_ret)

def scale_features(X):
    scale_X = np.zeros((X.shape[0], X.shape[1]), dtype=float)
    for j in range(1, int(X.shape[1])):
        max_val = get_max_min(X, j, 1)
        min_val = get_max_min(X, j, 0)
        for i in range(1, int(X.shape[0])):
            scale_X[i][j] = X[i][j] / (max_val - min_val)
    for i in range(int(X.shape[0])):
        scale_X[i][0] = 1
    return (scale_X)

def read_file():
    if (sys.argv[1].lower().endswith(".csv") != 1):
        sys.exit("the first file must be a .csv")
    try:
        X = pd.read_csv(sys.argv[1], header=None)
    except:
        sys.exit("file doesn't exist")
    try:
        thetas = pd.read_csv("thetas", header=None)
    except:
        sys.exit("file doesn't exist, use logistic_reg.py to create")
    X.insert(0, None, 1)
    X = X.iloc[ :, :]
    X = np.array(X.values, dtype=float)
    thetas = thetas.iloc[ :, :]
    thetas = np.array(thetas.values, dtype=float)
    return (thetas, X, scale_features(X))

def write_prediction(X, Y_predict):
    try:
        file_with_pred = csv.writer(open("predictions", "w"))
    except:
        sys.exit("fail to create file")
    X = X[:, 1: ]
    for i in range(int(X.shape[0])):
        file_with_pred.writerow([str(X[i]), str(Y_predict[i])])

def main():
    if (len(sys.argv) != 2):
        sys.exit("usage: python3 [nameOfDataToPredict]")
    thetas, X, X_scale = read_file()
    if (thetas.shape[0] != X.shape[1]):
        sys.exit("number of features in file must be the same as number of thetas")
    y_predict = np.zeros((X_scale.shape[0], 1), dtype=int)
    for i in range(int(X_scale.shape[0])):
        pred = 0
        for j in range(int(thetas.shape[0])):
            pred += thetas[j][0] * X_scale[i][j]
        if (pred >= 0.5):
            y_predict[i][0] = 1
        else:
            y_predict[i][0] = 0
    write_prediction(X, y_predict)

if __name__ == "__main__":
    main()
