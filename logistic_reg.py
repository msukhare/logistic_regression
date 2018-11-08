# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logistic_reg.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/06/04 09:03:39 by msukhare          #+#    #+#              #
#    Updated: 2018/11/08 13:14:41 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from metrics_for_binary_classification import metrics_for_binary_classification

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("file doesn't exit")
    data.insert(0, '0', 1)
    tmp_data = data.sample(frac=1, random_state=3)
    col = tmp_data.shape[1]
    X = tmp_data.iloc[:, 0 : col - 1]
    Y = tmp_data.iloc[:, col - 1 :]
    X = np.array(X.values, dtype=float)
    Y = np.array(Y.values, dtype=float)
    return (tmp_data, X, Y)

def get_max_min(tab, j, row, max_min):
    to_ret = tab[0][j]
    i = 1
    while (i < row):
        if (max_min == 0 and tab[i][j] < to_ret):
            to_ret = tab[i][j]
        elif (max_min == 1 and tab[i][j] > to_ret):
            to_ret = tab[i][j]
        i += 1
    return (to_ret)

def scale_mat(tab, row, col):
    j = 1
    while (j < col):
        i = 0
        max_val = get_max_min(tab, j, row, 1)
        min_val = get_max_min(tab, j, row, 0)
        while (i < row):
            tab[i][j] = (tab[i][j] - min_val) / (max_val - min_val)
            i += 1
        j += 1

def split_array(tab, row):
    to_ret_train = tab[0 : math.floor(row * 0.70)]
    to_ret_costfct = tab[math.floor(row * 0.70) : math.floor(row * 0.85)]
    to_ret_try = tab[math.floor(row * 0.85): ]
    return (to_ret_train, to_ret_costfct, to_ret_try)

def hypo(tab, i, thetas):
    return (1 / (1 + np.exp(-thetas.transpose().dot(tab[i]))))

def cost_fct(thetas, X, Y, col, row):
    result = 0
    for i in range(int(row)):
        if (Y[i] == 1):
            result += np.log(hypo(X, i, thetas))
        else:
            result += np.log((1 - hypo(X, i, thetas)))
    return (-(1 / row) * result)

def get_somme(X, Y, row, thetas, j):
    result = 0
    for i in range(row):
        result += (hypo(X, i, thetas) - Y[i][0]) * X[i][j]
    return (result);

def gradient_descent(thetas, tmp, X, Y, row, col):
    alpha = 0.03
    for j in range(int(col)):
        tmp[j][0] = thetas[j][0] - ((alpha / row) * get_somme(X, Y, row, thetas, j))
    for j in range(int(col)):
        thetas[j][0] = tmp[j][0]

def train_thetas(thetas, tmp, col, row, X_train, Y_train, X_test, Y_test):
    test_cost = []
    train_cost = []
    tmp_iter = []
    for i in range(10000):
        gradient_descent(thetas, tmp, X_train, Y_train, math.floor(row * 0.70), col)
        res = cost_fct(thetas, X_test, Y_test, col, math.floor(row * 0.15))
        res_train = cost_fct(thetas, X_train, Y_train, col, math.floor(row * 0.70))
        train_cost.append(res_train)
        test_cost.append(res)
        tmp_iter.append(i)
    plt.plot(tmp_iter, train_cost)
    plt.plot(tmp_iter, test_cost)
    plt.show()

def get_pred_Y(X, thetas, Y):
    pred_Y = np.zeros((X.shape[0], 1), dtype=float)
    for i in range(int(X.shape[0])):
        pred_Y[i][0] = hypo(X, i, thetas)
        print("Y=", Y[i][0], "pred_Y=", pred_Y[i][0])
    return (pred_Y)

def write_thetas_in_file(thetas):
    try:
        new_file = csv.writer(open("thetas.csv", "w"))
    except:
        sys.exit("fail to create file")
    for i in range(int(thetas.shape[0])):
        new_file.writerow([str(thetas[i][0])])

def check_argv():
    if (len(sys.argv) != 2):
        sys.exit("usage: python3 [fileWithData.csv]")

def main():
    check_argv()
    data, X, Y = read_file()
    row = X.shape[0]
    col = X.shape[1]
    thetas = np.zeros((col, 1), dtype=float)
    tmp_the = np.zeros((col, 1), dtype=float)
    scale_mat(X, row, col)
    X_train, X_test, X_validation = split_array(X, row)
    Y_train, Y_test, Y_validation = split_array(Y, row)
    train_thetas(thetas, tmp_the, col, row, X_train, Y_train, X_test, Y_test)
    get_metrics = metrics_for_binary_classification()
    get_metrics.confused_matrix_sigmoid(get_pred_Y(X_validation, thetas, Y_validation), Y_validation, 1)
    write_thetas_in_file(thetas)

if __name__ == "__main__":
    main()
