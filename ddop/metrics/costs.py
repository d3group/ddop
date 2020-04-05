from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import _num_samples
import numpy as np


def cost(cp, ch, Y_pred, Y_true):
    if Y_pred > Y_true:
        cost = (Y_pred - Y_true) * ch
    else:
        cost = (Y_true - Y_pred) * cp
    return cost


def costs(cp, ch, Y_pred, Y_true):
    check_consistent_length(Y_pred, Y_true)
    Y_true = np.asanyarray(Y_true)
    costs = []
    length = len(Y_pred)
    for i in range(length):
        costs.append(cost(cp, ch, Y_pred[i], Y_true[i]))
    return costs


def total_costs(cp, ch, Y_pred, Y_true):
    check_consistent_length(Y_pred, Y_true)
    Y_true = np.asanyarray(Y_true)
    totalCosts = 0
    length = len(Y_pred)
    for i in range(length):
        totalCosts += cost(cp, ch, Y_pred[i], Y_true[i])
    return totalCosts


def avg_costs(cp, ch, Y_pred, Y_true):
    totalCosts = total_costs(cp, ch, Y_pred, Y_true)
    avgCosts = totalCosts/len(Y_pred)
    return avgCosts