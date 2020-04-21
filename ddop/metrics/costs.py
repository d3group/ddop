from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
import numpy as np


def _check_newsvendor_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same newsvendor task
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    y_true : array-like of shape (n_samples,n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,n_outputs)
        Estimated target values.
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype="numeric")
    y_pred = check_array(y_pred, ensure_2d=False, dtype="numeric")

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    return y_true, y_pred


def _multiply_cost_weights(x, cp, ch):
    if x < 0:
        # underage costs.
        return abs(x) * cp
    else:
        # overage costs
        return x * ch


def calc_costs(y_true, y_pred, cp, ch):
    """ Compute pairwise costs based on the the difference between y_true and y_pred
        and the given underage and overage costs.
        ----------
        y_true : array-like
        y_pred : array-like
        cp : int or float
            the underage costs per unit.
        cp : int or float
            the overage costs per unit.

        Returns
        -------
        costs : array of floating point values, one for each individual target.
        """
    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    y_diff = y_pred - y_true
    func = np.vectorize(_multiply_cost_weights)
    costs = func(y_diff, cp, ch)
    return costs


def calc_total_costs(y_true, y_pred, cp, ch):
    """ Compute total costs based on the the difference between y_true and y_pred
        and the given underage and overage costs.
        ----------
        y_true : array-like
        y_pred : array-like
        cp : int or float
            the underage costs per unit.
        cp : int or float
            the overage costs per unit.

        Returns
        -------
        total_costs : float
        """
    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    y_diff = y_pred - y_true
    func = np.vectorize(_multiply_cost_weights)
    costs = func(y_diff, cp, ch)
    total_costs = np.sum(costs, axis=0)
    return total_costs


def calc_avg_costs(y_true, y_pred, cp, ch):
    """ Compute average costs based on the the difference between y_true and y_pred
        and the given underage and overage costs.
        ----------
        y_true : array-like
        y_pred : array-like
        cp : int or float
            the underage costs per unit.
        cp : int or float
            the overage costs per unit.

        Returns
        -------
        avg_costs : float
        """
    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    y_diff = y_pred - y_true
    func = np.vectorize(_multiply_cost_weights)
    costs = func(y_diff, cp, ch)
    total_costs = np.sum(costs, axis=0)
    avg_costs = total_costs / costs.shape[0]
    return avg_costs
