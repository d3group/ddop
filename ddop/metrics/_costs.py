from ..utils.validation import check_cu_co
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_array
import numpy as np


def _check_newsvendor_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same newsvendor task

    Parameters
    ----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted values

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


def _multiply_cost_weights(x, cu, co):
    if x < 0:
        # underage costs.
        return abs(x) * cu
    else:
        # overage costs
        return x * co


def pairwise_costs(y_true, y_pred, cu, co):
    """ Compute pairwise costs based on the the difference between y_true and y_pred
    and the given underage and overage costs.

    Parameters
    ----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted vales
    cu : int or float
        the underage costs per unit.
    co : int or float
        the overage costs per unit.

    Returns
    -------
    costs : ndarray of shape (n_samples, n_outputs)

    Examples
    --------
    >>> from ddop.metrics import pairwise_costs
    >>> y_true = [[2,2], [2,4], [3,6]]
    >>> y_pred = [[1,2], [3,3], [4,7]]
    >>> cu = [2,4]
    >>> co = [1,1]
    >>> pairwise_costs(y_true, y_pred, cu, co)
    array([[2., 0.],
           [1., 4.]
           [1., 1.])
    """
    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    y_diff = y_pred - y_true
    func = np.vectorize(_multiply_cost_weights)
    cu, co = check_cu_co(cu, co, y_true.shape[1])
    costs = func(y_diff, cu, co)
    return costs


def total_costs(y_true, y_pred, cu, co, multioutput="cumulated"):
    """ Compute total costs based on the the difference between y_true and y_pred
    and the given underage and overage costs.

    Parameters
    ----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted vales
    cu : int or float
        the underage costs per unit.
    co : int or float
        the overage costs per unit.
    multioutput: {"raw_values", "cumulated"}, default="cumulated"
        Defines aggregating of multiple output values. Default is "cumulated".
         'raw_values' :
            Returns a full set of cost values in case of multioutput input.
         'cumulated' :
            Costs of all outputs are cumulated.
    Returns
    -------
    total_costs :  float or ndarray of floats
        The total costs. If multioutput is ‘raw_values’, then the total costs are returned for each
        output separately. If multioutput is ‘cumulated’, then the cumulated costs of all outputs
        is returned. The total costs are non-negative floating points. The best value is 0.0.

    Examples
    --------
    >>> from ddop.metrics import total_costs
    >>> y_true = [[2,2], [2,4], [3,6]]
    >>> y_pred = [[1,2], [3,3], [4,7]]
    >>> cu = [2,4]
    >>> co = [1,1]
    >>> total_costs(y_true, y_pred, cu, co, multioutput="raw_values")
    array([4,5])
    >>> total_costs(y_true, y_pred, cu, co, multioutput="cumulated")
    9
    """
    costs = pairwise_costs(y_true, y_pred, cu, co)
    total_costs = np.sum(costs, axis=0)

    if multioutput == "raw_values":
        return total_costs

    return np.sum(total_costs)


def average_costs(y_true, y_pred, cu, co, multioutput="uniform_average"):
    """ Compute average costs based on the the difference between y_true and y_pred
    and the given underage and overage costs.

    Parameters
    ----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted vales
    cu : int or float
        the underage costs per unit.
    co : int or float
        the overage costs per unit.
    multioutput: {"raw_values", "uniform_average"}, default="raw_values"
        Defines aggregating of multiple output values. Default is "raw_values".
         'raw_values' :
            Returns a full set of cost values in case of multioutput input.
         'uniform_average' :
            Costs of all outputs are averaged with uniform weight.

    Returns
    -------
    costs :  float or ndarray of floats
        The average costs. If multioutput is ‘raw_values’, then the average costs are returned for each
        output separately. If multioutput is ‘uniform_average’, then the average of all output costs is
        returned. The average costs are non-negative floating points. The best value is 0.0.

    Examples
    --------
    >>> from ddop.metrics import average_costs
    >>> y_true = [[2,2], [2,4], [3,6]]
    >>> y_pred = [[1,2], [3,3], [4,7]]
    >>> cu = [2,4]
    >>> co = [1,1]
    >>> average_costs(y_true, y_pred, cu, co, multioutput="raw_values")
    array([1.33.., 1.66..])
    >>> average_costs(y_true, y_pred, cu, co, multioutput="uniform_average")
    1.5
    """
    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    total = total_costs(y_true, y_pred, cu, co, multioutput="raw_values")
    average_costs = total / y_true.shape[0]

    if multioutput == "raw_values":
        return average_costs

    return np.average(average_costs)


def prescriptiveness_score(y_true, y_pred, y_pred_saa, cu, co, multioutput="uniform_average"):
    """ Compute the coefficient of prescriptiveness that is defined as (1 - u/v), where u are the average
    costs between the true and predicted values (y_true,y_pred), and v are the average costs between the
    true values and the predictions obtained by SAA (y_pred_saa, y_pred). The best possible score
    is 1.0 and it can be negative (because the model can be arbitrarily worse).

    Parameters
    ----------
    y_true : array-like
        The true values
    y_pred : array-like
        The predicted vales
    cu : int or float
        the underage costs per unit.
    co : int or float
        the overage costs per unit.
    multioutput: {"raw_values", "uniform_average"}, default="raw_values"
        Defines aggregating of multiple output scores. Default is “uniform_average”.
         'raw_values' :
            Returns a full set of scores in case of multioutput input.
         'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

    Returns
    -------
    score :  float or ndarray of floats
        The prescriptiveness score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Examples
    --------
    >>> from ddop.metrics import prescriptiveness_score
    >>> y_true = [[2,2], [2,4], [3,6]]
    >>> y_pred = [[1,2], [3,3], [4,7]]
    >>> y_pred_saa = [[4,5],[4,5],[4,5]]
    >>> cu = [2,4]
    >>> co = [1,1]
    >>> prescriptiveness_score(y_true, y_pred, cu, co, multioutput="raw_values")
    array([0.2, 0.375])
    >>> prescriptiveness_score(y_true, y_pred, cu, co, multioutput="uniform_average")
    0.2875
    """

    y_true, y_pred = _check_newsvendor_targets(y_true, y_pred)
    y_true, y_pred_saa = _check_newsvendor_targets(y_true, y_pred_saa)

    numerator = average_costs(y_true, y_pred, cu, co)
    denominator = average_costs(y_true, y_pred_saa, cu, co)

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring anyway
    # output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    if multioutput == "raw_values":
        return output_scores

    return np.average(output_scores)
