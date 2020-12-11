from sklearn.utils.validation import check_array
import numpy as np
import numbers


def check_cu_co(cu, co, n_outputs):
    """Validate under- and overage costs.

    Parameters
    ----------
    cu : {ndarray, Number or None}, shape (n_outputs,)
       The underage costs per unit. Passing cu=None will output an array of ones.
    co : {ndarray, Number or None}, shape (n_outputs,)
       The overage costs per unit. Passing co=None will output an array of ones.
    n_outputs : int
       The number of outputs.
    Returns
    -------
    cu : ndarray, shape (n_outputs,)
       Validated underage costs. It is guaranteed to be "C" contiguous.
    co : ndarray, shape (n_outputs,)
       Validated overage costs. It is guaranteed to be "C" contiguous.
    """
    costs = [[cu, "cu"], [co, "co"]]
    costs_validated = []
    for c in costs:
        if c[0] is None:
            cost = np.ones(n_outputs, dtype=np.float64)
        elif isinstance(c[0], numbers.Number):
            cost = np.full(n_outputs, c[0], dtype=np.float64)
        else:
            cost = check_array(
                c[0], accept_sparse=False, ensure_2d=False, dtype=np.float64,
                order="C"
            )
            if cost.ndim != 1:
                raise ValueError(c[1], "must be 1D array or scalar")

            if cost.shape != (n_outputs,):
                raise ValueError("{}.shape == {}, expected {}!"
                                 .format(c[1], cost.shape, (n_outputs,)))
        costs_validated.append(cost)
    cu = costs_validated[0]
    co = costs_validated[1]
    return cu, co


def formate_hyperparameter(value, name, n_outputs):
    # make sure value is an array of shape (n_outputs,)
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = np.full((n_outputs,), value)
    value = np.array(value)

    if value.ndim != 1:
        raise ValueError("%s must be 1D array or scalar" % (name))
    return value
