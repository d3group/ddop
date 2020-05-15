from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from inspect import isclass
import numpy as np
import numbers


def check_is_fitted(estimator):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError.
    This utility is meant to be used internally by estimators themselves,
    typically in their own predict methods.

    This function is based on check_is_fitted from sklearn.utils.validation
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """

    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this estimator.")

    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def check_cu_co(cu, co, n_outputs):
    """Validate under- and overage costs.

    Code was inspired from [1]

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

     References
    ----------
    .. [1] scikit-learn, _check_sample-weight(),
           <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py>
    """
    costs = [[cu,"cu"], [co,"co"]]
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
                raise ValueError(c[1],"must be 1D array or scalar")

            if cost.shape != (n_outputs,):
                raise ValueError("{}.shape == {}, expected {}!"
                                .format(c[1], cost.shape, (n_outputs,)))
        costs_validated.append(cost)
    cu = costs_validated[0]
    co = costs_validated[1]
    return cu, co
