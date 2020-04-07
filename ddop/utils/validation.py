from sklearn.exceptions import NotFittedError
from inspect import isclass


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