from sklearn.exceptions import NotFittedError


def check_is_fitted(estimator):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the message: "This %(name)s instance is not fitted
    yet. Call 'fit' with appropriate arguments before using this
    estimator.".
    This utility is meant to be used internally by estimators themselves,
    typically in their own predict methods.

    Parameters
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
    # nochmal schauen wozu
    #if isclass(estimator):
    #    raise TypeError("{} is a class, not an instance.".format(estimator))

    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this estimator.")

    attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})