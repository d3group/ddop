from sklearn.metrics._scorer import _BaseScorer
from ..newsvendor import _SampleAverageApproximationNewsvendor as SAA
import inspect
import numpy as np


class _Scorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = method_caller(estimator, "predict", X)
        cu = estimator.cu_
        co = estimator.co_

        if "y_pred_saa" in inspect.getfullargspec(self._score_func).args:
            X = estimator.X_
            y = estimator.y_
            y_pred_saa = SAA.SampleAverageApproximationNewsvendor(cu, co).fit(y_true).predict().flatten()
            y_pred_saa = np.full(y_true.shape, y_pred_saa)
            return self._sign * self._score_func(y_true, y_pred, y_pred_saa, cu, co, **self._kwargs)

        else:
            print("else")
            return self._sign * self._score_func(y_true, y_pred, cu, co, **self._kwargs)


def make_scorer(score_func, greater_is_better=True, **kwargs):
    """Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in
    `sklearn.model_selection.GridSearchCV` and
    `sklearn.model_selection.cross_val_score`.
    It takes a score function from `ddop.metrics`, such as `ddop.metrics.total_costs`,
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling.

    Parameters
    ----------
    score_func : callable
        Score function included in ddop.metrics.
    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1

    cls = _Scorer

    return cls(score_func, sign, kwargs)