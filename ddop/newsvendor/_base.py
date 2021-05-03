from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from abc import ABC, abstractmethod
from ..metrics._costs import average_costs
import numpy as np


class BaseNewsvendor(BaseEstimator, ABC):
    """
    Base class for newsvendor.
    """
    @abstractmethod
    def __init__(self, cu, co):
        self.cu = cu
        self.co = co


class DataDrivenMixin:
    _estimator_type = "regressor"
    def score(self, X, y):
        """
        Return the average costs of the prediction

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples, n_outputs)
            The true values for x.

        Returns
        -------
         score: float
            The average costs
        """

        y_pred = self.predict(X)
        score = -average_costs(y, y_pred, self.cu_, self.co_, multioutput="uniform_average")
        return score


class ClassicMixin:
    _estimator_type = "regressor"
    def score(self, y, X=None):
        """
        Return the average costs of the prediction

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Exogenous variables are ignored
        y : array-like of shape (n_samples, n_outputs)
            The true values.

        Returns
        -------
         score: float
            The average costs
        """

        y = check_array(y, ensure_2d=False)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        y_pred = self.predict(y.shape[0])
        return -average_costs(y, y_pred, self.cu_, self.co_, multioutput="uniform_average")
