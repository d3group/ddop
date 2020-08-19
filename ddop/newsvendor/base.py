from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from abc import ABC, abstractmethod
from ..metrics.costs import calc_avg_costs
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
    def score(self, X, y):
        y_pred = self.predict(X)
        score = calc_avg_costs(y, y_pred, self.cu_, self.co_)
        return score


class ClassicMixin:
    def score(self, y, X=None):

        y = check_array(y, ensure_2d=False)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        y_pred = self.predict(y.shape[0])
        return calc_avg_costs(y, y_pred, self.cu_, self.co_)
