from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from abc import ABC, abstractmethod
from ..metrics.costs import calc_avg_costs


class BaseNewsvendor(BaseEstimator, ABC):
    """
    Base class for newsvendor.
    """

    @abstractmethod
    def __init__(self, cu, co):
        self.cu = cu
        self.co = co

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        y_pred = self.predict(X)
        return calc_avg_costs(y, y_pred, self.cu_, self.co_)