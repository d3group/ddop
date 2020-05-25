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

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict"""
        X = check_array(X)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self.n_features_, n_features))

        return X
