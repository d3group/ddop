from ._base import BaseNewsvendor, ClassicMixin
from ..utils.validation import check_cu_co
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array


class SampleAverageApproximationNewsvendor(BaseNewsvendor, ClassicMixin):
    """ A sample average approximation model to solve the newsvendor problem

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
        The underage costs per unit. If None, then underage costs are one
        for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
        The overage costs per unit. If None, then overage costs are one
        for each target variable

    Attributes
    -----------
    X_ : array of shape (n_samples, n_features)
        The X training data
    y_ : array of shape (n_samples, n_outputs)
        The y training data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    n_samples_ : int
        The number of samples when ``fit`` is performed.

    Notes
    -----

    References
    ----------
    .. [1] Levi, Retsef, Georgia Perakis, and Joline Uichanco. "The data-driven newsvendor problem: new bounds and insights."
           Operations Research 63.6 (2015): 1294-1306.

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import SampleAverageApproximationNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = SampleAverageApproximationNewsvendor(cu, co)
    >>> mdl.fit(X_train, Y_train)
    >>> score(X_test, Y_test)
    TODO: Add output
    """

    def __init__(self,
                 cu=None,
                 co=None
                 ):
        self.cu = cu
        self.co = co

    def _calc_weights(self):
        weights = np.full(self.n_samples_, 1 / self.n_samples_)
        return weights

    def fit(self, y, X=None):
        """ Fit the estimator for training data y

        Parameters
        ----------
        y : array-like of shape (n_samples, n_features)
            The training target values.
        X : array-like of shape (n_samples, n_features), default=None
            Exogenous variables are ignored

        Returns
        ----------
        self : KNeighborsWeightedNewsvendor
            Fitted estimator
        """

        y = check_array(y, ensure_2d=False, accept_sparse='csr')

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Training data
        self.y_ = y
        self.n_samples_ = y.shape[0]

        # Determine output settings
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        return self

    def _findQ(self, weights):
        """Calculate the optimal order quantity q"""

        y = self.y_
        q = []

        for k in range(self.n_outputs_):
            alpha = self.cu_[k] / (self.cu_[k] + self.co_[k])
            q.append(np.quantile(y[:, k], alpha, interpolation="higher"))

        return q

    def predict(self, n_steps=1):
        """Predict n time-steps

        Parameters
        ----------
        n_steps : int, default=1
            The number of steps to predict ahead

        Returns
        ----------
        y : array-like of shape (n, n_outputs)
            The predicted values
        """

        check_is_fitted(self)
        weights = self._calc_weights()
        pred = self._findQ(weights)
        pred = np.full((n_steps, self.n_outputs_), pred)

        return pred
