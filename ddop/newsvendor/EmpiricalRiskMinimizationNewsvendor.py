import pulp
from ..utils.validation import check_cu_co
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from .base import BaseNewsvendor, DataDrivenMixin


class EmpiricalRiskMinimizationNewsvendor(BaseNewsvendor, DataDrivenMixin):
    """A Empirical Risk Minimization Newsvendor estimator

    Implements the Empirical Risk Minimization Method described in [1]

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable

    Attributes
    ----------
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs.
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    feature_weights_: array of shape (n_outputs, n_features)
        The calculated feature weights

     References
    ----------
    .. [1] Gah-Yi Ban, Cynthia Rudin, "The Big Data Newsvendor: Practical Insights from
    Machine Learning", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import EmpiricalRiskMinimizationNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = EmpiricalRiskMinimizationNewsvendor(cu=15, co=10)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, Y_test)
    [65.05263158]
    """

    def __init__(self, cu, co):
        super().__init__(
            cu=cu,
            co=co)

    def fit(self, X, y):
        """ Calculate the feature weights for estimator

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_outputs)
            The target values.

        Returns
        ----------
        self : EmpiricalRiskMinimizationNewsvendor
            Fitted estimator
        """
        X, y = self._validate_data(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Determine output settings
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        # Add intercept
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]
        n_features = X.shape[1]

        feature_weights = []

        # Define and solve LpProblem for each target variable
        # Then safe the calculated feature weights
        for k in range(self.n_outputs_):
            opt_model = pulp.LpProblem(sense=pulp.LpMinimize)
            n = np.arange(n_samples)
            p = np.arange(n_features)

            q = pulp.LpVariable.dicts('q', p)
            u = pulp.LpVariable.dicts('u', n, lowBound=0)
            o = pulp.LpVariable.dicts('o', n, lowBound=0)

            overage = pulp.LpAffineExpression([(u[i], self.cu_[k]) for i in n])
            underage = pulp.LpAffineExpression([(o[i], self.co_[k]) for i in n])

            objective = (underage + overage)/len(n)
            opt_model.setObjective(objective)

            for i in n:
                opt_model += u[i] >= y[i,k] - q[0] - sum([q[j] * X[i, j] for j in p if j != 0])
                opt_model += o[i] >= q[0] + sum([q[j] * X[i, j] for j in p if j != 0]) - y[i,k]
            opt_model.solve()

            feature_weights_yk = []
            for feature in q:
                feature_weights_yk += [q[feature].value()]

            feature_weights.append(feature_weights_yk)

        self.feature_weights_ = np.array(feature_weights)

        return self

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict"""
        X = check_array(X)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must match the input. "
                             "Model n_features is %s and input n_features is %s "
                             % (self.n_features_, n_features))
        return X

    def predict(self, X):
        """Predict value for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        ----------
        y : array-like of shape (n_samples, n_outputs)
            The predicted values
        """

        check_is_fitted(self)
        X = self._validate_X_predict(X)

        # Add intercept
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]

        pred = []
        for weights in self.feature_weights_:
            pred.append(X.dot(weights))

        pred = np.array(pred).T

        return pred


