from ._base import BaseNewsvendor, DataDrivenMixin
from ..utils.validation import check_cu_co
import numpy as np
import pulp
from sklearn.utils.validation import check_array, check_is_fitted


class LinearRegressionNewsvendor(BaseNewsvendor, DataDrivenMixin):
    """A linear regression model to solve the Newsvendor problem

    This class implements the approach described in [1].

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
    .. [1] Gah-Yi Ban, Cynthia Rudin, "The Big Data Newsvendor: Practical Insights
        from Machine Learning", 2018.

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import LinearRegressionNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = LinearRegressionNewsvendor(cu=15, co=10)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, Y_test)
    TODO: ADD SCORE
    """

    def __init__(self, cu=None, co=None):
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
        self : LinearRegressionNewsvendor
            Fitted estimator
        """
        X, y = self._validate_data(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Determine output settings
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        n_samples = X.shape[0]

        feature_weights = []
        intercept = []

        # Define and solve LpProblem for each target variable
        # Then safe the calculated feature weights
        for k in range(self.n_outputs_):
            opt_model = pulp.LpProblem(sense=pulp.LpMinimize)
            n = np.arange(n_samples)
            p = np.arange(self.n_features_)
            q0 = pulp.LpVariable('q0')
            q = pulp.LpVariable.dicts('q', p)
            u = pulp.LpVariable.dicts('u', n, lowBound=0)
            o = pulp.LpVariable.dicts('o', n, lowBound=0)

            overage = pulp.LpAffineExpression([(u[i], self.cu_[k]) for i in n])
            underage = pulp.LpAffineExpression([(o[i], self.co_[k]) for i in n])

            objective = (underage + overage)/len(n)
            opt_model.setObjective(objective)

            for i in n:
                opt_model += u[i] >= y[i,k] - q0 - sum([q[j] * X[i, j] for j in p])
                opt_model += o[i] >= q0 + sum([q[j] * X[i, j] for j in p]) - y[i,k]
            opt_model.solve()

            feature_weights_yk = []

            for feature in q:
                feature_weights_yk += [q[feature].value()]

            feature_weights.append(feature_weights_yk)
            intercept.append(q0.value())

        #make sure no weight is None
        feature_weights = np.array(feature_weights)
        feature_weights[feature_weights == None] = 0.0
        self.feature_weights_ = feature_weights
        self.intercept_ = intercept

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

        pred = X.dot(self.feature_weights_.T)+self.intercept_

        return pred


