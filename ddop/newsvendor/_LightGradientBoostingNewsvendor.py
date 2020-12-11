from ._base import BaseNewsvendor, DataDrivenMixin
from ..utils.validation import check_cu_co
import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.utils.validation import check_is_fitted


def _create_objective(cu, co):
    """Create a newsvendor-like objective function with the given under- and overage costs"""

    def nv_objective(y_true, y_pred):
        residual = (y_true - y_pred).astype('float')
        grad = np.where(residual < 0, 2 * (co ** 2) * (y_pred - y_true), 2 * (cu ** 2) * (y_pred - y_true))
        hess = np.where(residual < 0, 2 * (co ** 2), 2 * (cu ** 2))
        return grad, hess

    return nv_objective


def _nv_eval_metric(cu, co):
    """Create a newsvendor evaluation metric with the given under- and overage costs"""

    def custom_eval_metric(y_true, y_pred):
        residual = (y_true - y_pred).astype('float')
        loss = np.where(residual < 0, (co * (y_true - y_pred)) ** 2, (cu * (y_true - y_pred)) ** 2)
        return "custom_asymmetric_eval", np.mean(loss), False

    return custom_eval_metric


class LightGradientBoostingNewsvendor(BaseNewsvendor, DataDrivenMixin):
    """A newsvendor based on light gradient boosting.

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    boosting_type : string, optional (default='gbdt')
        'gbdt', traditional Gradient Boosting Decision Tree.
        'dart', Dropouts meet Multiple Additive Regression Trees.
        'goss', Gradient-based One-Side Sampling.
        'rf', Random Forest.
    num_leaves : int, optional (default=31)
        Maximum tree leaves for base learners.
    max_depth : int, optional (default=-1)
        Maximum tree depth for base learners, <=0 means no limit.
    learning_rate : float, optional (default=0.1)
        Boosting learning rate.
        You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
        in training using ``reset_parameter`` callback.
        Note, that this will ignore the ``learning_rate`` argument in training.
    n_estimators : int, optional (default=100)
        Number of boosted trees to fit.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The model will train until the validation score stops improving.
        Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue training.
        Requires at least one validation data and one metric.
        If there's more than one, will check all of them. But the training data is ignored anyway.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True``
        in additional parameters ``**kwargs`` of the model constructor.
    subsample_for_bin : int, optional (default=200000)
        Number of samples for constructing bins.
    min_split_gain : float, optional (default=0.)
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : float, optional (default=1e-3)
        Minimum sum of instance weight (hessian) needed in a child (leaf).
    min_child_samples : int, optional (default=20)
        Minimum number of data needed in a child (leaf).
    subsample : float, optional (default=1.)
        Subsample ratio of the training instance.
    subsample_freq : int, optional (default=0)
        Frequence of subsample, <=0 means no enable.
    colsample_bytree : float, optional (default=1.)
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, optional (default=0.)
        L1 regularization term on weights.
    reg_lambda : float, optional (default=0.)
        L2 regularization term on weights.
    random_state : int, RandomState object or None, optional (default=None)
        Random number seed.
        If int, this number is used to seed the C++ code.
        If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
        If None, default seeds in C++ code are used.
    n_jobs : int, optional (default=-1)
        Number of parallel threads.
    silent : bool, optional (default=True)
        Whether to print messages while running boosting.
    importance_type : string, optional (default='split')
        The type of feature importance to be filled into ``feature_importances_``.
        If 'split', result contains numbers of times the feature is used in a model.
        If 'gain', result contains total gains of splits which use the feature.

    Attributes
    ----------
    estimators_ : list of ``n_output`` LightGradientBoostingNewsvendor
        The collection of fitted estimators used for predictions
    n_features_ : int
        The number of features of fitted model.
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import LightGradientBoostingNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = LightGradientBoostingNewsvendor(cu,co)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, Y_test)
    TODO: ADD SCORE
    """

    def __init__(self,
                 cu=None,
                 co=None,
                 boosting_type='gbdt',
                 num_leaves=31,
                 max_depth=-1,
                 learning_rate=0.1,
                 n_estimators=100,
                 early_stopping_rounds=None,
                 subsample_for_bin=200000,
                 min_split_gain=0.,
                 min_child_weight=1e-3,
                 min_child_samples=20,
                 subsample=1.,
                 subsample_freq=0,
                 colsample_bytree=1.,
                 reg_alpha=0.,
                 reg_lambda=0.,
                 random_state=None,
                 n_jobs=-1,
                 silent=True,
                 importance_type='split'):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        super().__init__(
            cu=cu,
            co=co)

    def _create_estimator(self, cu, co):
        """Create LGBMRegressor with a newsvendor-like objective function"""
        objective = _create_objective(cu, co)
        estimator = LGBMRegressor(boosting_type=self.boosting_type,
                                  num_leaves=self.num_leaves,
                                  max_depth=self.max_depth,
                                  learning_rate=self.learning_rate,
                                  n_estimators=self.n_estimators,
                                  subsample_for_bin=self.subsample_for_bin,
                                  objective=objective,
                                  min_split_gain=self.min_split_gain,
                                  min_child_weight=self.min_child_weight,
                                  min_child_samples=self.min_child_samples,
                                  subsample=self.subsample,
                                  subsample_freq=self.subsample_freq,
                                  colsample_bytree=self.colsample_bytree,
                                  reg_alpha=self.reg_alpha,
                                  reg_lambda=self.reg_lambda,
                                  random_state=self.random_state,
                                  n_jobs=self.n_jobs,
                                  silent=self.silent,
                                  importance_type=self.importance_type)
        return estimator

    def _get_fitted_estimator(self, X, y, sample_weight, cu, co,):
        """fit estimator with a newsvendor evaluation metric"""
        estimator = self._create_estimator(cu, co)
        eval_metric = _nv_eval_metric(cu, co)
        estimator.fit(X=X, y=y, sample_weight=sample_weight, eval_metric=eval_metric)
        return estimator

    def fit(self, X, y, sample_weight=None):
        """Fit model with training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_outputs)
            The target values.
        sample_weight : array-like of shape = [n_samples] or None, optional (default=None)
            Weights of training data.

        Returns
        -------
        self : LightGradientBoostingNewsvendor
            Fitted estimator.

        Notes
        -------
        For multi-output, fit a separate model for each output variable.
        """

        X, y = self._validate_data(X, y, multi_output=True, accept_sparse=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.X_ = X
        self.y_ = y
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        # Create list of fitted estimators.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_fitted_estimator)(X, y[:, i], sample_weight, self.cu_[i], self.co_[i])
            for i in range(self.n_outputs_))

        return self

    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples, n_outputs)
            The predicted values.

        Notes
        ----
        For multi-output, targets are predicted across multiple predictors.
        """
        check_is_fitted(self)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X)
            for e in self.estimators_)
        return np.asarray(y).T
