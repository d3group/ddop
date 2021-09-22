from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ._base import BaseNewsvendor, ClassicMixin
from ..utils.validation import check_cu_co
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_absolute_error
import numpy as np


class ExponentialSmoothingNewsvendor(BaseNewsvendor, ClassicMixin):
    """
    A SEO newsvendor model with exponential smoothing as underlying forecaster

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    trend : {"add", "mul", "additive", "multiplicative", None}, default=None
        Type of trend component.
    damped_trend : bool, optional (default=None)
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, default=None
        Type of seasonal component.
    seasonal_periods : int, default=None
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.
    initialization_method : str, optional
        Method for initialize the recursions. One of:
        * 'estimated'
        * 'heuristic'
        * 'legacy-heuristic'
        * 'known'
        If 'known' initialization is used, then `initial_level`
        must be passed, as well as `initial_trend` and `initial_seasonal` if
        applicable. Default is 'estimated'.
    initial_level : float, optional
        The initial level component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_trend : float, optional
        The initial trend component. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    initial_seasonal : array_like, optional
        The initial seasonal component. An array of length `seasonal`
        or length `seasonal - 1` (in which case the last initial value
        is computed to make the average effect zero). Only used if
        initialization is 'known'. Required if estimation method is "known".
        If set using either "estimated" or "heuristic" this value is used.
        This allows one or more of the initial values to be set while
        deferring to the heuristic for others or estimating the unset
        parameters.
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use lambda equal to float.
    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    smoothing_trend : float, optional
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    smoothing_seasonal : float, optional
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    optimized : bool, optional
        Estimate model parameters by maximizing the log-likelihood
    remove_bias : bool, optional
        Remove bias from forecast values and fitted values by enforcing
        that the average residual is equal to zero.
    method : str, default "L-BFGS-B"
        The minimizer used. Valid options are "L-BFGS-B" (default), "TNC",
        "SLSQP", "Powell", "trust-constr", "basinhopping" (also "bh") and
        "least_squares" (also "ls"). basinhopping tries multiple starting
        values in an attempt to find a global minimizer in non-convex
        problems, and so is slower than the others.

    Attributes
    ----------
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.

    Notes
    -----

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import ExponentialSmoothingNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> Y_train, Y_test = train_test_split(Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = ExponentialSmoothingNewsvendor(cu, co, 'add', False, 'add',7)
    >>> mdl.fit(Y_train)
    >>> mdl.score(Y_test)
    TODO: ADD SCORE
    """

    def __init__(
            self,
            cu=None,
            co=None,
            trend=None,
            damped_trend=False,
            seasonal=None,
            seasonal_periods=None,
            initialization_method="estimated",
            initial_level=None,
            initial_trend=None,
            initial_seasonal=None,
            use_boxcox=False,
            smoothing_level=None,
            smoothing_trend=None,
            smoothing_seasonal=None,
            optimized=True,
            remove_bias=False,
            method="L-BFGS-B"
    ):
        self.cu = cu
        self.co = co
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.use_boxcox = use_boxcox
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.optimized = optimized
        self.remove_bias = remove_bias
        self.method = method
        super().__init__(
            cu=cu,
            co=co)

    def fit(self, y, X=None):
        """Fit the model to the training data y.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_outputs)
            The target values.
        X : array-like of shape (n_samples, n_features), optional (default=None)
            Exogenous variables are ignored

        Returns
        ----------
        self : HoltWintersNewsvendor
            Fitted estimator
        """

        y = check_array(y, ensure_2d=False)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.X_ = X
        self.y_ = y
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        forecasters = [ExponentialSmoothing(
            endog=y[:, i],
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method,
            initial_level=self.initial_level,
            initial_trend=self.initial_trend,
            initial_seasonal=self.initial_seasonal,
            use_boxcox=self.use_boxcox) for i in range(self.n_outputs_)]

        fitted_forecasters = [forecasters[i].fit(
            smoothing_level=self.smoothing_level,
            smoothing_trend=self.smoothing_trend,
            smoothing_seasonal=self.smoothing_seasonal,
            optimized=self.optimized,
            remove_bias=self.remove_bias,
            method=self.method) for i in range(self.n_outputs_)]

        error = np.array([y[:, i] - fitted_forecasters[i].fittedvalues for i in range(self.n_outputs_)])
        error_mean = error.mean(axis=1)
        error_std = error.std(axis=1)

        mae = np.array([mean_absolute_error(y[:, i], fitted_forecasters[i].fittedvalues)
                        for i in range(self.n_outputs_)])

        self.safety_buffer_mae = np.array(
            [1.25 * mae[i] * norm.ppf(self.cu_[i] / (self.co_[i] + self.cu_[i])) for i in range(self.n_outputs_)])

        self.safety_buffer_ = np.array(
            [norm(error_mean[i], error_std[i]).ppf(self.cu_[i] / (self.co_[i] + self.cu_[i])) for i in
             range(self.n_outputs_)])

        self.forecasters_ = fitted_forecasters

        return self

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

        forecasts = np.array([forecaster.forecast(n_steps) for forecaster in self.forecasters_]).T
        pred = forecasts + self.safety_buffer_

        return pred
