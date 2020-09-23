from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import BaseNewsvendor, ClassicMixin
from ..utils.validation import check_cu_co, formate_hyperparameter
from sklearn.utils.validation import check_array
from sklearn.model_selection import train_test_split
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
    damped : bool, optional (default=None)
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, default=None
        Type of seasonal component.
    seasonal_periods : int, default=None
        The number of periods in a complete seasonal cycle, e.g., 4 for
        quarterly data or 7 for daily data with a weekly cycle.
    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    smoothing_slope : float, optional
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    smoothing_seasonal : float, optional
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    damping_slope : float, optional
        The phi value of the damped method, if the value is
        set then this value will be used as the value.
    optimized : bool, optional
        Estimate model parameters by maximizing the log-likelihood
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use lambda equal to float.
    remove_bias : bool, optional
        Remove bias from forecast values and fitted values by enforcing
        that the average residual is equal to zero.
    use_basinhopping : bool, optional
        Using Basin Hopping optimizer to find optimal values

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
            damped=False,
            seasonal=None,
            seasonal_periods=None,
            smoothing_level=None,
            smoothing_slope=None,
            smoothing_seasonal=None,
            damping_slope=None,
            optimized=True,
            use_boxcox=False,
            remove_bias=False,
            use_basinhopping=False
    ):
        self.cu = cu
        self.co = co
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.smoothing_slope = smoothing_slope
        self.smoothing_seasonal = smoothing_seasonal
        self.damping_slope = damping_slope
        self.use_boxcox = use_boxcox
        self.remove_bias = remove_bias
        self.use_basinhopping = use_basinhopping
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

        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        # Formate hyperparameter
        self.trend = formate_hyperparameter(self.trend, "trend", self.n_outputs_)
        self.damped = formate_hyperparameter(self.damped, "damped", self.n_outputs_)
        self.seasonal = formate_hyperparameter(self.seasonal, "seasonal", self.n_outputs_)
        self.seasonal_periods = formate_hyperparameter(self.seasonal_periods, "seasonal_periods", self.n_outputs_)
        self.smoothing_level = formate_hyperparameter(self.smoothing_level, "smoothing_level", self.n_outputs_)
        self.optimized = formate_hyperparameter(self.optimized, "optimized", self.n_outputs_)
        self.smoothing_slope = formate_hyperparameter(self.smoothing_slope, "smoothing_slope", self.n_outputs_)
        self.smoothing_seasonal = formate_hyperparameter(self.smoothing_seasonal, "smoothing_seasonal", self.n_outputs_)
        self.damping_slope = formate_hyperparameter(self.damping_slope, "damping_slope", self.n_outputs_)
        self.use_boxcox = formate_hyperparameter(self.use_boxcox, "use_boxcox", self.n_outputs_)
        self.remove_bias = formate_hyperparameter(self.remove_bias, "remove_bias", self.n_outputs_)
        self.use_basinhopping = formate_hyperparameter(self.use_basinhopping, "use_basinhopping", self.n_outputs_)

        forecasters = [ExponentialSmoothing(
            endog=y[:, i],
            trend=self.trend[i],
            damped=self.damped[i],
            seasonal=self.seasonal[i],
            seasonal_periods=self.seasonal_periods[i]) for i in range(self.n_outputs_)]

        fitted_forecasters = [forecasters[i].fit(
            smoothing_level=self.smoothing_level[i],
            optimized=self.optimized[i],
            smoothing_slope=self.smoothing_slope[i],
            smoothing_seasonal=self.smoothing_seasonal[i],
            damping_slope=self.damping_slope[i],
            use_boxcox=self.use_boxcox[i],
            remove_bias=self.remove_bias[i],
            use_basinhopping=self.use_basinhopping[i]) for i in range(self.n_outputs_)]

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
