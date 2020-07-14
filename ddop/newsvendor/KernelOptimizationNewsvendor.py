from .base import BaseNewsvendor
from ..utils.validation import check_cu_co
from ..utils.kernels import Kernel
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.spatial import distance_matrix
from .base import DataDrivenMixin


class KernelOptimizationNewsvendor(BaseNewsvendor, DataDrivenMixin):
    """A Kernel Optimization Newsvendor estimator

    Implements the Kernel Optimization Method described in [1]

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    kernel_type:  {"uniform", "gaussian"}, default="uniform"
        The type of the kernel function
    kernel_bandwidth: float or int, default=1
        The bandwidth of the kernel function

    Attributes
    ----------
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs
    X_ : array of shape (n_samples, n_features)
        The historic X-data
    y_ : array of shape (n_samples, n_outputs)
        The historic y-data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    kernel_ : Kernel
        The Kernel used for prediction

    References
    ----------
    .. [1] Gah-Yi Ban, Cynthia Rudin, "The Big Data Newsvendor: Practical Insights from
    Machine Learning", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import KernelOptimizationNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 2,1
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    >>> mdl = KernelOptimizationNewsvendor(cu, co)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test,Y_test)
    [80.86842105]
    """

    def __init__(self, cu, co, kernel_type="gaussian", kernel_bandwidth=1):
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth
        super().__init__(
            cu=cu,
            co=co)

    def fit(self, X, y):
        """ Fit the estimator and save the historic X- and y-data needed
        for the prediction method

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.

        Returns
        ----------
        self : KernelOptimizationNewsvendor
            Fitted estimator
        """
        X, y = self._validate_data(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Historic data for predict method
        self.X_ = X
        self.y_ = y

        # Determine output settings
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        # Generate Kernel
        self.kernel_ = Kernel(self.kernel_type, self.kernel_bandwidth)

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
        pred = []

        for k in range(self.n_outputs_):
            # concat X and yk to sort data by yk. Then split again.
            data = np.append(self.X_, self.y_[:, k].reshape(-1, 1), axis=1)
            data = data[np.argsort(data[:, self.n_features_])]
            X_hist = data[:, np.r_[0:self.n_features_]]
            y_hist = data[:, self.n_features_]
            pred_yk = []
            for row in X:
                distances = distance_matrix(X_hist, [row]).ravel()
                distances_kernel_weighted = np.array([self.kernel_.get_kernel_output(x) for x in distances])
                K_sum_total = np.sum(distances_kernel_weighted)
                K_sum = 0
                last_value = 0
                for index, value in np.ndenumerate(y_hist):
                    i = index[0]
                    if i < 1:
                        if i == y_hist.shape[0] - 1:
                            pred_yk += [value]

                        else:
                            K_sum += distances_kernel_weighted[i]
                            last_value = value
                            continue

                    elif value != last_value:
                        tf_lhs = K_sum / K_sum_total
                        tf_rhs = self.cu_[k] / (self.cu_[k] + self.co_[k])
                        if tf_lhs >= tf_rhs:
                            pred_yk += [last_value]
                            break
                        elif i==y_hist.shape[0]-1:
                            pred_yk += [value]
                        else:
                            K_sum += distances_kernel_weighted[i]
                            last_value = value

                    else:
                        if i==y_hist.shape[0]-1:
                            pred_yk += [value]
                        else:
                            K_sum += distances_kernel_weighted[i]

            pred.append(pred_yk)
        return np.asarray(pred).T
