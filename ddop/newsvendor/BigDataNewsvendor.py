import pulp
from ..utils.validation import check_is_fitted
from ..utils.kernels import Kernel
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from scipy.spatial import distance_matrix


class ERM:
    """A ERM (Empirical Risk Minimization Algorithms) newsvendor estimator
    Parameters
    ----------
    cp : float or int, default=None
        the overage costs per unit.
    ch : float or int, default=None
        the underage costs per unit:

     Attributes
    ----------
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs.
    feature_weights_: array of shape (n_outputs, n_features)
        The calculated feature weights

     References
    ----------
    .. [1] Gah-Yi Ban, Cynthia Rudin, "The Big Data Newsvendor: Practical Insights from
    Machine Learning", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import ERM
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cp,ch = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = ERM(cp=15, ch=10)
    >>> mdl.fit(X_train, Y_train)
    >>> y_pred = mdl.predict(X_test)
    >>> calc_avg_costs(Y_test, y_pred, cp, ch)
    48.85
    """

    def __init__(self, cp, ch):
        self.cp = cp
        self.ch = ch

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Determine output settings
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Add intercept
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]
        n_features = X.shape[1]

        feature_weights = []
        for output_row in y.T:
            # define LpProblem
            nvAlgo = pulp.LpProblem(sense=pulp.LpMinimize)
            n = np.arange(n_samples)
            p = np.arange(n_features)

            q = pulp.LpVariable.dicts('q', p)
            u = pulp.LpVariable.dicts('u', n, lowBound=0)
            o = pulp.LpVariable.dicts('o', n, lowBound=0)

            nvAlgo += (sum([self.cp * u[i] for i in n]) + sum([self.ch * o[i] for i in n])) / len(n)

            for i in n:
                nvAlgo += u[i] >= output_row[i] - q[0] - sum([q[j] * X[i, j] for j in p if j != 0])
                nvAlgo += o[i] >= q[0] + sum([q[j] * X[i, j] for j in p if j != 0]) - output_row[i]
            nvAlgo.solve()

            feature_weights_output_row = []
            for feature in q:
                feature_weights_output_row += [q[feature].value()]

            feature_weights.append(feature_weights_output_row)

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
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        # Add intercept
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]

        if self.n_outputs_ == 1:
            pred = X.dot(self.feature_weights_[0])
            pred = np.reshape(pred, (-1, 1))
        else:
            pred = []
            for weights in self.feature_weights_:
                pred.append(X.dot(weights))
            pred = np.array(pred).T

        return pred


class KernelOptimization:
    """A Kernel Optimization newsvendor estimator
    Parameters
    ----------
    cp : float or int, default=None
        the overage costs per unit.
    ch : float or int, default=None
        the underage costs per unit:
    kernel_type:  {"uniform", "gaussian"}, default="uniform"
        The type of the kernel function
    kerne_weight: float or int, default=1
        The weight of the kernel function

    Attributes
    ----------
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs
    X_ : array of shape (n_samples, n_features)
        The historic X-data
    Y_ : array of shape (n_samples, n_outputs)
        The historic y-data

    References
    ----------
    .. [1] Gah-Yi Ban, Cynthia Rudin, "The Big Data Newsvendor: Practical Insights from
    Machine Learning", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import KernelOptimization
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cp,ch = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = KernelOptimization(cp, ch)
    >>> mdl.fit(X_train, Y_train)
    >>> y_pred = mdl.predict(X_test)
    >>> calc_avg_costs(Y_test, y_pred, cp, ch)
    61.68
    """

    def __init__(self, cp, ch, kernel_type="uniform", kernel_weight=1):
        self.cp = cp
        self.ch = ch
        self.kernel_type = kernel_type
        self.kernel_weight = kernel_weight
        self.kernel = Kernel(self.kernel_type, self.kernel_weight)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.X_ = X
        self.y_ = y

        # Determine output settings
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
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
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        pred_all = []
        for i in range(self.n_outputs_):
            # concat X, y to sort by target column (y). Then split again.
            data_hist = np.append(self.X_, self.y_[:, i].reshape(-1, 1), axis=1)
            data_hist = data_hist[np.argsort(data_hist[:, self.n_features_])]

            X_hist = data_hist[:, np.r_[0:self.n_features_]]
            y_hist = data_hist[:, self.n_features_]
            pred = []
            for row in X:
                distances = distance_matrix(X_hist, [row]).ravel()
                distances_kernel_weighted = np.array([self.kernel.get_kernel_output(x) for x in distances])
                K_sum_total = np.sum(distances_kernel_weighted)
                K_sum = 0
                last_value = 0
                for index, value in np.ndenumerate(y_hist):
                    i = index[0]
                    if i < 1:
                        if i == y_hist.shape[0] - 1:
                            pred += [value]

                        else:
                            K_sum += distances_kernel_weighted[i]
                            last_value = value
                            continue

                    elif value != last_value and i < (y_hist.shape[0] - 1):
                        tf_lhs = K_sum / K_sum_total
                        tf_rhs = self.cp / (self.cp + self.ch)
                        if tf_lhs >= tf_rhs:
                            pred += [last_value]
                            break
                        else:
                            K_sum += distances_kernel_weighted[i]
                            last_value = value

                    elif value == last_value and i < (y_hist.shape[0] - 1):
                        K_sum += distances_kernel_weighted[i]

                    else:
                        pred += [value]
            pred_all.append(pred)
        return np.asarray(pred_all).T
