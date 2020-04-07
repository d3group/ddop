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
    feature_weights_: 1d array
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
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = ERM(cp=15, ch=10)
    >>> mdl.fit(X_train, Y_train)
    >>> y_pred = mdl.predict(X_test)
    >>> calc_avg_costs(cp, ch, Y_test, y_pred)
    48.85
    """
    def __init__(self, cp, ch):
        self.cp = cp
        self.ch = ch

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # Determine output settings
        self.n_features_ = X.shape[1]

        # Add intercept
        n_samples = X.shape[0]
        X = np.c_[np.ones(n_samples), X]

        """ Following code is based on dataframe/series format. Therefore convert X to dataframe and
        y to series. Procedure will be changed to a numpy based format in the next development step
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)

        nvAlgo = pulp.LpProblem(sense=pulp.LpMinimize)

        n = X.index.values
        p = X.columns.values

        q = pulp.LpVariable.dicts('q', p)
        u = pulp.LpVariable.dicts('u', n, lowBound=0)
        o = pulp.LpVariable.dicts('o', n, lowBound=0)

        nvAlgo += (sum([self.cp * u[i] for i in n]) + sum([self.ch * o[i] for i in n])) / len(n)

        for i in n:
            nvAlgo += u[i] >= y.loc[i] - q[0] - sum([q[j] * X.loc[i, j] for j in p if j != 0])
            nvAlgo += o[i] >= q[0] + sum([q[j] * X.loc[i, j] for j in p if j != 0]) - y[i]

        nvAlgo.solve()

        feature_weights = []
        for feature in q:
            feature_weights += [q[feature].value()]

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
        pred = X.dot(self.feature_weights_)
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
    X_ : 2d array
        The historic X-data
    Y_ : 1d array
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
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = KernelOptimization(cp, ch)
    >>> mdl.fit(X_train, Y_train)
    >>> y_pred = mdl.predict(X_test)
    >>> calc_avg_costs(cp, ch, Y_test, y_pred)
    61.68
    """
    def __init__(self, cp, ch, kernel_type="uniform", kernel_weight=1):
        self.cp = cp
        self.ch = ch
        self.kernel_type = kernel_type
        self.kernel_weight = kernel_weight
        self.kernel = Kernel(self.kernel_type, self.kernel_weight)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        n_features = X.shape[1]

        # concat X, y inorder to sort by target column (y). Then split again.
        data_hist = np.append(X, y.reshape(-1, 1), axis=1)
        data_hist = data_hist[np.argsort(data_hist[:, n_features])]

        self.X_ = data_hist[:, np.r_[0:n_features]]
        self.y_ = data_hist[:, n_features]

        # Determine output settings
        self.n_features_ = n_features

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
        X_hist = self.X_
        y_hist = self.y_

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

        return pred
