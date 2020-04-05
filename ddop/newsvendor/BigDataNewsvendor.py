import pulp
from ..utils.validation import check_is_fitted
from ..utils.kernels import Kernel
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from scipy.spatial import distance_matrix


class ERM:
    def __init__(self, cp, ch, lp_type):
        self.cp = 2
        self.ch = 1
        self.lp_type = 'minimize'

    def fit(self, X, y):
        # Add intercept
        X.insert(0, 'intercept', 1)

        if self.lp_type == 'minimize':
            nvAlgo = pulp.LpProblem(sense=pulp.LpMinimize)

        elif self.lp_type == 'maximize':
            nvAlgo = pulp.LpProblem(sense=pulp.LpMaximize)

        else:
            raise NameError('lp_type must be eigther minimize or maximize')

        n = X.index.values
        p = X.columns.values

        q = pulp.LpVariable.dicts('q', p)
        u = pulp.LpVariable.dicts('u', n, lowBound=0)
        o = pulp.LpVariable.dicts('o', n, lowBound=0)

        nvAlgo += (sum([self.cp * u[i] for i in n]) + sum([self.ch * o[i] for i in n])) / len(n)

        for i in n:
            nvAlgo += u[i] >= y.loc[i] - q['intercept'] - sum([q[j] * X.loc[i, j] for j in p if j != 'intercept'])
            nvAlgo += o[i] >= q['intercept'] + sum([q[j] * X.loc[i, j] for j in p if j != 'intercept']) - y[i]

        nvAlgo.solve()

        self.q_ = q

        return self

    def predict(self, X):
        check_is_fitted(self)
        X.insert(0, 'intercept', 1)
        pred = []
        for index, row in X.iterrows():
            # access data using column names
            value = 0
            for name in X.columns:
                value = value + row[name] * self.q_[name].value()
            pred += [value]
        return pred


class KernelOptimization:
    def __init__(self, cp, ch, kernel_type, kernel_weight):
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
