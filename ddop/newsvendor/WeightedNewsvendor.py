from .base import BaseNewsvendor, DataDrivenMixin
from ..utils.validation import check_cu_co
import pulp
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_array
from time import time


class BaseWeightedNewsvendor(BaseNewsvendor, DataDrivenMixin, ABC):
    @abstractmethod
    def __init__(self,
                 cu,
                 co,
                 ):
        self.cu = cu
        self.co = co

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

        self._get_fitted_model(X, y)

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

    @abstractmethod
    def _get_fitted_model(self, X, y=None):
        """
        set up the underlying model
        """

    @abstractmethod
    def _calc_weights(self, sample):
        """Calculate the sample weights"""

    def _findQ(self, weights):

        y = self.y_
        q = []

        for k in range(self.n_outputs_):
            opt_model = pulp.LpProblem(sense=pulp.LpMinimize)
            n = np.arange(self.n_samples_)

            q_k = pulp.LpVariable('q_k', lowBound=0)
            u = pulp.LpVariable.dicts('u', n, lowBound=0)
            o = pulp.LpVariable.dicts('o', n, lowBound=0)

            u_weighted = pulp.LpAffineExpression([(u[i], weights[i]) for i in n])
            o_weighted = pulp.LpAffineExpression([(o[i], weights[i]) for i in n])

            objective = u_weighted * self.cu_[k] + o_weighted * self.co_[k]

            opt_model.setObjective(objective)

            for i in n:
                opt_model += u[i] >= y[i, k] - q_k
                opt_model += o[i] >= q_k - y[i, k]
            opt_model.solve()

            q.append(q_k.value())

        return q

    def predict(self, X):
        check_is_fitted(self)
        weights = np.apply_along_axis(self._calc_weights, 1, X)
        pred = np.apply_along_axis(self._findQ, 1, weights)
        return pred


class EqualWeightedNewsvendor(BaseWeightedNewsvendor):
    def __init__(self,
                 cu,
                 co):
        super().__init__(
            cu=cu,
            co=co)

    def _get_fitted_model(self, X, y=None):
        pass

    def _calc_weights(self, sample):
        weights = np.full((self.n_samples_),1/self.n_samples_)
        return weights


class RandomForestWeightedNewsvendor(BaseWeightedNewsvendor):
    def __init__(self,
                 cu,
                 co,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None
                 ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        super().__init__(
            cu=cu,
            co=co)

    def _get_fitted_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples
        )

        self.model_ = model.fit(X, y)
        self.train_leaf_indices = model.apply(X)

    def _calc_weights(self, sample):
        sample_leaf_indices = self.model_.apply([sample])
        n = np.sum(sample_leaf_indices == self.train_leaf_indices, axis=0)
        treeWeights = (sample_leaf_indices == self.train_leaf_indices) / n
        weights = np.sum(treeWeights, axis=1) / self.n_estimators
        print(weights.shape)
        return weights


class KNeighborsWeightedNewsvendor(BaseWeightedNewsvendor):
    def __init__(self,
                 cu,
                 co,
                 n_neighbors=5,
                 radius=1.0,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 metric_params=None,
                 n_jobs=None
                 ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        super().__init__(
            cu=cu,
            co=co)

    def _get_fitted_model(self, X, y=None):
        model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs
        )

        self.model_ = model.fit(X)

    def _calc_weights(self, sample):
        neighbors = self.model_.kneighbors([sample], return_distance=False)[0]
        weights = np.array([1 / self.n_neighbors if i in neighbors else 0 for i in range(self.n_samples_)])
        return weights

