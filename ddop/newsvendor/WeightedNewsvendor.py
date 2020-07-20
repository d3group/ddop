from .base import BaseNewsvendor, DataDrivenMixin
from ..utils.validation import check_cu_co
import pulp
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_array


class BaseWeightedNewsvendor(BaseNewsvendor, DataDrivenMixin, ABC):
    """Base class for weighted newsvendor.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 cu,
                 co,
                 ):
        self.cu = cu
        self.co = co

    def fit(self, X, y):

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
        """Initialise the underlying model"""

    @abstractmethod
    def _calc_weights(self, sample):
        """Calculate the sample weights"""

    def _findQ(self, weights):
        """Calculate the optimal order quantity q"""

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
        weights = np.apply_along_axis(self._calc_weights, 1, X)
        pred = np.apply_along_axis(self._findQ, 1, weights)
        return pred


class EqualWeightedNewsvendor(BaseWeightedNewsvendor):
    """A equal weighted SAA newsvendor

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
        The underage costs per unit. If None, then underage costs are one
        for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
        The overage costs per unit. If None, then overage costs are one
        for each target variable

    Attributes
    ---------
    X_ : array of shape (n_samples, n_features)
        The X training data
    y_ : array of shape (n_samples, n_outputs)
        The y training data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    n_samples_ : int
        The number of samples when ``fit`` is performed.

    Notes
    -----

    References
    ----------
    .. [1] Bertsimas, Dimitris, and Nathan Kallus, "From predictive to prescriptive analytics."
           arXiv preprint arXiv:1402.5481 (2014).

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import EqualWeightedNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = EqualWeightedNewsvendor(cu, co)
    >>> mdl.fit(X_train, Y_train)
    >>> score(X_test, Y_test)
    TODO: Add output
    """
    def __init__(self,
                 cu,
                 co):
        super().__init__(
            cu=cu,
            co=co)

    def _get_fitted_model(self, X, y=None):
        pass

    def _calc_weights(self, sample):
        weights = np.full((self.n_samples_), 1 / self.n_samples_)
        return weights

    def fit(self, X, y):
        """ Fit the estimator from the training set (X,y)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.

        Returns
        ----------
        self : EqualWeightedNewsvendor
            Fitted estimator
        """

        super().fit(X,y)

        return self


class RandomForestWeightedNewsvendor(BaseWeightedNewsvendor):
    """A random forest weighted SAA newsvendor

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
        The underage costs per unit. If None, then underage costs are one
        for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
        The overage costs per unit. If None, then overage costs are one
        for each target variable
    criterion : {"mse", "friedman_mse", "mae"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    oob_score : bool, default=False
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ---------
    X_ : array of shape (n_samples, n_features)
        The X training data
    y_ : array of shape (n_samples, n_outputs)
        The y training data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    n_samples_ : int
        The number of samples when ``fit`` is performed.
    model_ : RandomForestRegressor
        The underlying model used to calculate the sample weights

    See Also
    --------
    DecisionTreeNewsvendor, ForestRegressor [4]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.[4]

    References
    ----------
    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.
    .. [3] Bertsimas, Dimitris, and Nathan Kallus, "From predictive to prescriptive analytics."
           arXiv preprint arXiv:1402.5481 (2014).
    .. [4] scikit-learn, ForestRegressor,
           <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py>
    .. [5] N. Meinshausen, "Quantile regression forests." Journal of Machine Learning
           Research 7.Jun (2006): 983-999.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import RandomForestWeightedNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = RandomForestWeightedNewsvendor(cu, co, random_state=0)
    >>> mdl.fit(X_train, Y_train)
    >>> score(X_test, Y_test)
    TODO: Add output
    """
    def __init__(self,
                 cu,
                 co,
                 criterion="mse",
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
        self.criterion = criterion
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
            criterion=self.criterion,
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
        self.train_leaf_indices_ = model.apply(X)

    def _calc_weights(self, sample):
        sample_leaf_indices = self.model_.apply([sample])
        n = np.sum(sample_leaf_indices == self.train_leaf_indices_, axis=0)
        treeWeights = (sample_leaf_indices == self.train_leaf_indices_) / n
        weights = np.sum(treeWeights, axis=1) / self.n_estimators
        print(weights.shape)
        return weights

    def fit(self, X, y):
        """ Fit the estimator from the training set (X,y)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.

        Returns
        ----------
        self : RandomForestWeightedNewsvendor
            Fitted estimator
        """

        super().fit(X,y)

        return self


class KNeighborsWeightedNewsvendor(BaseWeightedNewsvendor):
    """A k-nearest-neighbor weighted SAA newsvendor

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
        The underage costs per unit. If None, then underage costs are one
        for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
        The overage costs per unit. If None, then overage costs are one
        for each target variable
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.
    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ---------
    X_ : array of shape (n_samples, n_features)
        The X training data
    y_ : array of shape (n_samples, n_outputs)
        The y training data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    n_samples_ : int
        The number of samples when ``fit`` is performed.
    model_ : NearestNeighbors
        The underlying model used to calculate the sample weights

    See Also
    --------
    NearestNeighbors [2]

    References
    ----------
    .. [1] Bertsimas, Dimitris, and Nathan Kallus, "From predictive to prescriptive analytics."
           arXiv preprint arXiv:1402.5481 (2014).
    .. [2] scikit-learn, NearestNeighbors,
           <https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/neighbors/_unsupervised.py>

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import KNeighborsWeightedNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = KNeighborsWeightedNewsvendor(cu, co, random_state=0)
    >>> mdl.fit(X_train, Y_train)
    >>> score(X_test, Y_test)
    TODO: Add output
    """
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

    def fit(self, X, y):
        """ Fit the estimator from the training set (X,y)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.

        Returns
        ----------
        self : KNeighborsWeightedNewsvendor
            Fitted estimator
        """

        super().fit(X,y)

        return self
