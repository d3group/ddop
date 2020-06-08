from ..utils.validation import check_cu_co
from ..metrics.costs import calc_avg_costs
from .DecisionTreeNewsvendor import DecisionTreeNewsvendor
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import _tree
import numpy as np

DTYPE = _tree.DTYPE


class RandomForestNewsvendor(ForestRegressor):
    """A random forest regressor for a newsvendor problem.

    The implementation is based on the RandomForestRegressor from scikit-learn [4].
    It was adaped to solve the newsvendor problem by using a random forest as described
    in [3]. Therefore it takes two additional parameter co and cu and a custom criterion
    is used to build the trees.

    Parameters
    ----------
    criterion : {"newsvendor"}, default="newsvendor"
        The function to measure the quality of a split. Supported is only
        "newsvendor", which minimizes the Loss-function described in [5]
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
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
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : DecisionTreeNewsvendor
        The child estimator template used to create the collection of fitted
        sub-estimators.
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    X_ : array of shape (n_samples, n_features)
        The historic X-data
    y_ : array of shape (n_samples, n_outputs)
        The historic y-data
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.
    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

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
    .. [3] J. Meller and F. Taigel "Machine Learning for Inventory Management:
            Analyzing Two Concepts to Get From Data to Decisions",
            Available at SSRN 3256643 (2019)
    .. [4]  scikit-learn, ForestRegressor,
            <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py>
    .. [5] N. Meinshausen, "Quantile regression forests." Journal of Machine Learning
           Research 7.Jun (2006): 983-999.
     Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import RandomForestNewsvendor
    >>> from ddop.metrics.costs import calc_avg_costs
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = RandomForestNewsvendor(max_depth=5, cu=cu, co=co, random_state=0)
    >>> mdl.fit(X_train, Y_train)
    >>> score(X_test, Y_test)
    [76.68421053]
    """

    def __init__(self,
                 criterion="NewsvendorCriterion",
                 cu=None,
                 co=None,
                 n_estimators=100, *,
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
                 max_samples=None):
        super().__init__(
            base_estimator=DecisionTreeNewsvendor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "cu", "co", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples)
        self.criterion = criterion
        self.cu = cu
        self.co = co
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y),
        and save the historic X- and y-data needed for the prediction method

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        Returns
        ----------
        self : RandomForestNewsvendor
            Fitted estimator
        """

        super().fit(X, y, sample_weight)

        check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        check_y_params = dict(ensure_2d=False, dtype=None)

        X, y = self._validate_data(X, y,
                                   validate_separately=(check_X_params,
                                                        check_y_params))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Historic data for predict method
        self.X_ = X
        self.y_ = y

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        return self

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
        X_hist = self.X_
        y_hist = self.y_
        X_leaf_indices = self.apply(X)
        X_hist_leaf_indices = self.apply(X_hist)
        pred = []
        for xi in X_leaf_indices:
            N_theta_t = np.array([sum(xi == xi_hist) for xi_hist in X_hist_leaf_indices])
            sample_weights = N_theta_t / sum(N_theta_t)
            pred_xi = []
            for i in range(self.n_outputs_):
                data = np.c_[sample_weights, y_hist[:, i]]
                data = data[np.argsort(data[:, 1])]
                sum_wi = 0
                for row in data:
                    sum_wi = sum_wi + row[0]
                    if sum_wi >= self.cu_[i] / (self.cu_[i] + self.co_[i]):
                        pred_xi.append(row[1])
                        break
            pred.append(pred_xi)
        return np.asarray(pred)

    def quantil_predict(self, X):
        """Predict value for X.

        Note: Compared to the method 'predict', which predicts the value of X  using
        a sample weight approach [3], this method formulates the prediction as
        a quantile regression problem [5]. The result of both methods is the same

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
        X_hist = self.X_
        y_hist = self.y_
        X_leaf_indices = self.apply(X)
        X_hist_leaf_indices = self.apply(X_hist)
        pred = []
        for xi in X_leaf_indices:
            cnt = np.array([sum(xi == xi_hist) for xi_hist in X_hist_leaf_indices])
            d = np.zeros(sum(cnt))
            pred_xi = []
            for k in range(self.n_outputs_):
                i = 0
                for j in range(cnt.size):
                    d[i:cnt[j] + i] = y_hist[j,k]
                    i = i + cnt[j]
                pred_xi.append(np.quantile(d,self.cu_[k]/(self.cu_[k]+self.co_[k]), interpolation='higher'))
            pred.append(pred_xi)
        return np.array(pred)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return calc_avg_costs(y, y_pred, self.cu_, self.co_)


