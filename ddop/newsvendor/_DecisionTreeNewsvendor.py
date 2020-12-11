from ..utils.criterion import NewsvendorCriterion
from ..utils.validation import check_cu_co
from ..metrics._newsvendor import calc_avg_costs

import numbers
import warnings
from math import ceil

import numpy as np
from scipy.sparse import issparse

from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.tree import _tree
from sklearn.tree._tree import Tree
from sklearn.tree._splitter import Splitter
from sklearn.tree import _splitter, _criterion
from sklearn.tree._tree import BestFirstTreeBuilder
from sklearn.tree._tree import DepthFirstTreeBuilder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_REG = {"mse": _criterion.MSE, "friedman_mse": _criterion.FriedmanMSE,
                "mae": _criterion.MAE}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
                    "random": _splitter.RandomSparseSplitter}


class DecisionTreeNewsvendor(DecisionTreeRegressor):
    """A Decision Tree Regressor

    The implementation is based on the DecisionTreeRegressor from scikit-learn [5],
    but extends it by another criterion. This criterion implements the newsvendor-loss
    function for which the model takes two additional parameter co and cu.

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    criterion: {"newsvendor", "mse", "friedman_mse, "mae"}, default="newsvendor"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        and "mae" for the mean absolute error, which minimizes the L1 loss
        using the median of each terminal node while "newsvendor" minimizes
        the newsvendor-loss function. For the criteria "mse", "friedman_mse
        and "mae", the model es equal to DecisionTreeRegressor from sklean [5].
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
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
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
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
    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
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
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values).
    max_features_ : int
        The inferred value of max_features.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.
    tree_ : Tree
        The underlying Tree object.

    See Also
    --------
    DecisionTreeRegressor [5]

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.
    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    .. [5]  scikit-learn, DecisionTreeRegressor,
            <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py>

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import DecisionTreeNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = DecisionTreeNewsvendor(max_depth=5, cu=cu, co=co, random_state=0)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, Y_test)
    TODO: ADD SCORE
    """

    def __init__(self,
                 cu=None,
                 co=None,
                 criterion="newsvendor",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha)
        self.cu = cu
        self.co = co

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a newsvendor decision tree regressor from the training set (X, y).

        Method is based on [1] and was adapted to enable usage of the newsvendor criterion

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        Returns
        -------
        self : NewsvendorDecisionTreeRegressor
            Fitted estimator.

        References
        ----------
        [1] scikit-learn, BaseDecisionTree.fit()
            <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py>
        """

        random_state = check_random_state(self.random_state)

        if self.ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be greater than or equal to 0")

        # Need to validate separately here.
        # We can't pass multi_ouput=True because that would allow y to be
        # csr.
        check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        check_y_params = dict(ensure_2d=False, dtype=None)

        X, y = self._validate_data(X, y,validate_separately=(check_X_params,check_y_params))

        if issparse(X):
            X.sort_indices()
            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        self.X_ = X
        self.y_ = y

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)

        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, numbers.Integral):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        min_impurity_split = self.min_impurity_split
        if min_impurity_split is not None:
            warnings.warn("The min_impurity_split parameter is deprecated. "
                          "Its default value has changed from 1e-7 to 0 in "
                          "version 0.23, and it will be removed in 0.25. "
                          "Use the min_impurity_decrease parameter instead.",
                          FutureWarning)

            if min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            min_impurity_split = 0

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")

        # Build tree
        if self.criterion == "newsvendor":
            criterion = NewsvendorCriterion(self.n_outputs_, n_samples, self.cu_, self.co_)

        else:
            criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        self.tree_ = Tree(self.n_features_,
                          # TODO: tree should't need this in this case
                          np.array([1] * self.n_outputs_, dtype=np.intp),
                          self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth,
                                            self.min_impurity_decrease,
                                            min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           min_impurity_split)

        builder.build(self.tree_, X, y, sample_weight)

        self._prune_tree()

        return self

    def score(self, X, y):
        """
        Return the average costs of the prediction

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.
        y : array-like of shape (n_samples, n_outputs)
            The true values for x.
        sample_weight: array-like of shape (n_samples)
            Sample weights

        Returns
        -------
         score: float
            The average costs
        """

        y_pred = self.predict(X)
        return calc_avg_costs(y, y_pred, self.cu_, self.co_, multioutput="uniform_average")


