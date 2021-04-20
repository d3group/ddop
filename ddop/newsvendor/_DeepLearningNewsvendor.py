from ._base import BaseNewsvendor, DataDrivenMixin
from ..utils.validation import check_cu_co
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import switch, less
from tensorflow.keras.backend import sum as ksum
from tensorflow import cast

from sklearn.utils.validation import check_is_fitted
import numpy as np

ACTIVATIONS = ['elu', 'selu', 'linear', 'tanh', 'relu', 'softmax', 'softsign', 'softplus',
               'sigmoid', 'hard_sigmoid', 'exponential']


class DeepLearningNewsvendor(BaseNewsvendor, DataDrivenMixin):
    """A Deep-Learning model to solve the Newsvendor problem.

    This class implements the approach described in [1].

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    neurons : list, default=[100,50]
        The ith element represents the number of neurons in the ith hidden layer
        Only used when hidden_layers='custom'.
    activations : list, default=['relu','relu']
        The ith element of the list represents the activation function of the ith layer.
        Valid activation functions are: 'elu', 'selu', 'linear', 'tanh', 'relu', 'softmax',
        'softsign', 'softplus','sigmoid', 'hard_sigmoid', 'exponential'.
        Only used when hidden_layers='custom'.
    optimizer: {'adam', 'sgd'}, default='adam'
        The optimizer to be used.
    epochs: int, default=100
        Number of epochs to train the model
    verbose: int 0, 1, or 2, default=0
        Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Attributes
    ----------
    model_ : tensorflow.keras.Sequential
        The underlying model
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs.
    cu_ : ndarray, shape (n_outputs,)
        Validated underage costs.
    co_ : ndarray, shape (n_outputs,)
        Validated overage costs.

    References
    ----------
    .. [1] Afshin Oroojlooyjadid, Lawrence V. Snyder, Martin Takáˇc,
            "Applying Deep Learning to the Newsvendor Problem", 2018.

    Examples
    --------
    >>> from ddop.datasets import load_yaz
    >>> from ddop.newsvendor import DeepLearningNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> X, Y = load_yaz(include_prod=['STEAK'],return_X_y=True)
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    >>> mdl = DeepLearningNewsvendor(cu, co)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, Y_test)
    TODO: ADD SCORE
    """

    def __init__(self, cu=None, co=None, neurons=[100, 50], activations=['relu', 'relu'], optimizer='adam', epochs=100,
                 verbose=0):
        self.neurons = neurons
        self.activations = activations
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        super().__init__(
            cu=cu,
            co=co)

    def _nv_loss(self, cu, co):
        """Create a newsvendor loss function with the given under- and overage costs"""

        def customized_loss(y_true, y_pred):
            y_true = cast(y_true, y_pred.dtype)
            loss = switch(less(y_pred, y_true), cu * (y_true - y_pred), co * (y_pred - y_true))
            return ksum(loss)

        return customized_loss

    def _create_model(self):
        """Create model"""
        neurons = self.neurons
        activations = self.activations
        n_features = self.n_features_
        n_outputs = self.n_outputs_

        model = Sequential()

        for size, activation in zip(neurons, activations):
            model.add(Dense(units=size, activation=activation))
        model.add(Dense(n_outputs))
        model.build((None, n_features))

        model.compile(loss=self._nv_loss(self.cu_, self.co_), optimizer=self.optimizer)

        return model

    def fit(self, X, y):
        """Fit the model to the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_outputs)
            The target values.

        Returns
        ----------
        self : DeepLearningNewsvendor
            Fitted estimator
        """

        # Validate input parameters
        self._validate_hyperparameters()

        X, y = self._validate_data(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Determine output settings
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        # Check and format under- and overage costs
        self.cu_, self.co_ = check_cu_co(self.cu, self.co, self.n_outputs_)
        model = self._create_model()
        model.fit(X, y, epochs=self.epochs, verbose=self.verbose)
        self.model_ = model
        return self

    def _validate_hyperparameters(self):
        """validate hyperparameters"""

        # Make sure self.neurons is a list
        neurons = self.neurons
        if not hasattr(neurons, "__iter__"):
            neurons = [neurons]
        neurons = list(neurons)

        # Make sure self.activations is a list
        activations = self.activations
        if not hasattr(activations, "__iter__"):
            activations = [activations]
        activations = list(activations)

        if np.any(np.array(neurons) <= 0):
            raise ValueError("neurons must be > 0, got %s." %
                             self.neurons)

        if np.any(np.array([activation not in ACTIVATIONS for activation in activations])):
            raise ValueError("Invalid activation function in activations. Supported are %s but got %s"
                             % (list(ACTIVATIONS), activations))

        if len(neurons) != len(activations):
            raise ValueError("Neurons and activations must have same length but neurons is of length %s and "
                             "activations %s " % (len(neurons), len(activations)))

        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be either 0, 1 or 2, got %s." %
                             self.verbose)

    def predict(self, X):
        """Predict values for X.

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
        pred = self.model_.predict(X)
        return pred
