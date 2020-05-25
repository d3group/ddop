from .base import BaseNewsvendor
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from sklearn.utils.validation import check_is_fitted
import numpy as np

ACTIVATIONS = ['elu', 'selu', 'linear', 'tanh', 'relu', 'softmax', 'softsign', 'softplus',
               'sigmoid', 'hard_sigmoid', 'exponential']


class DeepLearningNewsvendor(BaseNewsvendor):
    """A newsvendor estimator based on Deep Learning

    Parameters
    ----------
    cu : {array-like of shape (n_outputs,), Number or None}, default=None
       The underage costs per unit. If None, then underage costs are one
       for each target variable
    co : {array-like of shape (n_outputs,), Number or None}, default=None
       The overage costs per unit. If None, then overage costs are one
       for each target variable
    hidden_layers : {'auto', 'custom'}, default='auto'
        Whether to use a automated or customized hidden layer structure.
        -   When set to 'auto' the network will use two hidden layers. The first
            with 2*n_features neurons and 'relu' as activation function the second
            one with n_features neurons and 'linear' as activation function
        -   When set to 'custom' the settings specified in both parameters 'neurons' and
            'activations' will be used to build the hidden layers of the network
    neurons : list, default=[100]
        The ith element represents the number of neurons in the ith hidden layer
        Only used when hidden_layers='custom'.
    activations : list, default=['relu']
        The ith element of the list represents the activation function of the ith layer.
        Valid activation functions are: 'elu', 'selu', 'linear', 'tanh', 'relu', 'softmax',
        'softsign', 'softplus','sigmoid', 'hard_sigmoid', 'exponential'.
        Only used when hidden_layers='custom'.
    optimizer: {'adam', 'sgd'}, default='adam'
        The optimizer to be used.
    epochs: int, default=100
        Number of epochs to train the model
    verbose: int 0, 1, or 2, default=1
        Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Attributes
    ----------
    model_ : tensorflow.python.keras.engine.sequential.Sequential
        Sequential model from keras used for this estimator
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs.

    References
    ----------
    .. [1] Afshin Oroojlooyjadid, Lawrence V. Snyder, Martin Takáˇc,
            "Applying Deep Learning to the Newsvendor Problem", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import DeepLearningNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cu,co = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = DeepLearningNewsvendor(cu, co)
    >>> mdl.fit(X_train, Y_train)
    >>> mdl.score(X_test, y_test)
    52.97
    """

    def __init__(self, cu, co, hidden_layers='auto', neurons=[100],
                 activations=['relu'], optimizer='adam', epochs=100, verbose=1):
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.activations = activations
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        super().__init__(
            cu=cu,
            co=co)

    def __nv_loss(self, cu, co):
        def customized_loss(y_true, y_pred):
            self.tensor_ = y_true
            loss = K.switch(K.less(y_pred, y_true), cu * (y_true - y_pred), co * (y_pred - y_true))
            return K.sum(loss)

        return customized_loss

    def __create_model(self):
        hidden_layers = self.hidden_layers
        neurons = self.neurons
        activations = self.activations
        n_features = self.n_features_
        n_outputs = self.n_outputs_

        model = Sequential()

        if hidden_layers == 'auto':
            model.add(Dense(2 * n_features, activation='relu', input_dim=n_features))
            model.add(Dense(n_features))
            model.add(Dense(n_outputs))

        else:
            for size, activation in zip(neurons, activations):
                model.add(Dense(units=size, activation=activation))
            model.add(Dense(n_outputs))
            model.build((None, n_features))

        model.compile(loss=self.__nv_loss(self.cu, self.co), optimizer=self.optimizer)

        return model

    def fit(self, X, y):
        # Validate input parameters
        self._validate_hyperparameters()

        X, y = self._validate_data(X, y, multi_output=True)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Determine output settings
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        model = self.__create_model()
        model.fit(X, y, epochs=self.epochs, verbose=self.verbose)
        self.model_ = model
        return self

    def _validate_hyperparameters(self):
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

        if self.hidden_layers == "custom" and np.any(np.array(neurons) <= 0):
            raise ValueError("neurons must be > 0, got %s." %
                             self.neurons)

        if self.hidden_layers == "custom" and np.any(np.array(activations) not in ACTIVATIONS):
            raise ValueError("Invalid activation function in activations. Supported are %s but got %s"
                             % (list(ACTIVATIONS), activations))

        if self.hidden_layers not in ["auto", "custom"]:
            raise ValueError("hidden_layers %s is not supported." % self.hidden_layers)

        if self.hidden_layers == "custom" and len(neurons) != len(activations):
            raise ValueError("When customizing the hidden layers neurons and activations must have same "
                             "length but neurons is of length %s and activations %s"
                             % (len(neurons), len(activations)))

        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be either 0, 1 or 2, got %s." %
                             self.verbose)

    def predict(self, X):
        check_is_fitted(self)
        pred = self.model_.predict(X)
        return pred