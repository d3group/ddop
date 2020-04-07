from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from ..utils.validation import check_is_fitted


class DeepLNewsvendor:
    """A newsvendor estimator based on Deep Learning

    Parameters
    ----------
    cp : float or int, default=None
        the overage costs per unit.
    ch : float or int, default=None
        the underage costs per unit:

    Attributes
    ----------
    model_ : tensorflow.python.keras.engine.sequential.Sequential
        Sequential model from keras used for this estimator

    References
    ----------
    .. [1] Afshin Oroojlooyjadid, Lawrence V. Snyder, Martin Takáˇc,
            "Applying Deep Learning to the Newsvendor Problem", 2018.

    Examples
    --------
    >>> from ddop.datasets.load_datasets import load_data
    >>> from ddop.newsvendor import DeepLNewsvendor
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_data("yaz_steak.csv")
    >>> X = data.iloc[:,0:24]
    >>> Y = data.iloc[:,24]
    >>> cp,ch = 15,10
    >>> X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    >>> mdl = DeepLNewsvendor(cp, ch)
    >>> mdl.fit(X_train, Y_train)
    >>> y_pred = mdl.predict(X_test)
    >>> calc_avg_costs(Y_test, y_pred, cp, ch)
    52.97
    """
    def __init__(self, cp, ch):
        self.cp = cp
        self.ch = ch

    def __nv_loss(self, cp, ch):
        def customized_loss(y_true, y_pred):
            loss = K.switch(K.less(y_pred, y_true), cp * (y_true - y_pred), ch * (y_pred - y_true))
            return K.sum(loss)

        return customized_loss

    def __baseline_model(self, X):
        nFeatures = X.shape[1]
        model = Sequential()
        model.add(Dense(nFeatures, activation='relu', input_dim=nFeatures))
        model.add(Dense(3 * nFeatures))
        model.add(Dense(2 * nFeatures))
        model.add(Dense(1))
        model.compile(loss=self.__nv_loss(self.cp, self.ch), optimizer='adam')
        return model

    def fit(self, X, Y):
        model = self.__baseline_model(X)
        model.fit(X, Y, epochs=500, verbose=0)
        self.model_ = model
        return self

    def predict(self, X):
        check_is_fitted(self)
        pred = self.model_.predict(X)
        pred_flatted = pred.ravel()
        return pred_flatted
