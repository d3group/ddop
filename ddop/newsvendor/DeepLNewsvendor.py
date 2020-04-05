from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from ..utils.validation import check_is_fitted


class DeepLNewsvendor:
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
