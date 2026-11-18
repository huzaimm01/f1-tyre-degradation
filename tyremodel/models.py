"""The three models compared: a mean baseline, linear regression and a small network."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


def fit_mean(X, y):
    return DummyRegressor(strategy="mean").fit(X, y)


def fit_linear(X, y):
    return LinearRegression().fit(X, y)


class LapTimeNet:
    """Two hidden layers (32, 16) predicting lap time in seconds."""

    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.offset = 0.0

    def fit(self, X, y, epochs=300):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        # Lap times sit around 80 s, so train on the difference from the mean.
        self.offset = float(np.mean(y))
        inputs = X.to_numpy(dtype="float32")

        scaler = tf.keras.layers.Normalization(axis=-1)
        scaler.adapt(inputs)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(inputs.shape[1],)),
            scaler,
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
        stop = tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
        # The validation rows are a random slice of the training laps, used only
        # to decide when to stop. The test set is never touched.
        self.model.fit(
            inputs, np.asarray(y, dtype="float32") - self.offset,
            validation_split=0.2, epochs=epochs, batch_size=32,
            callbacks=[stop], verbose=0,
        )
        return self

    def predict(self, X):
        out = self.model.predict(X.to_numpy(dtype="float32"), verbose=0)
        return out.ravel() + self.offset
