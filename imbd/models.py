import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


class KerasModel(KerasRegressor):
    def __init__(self, build_fn=None, **sk_params):
        super(KerasModel, self).__init__(build_fn=self.create_func_model,
                                         **sk_params)

    @staticmethod
    def create_func_model(n_features, dropout_rate=0.3):
        inputs = tf.keras.Input(shape=(n_features, ))

        conv_1 = tf.keras.backend.expand_dims(inputs, axis=1)
        conv_1 = tf.keras.layers.Conv1D(kernel_size=1, filters=64)(conv_1)
        conv_2 = tf.keras.backend.expand_dims(inputs, axis=1)
        conv_2 = tf.keras.layers.Conv1D(kernel_size=1, filters=64)(conv_2)
        conv_3 = tf.keras.backend.expand_dims(inputs, axis=1)
        conv_3 = tf.keras.layers.Conv1D(kernel_size=1, filters=64)(conv_3)

        x = tf.keras.layers.Attention()([conv_1, conv_2, conv_3])

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(20)(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def fit(self, x, y=None, **sk_params):
        n_features = len(x.columns)
        epochs = self.get_params().get('epochs', 100)
        dropout_rate = self.get_params().get('dropout_rate', 0.3)
        super(KerasModel, self).__init__(build_fn=self.create_func_model,
                                         n_features=n_features,
                                         dropout_rate=dropout_rate,
                                         epochs=epochs,
                                         **sk_params)

        super(KerasModel, self).fit(x, y)
        return self

    def __call__(self, x):
        super(KerasModel, self).__call__(x)