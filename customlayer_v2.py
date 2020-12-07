import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape, Input, Lambda, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Add, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow.keras as keras


class preLSTM(keras.layers.Layer):
    def __init__(self, n_outputs, n_steps, **kwargs):
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        super(preLSTM, self).__init__(**kwargs)
        self.reshape = Reshape((self.n_steps, -1, self.n_outputs))

    def call(self, inputs):
        output = self.reshape(inputs)
        output = tf.transpose(output, perm=[0, 1, 3, 2])
        return output

    def get_config(self):
        config = {'n_outputs': self.n_outputs,
                  'n_steps': self.n_steps}
        base_config = super(preLSTM, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LSTMcluster(keras.layers.Layer):
    def __init__(self, n_outputs, n_steps, n_cell_und=1, dropout=0.2, **kwargs):
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.n_steps = n_steps
        self.n_cell_und = n_cell_und
        super(LSTMcluster, self).__init__(**kwargs)
        self.flat = Flatten()
        self.reshape = Reshape((self.n_steps, -1))
        self.LSTMmodel = LSTM(
            self.n_cell_und, return_sequences=False, dropout=self.dropout, activation="sigmoid")

    def call(self, inputs):
        output = self.LSTMmodel(self.reshape(inputs))
        output = self.flat(output)
        return output

    def get_config(self):
        config = {'n_outputs': self.n_outputs,
                  'dropout': self.dropout,
                  'n_steps': self.n_steps,
                  'n_cell_und': self.n_cell_und, }
        base_config = super(LSTMcluster, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class preAR(keras.layers.Layer):
    def __init__(self, n_outputs, **kwargs):
        self.n_outputs = n_outputs
        super(preAR, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs[:, :, :self.n_outputs]
        output = tf.transpose(x, perm=[0, 2, 1])
        return output

    def get_config(self):
        config = {'n_outputs': self.n_outputs}
        base_config = super(preAR, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ARmodel(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ARmodel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
            name="weight",
        )
        self.b = self.add_weight(
            shape=(1,), initializer="random_normal", trainable=True, name="bias",
        )

    def call(self, inputs):
        return tf.keras.activations.sigmoid(tf.add(tf.matmul(inputs, self.w), tf.reduce_sum(self.b*self.w)))
        # return tf.matmul(inputs, self.w)

    def get_config(self):
        config = {}
        base_config = super(ARmodel, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ARcluster(keras.layers.Layer):
    def __init__(self, n_outpus, **kwargs):
        self.n_outpus = n_outpus
        super(ARcluster, self).__init__(**kwargs)
        self.flat = Flatten()
        self.ARmodel = ARmodel()

    def call(self, inputs):
        output = self.ARmodel(inputs)
        output = self.flat(output)
        return output

    def get_config(self):
        config = {'n_outpus': self.n_outpus}
        base_config = super(ARcluster, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def r2_coeff_det(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def model_forecasting_v2(input_shape, n_outputs, saved_file=None, summary=True):
    n_steps = input_shape[0]
    x = Input(input_shape)
    LSTMbulk = LSTMcluster(n_outputs=n_outputs, n_steps=n_steps, n_cell_und=1)
    transAR = preAR(n_outputs)
    ARbulk = ARcluster(n_outputs)
    addLayer = Add()
    prelstm = preLSTM(n_outputs=n_outputs, n_steps=n_steps)
    dense = Dense(1, activation='sigmoid',)
    concatenate = Concatenate()
    # construyendo modelo
    y1 = prelstm(x)
    y1 = LSTMbulk(y1)
    y2 = transAR(x)
    y2 = ARbulk(y2)
    # out = concatenate([y1, y2])
    # out = dense(out, )  #
    out = addLayer([y1, y2])
    model = Model(inputs=x, outputs=out)
    metrics = []
    metrics += [tf.keras.metrics.MeanAbsolutePercentageError()]
    metrics += [tf.keras.metrics.MeanSquaredError()]
    metrics += [r2_coeff_det]
    # MeanSquaredError  mean_absolute_percentage_error    MeanAbsoluteError
    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(optimizer=Adam(lr=1e-4), loss=loss,
                  metrics=metrics)
    if (saved_file):
        model.load_weights(saved_file)
        try:
            # model.load_model(saved_file)
            print("Pesos cargados")
        except:
            print("No se puede cargar los pesos")
    if summary:
        model.summary()
    return model
