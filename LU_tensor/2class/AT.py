import os
import update
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model

# KERAS_ATTENTION_DEBUG: If set to 1. Will switch to debug mode.
# In debug mode, the class Attention is no longer a Keras layer.
# What it means in practice is that we can have access to the internal values
# of each tensor. If we don't use debug, Keras treats the object
# as a layer and we can only get the final output.
debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))


class Attention(object if debug_flag else Layer):

    def __init__(self, unit=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = unit

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: felixhao28, philipperemy.
        """
        if debug_flag:
            self.build(inputs.shape)
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.attention_score_vec(inputs)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config
    
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, models



def AT(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size):
    # Dummy data. There is nothing to learn in this example.
    num_samples, time_steps, input_dim, output_dim = 100, 1, 41, 1

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
    valid_data = valid_data.reshape(valid_data.shape[0], 1, valid_data.shape[1])

    # Define/compile the model.
    # model_input = Input(shape=(time_steps, input_dim))
    # x = LSTM(64, return_sequences=True)(model_input)
    # x = Attention()(x)
    # x = Dense(2)(x)
    # x = Activation('softmax')(x)
    #model = Model(inputs = model_input, outputs = x)
    #model.compile(loss='mae', optimizer='adam')
    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=(None, 122)))
    model.add(Attention(32))
    model.add(layers.Dense(16, activation='ReLU'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])
    model.summary()

    # train.
    history = model.fit(train_data, train_label, batch_size = batch_size, epochs = epochs_num)

    result_1 = model.predict(train_data)
    result_2 = model.predict(test_data)
    result_3 = model.predict(valid_data)

    train_score = []
    test_score = []
    valid_score = []

    train_score = update.metrics(result_1, train_label, "NSL-KDDTrain+_AT", "AT", "train")
    test_score = update.metrics(result_2, test_label, "NSL-KDDTest+_AT", "AT", "test")
    valid_score = update.metrics(result_3, valid_label, "NSL-KDDTest21_AT", "AT", "valid")

    update.result_update(train_score, test_score ,valid_score,"/home/wang/rong/wang/LU_tensor/2class/result/AT/")

    model.save('model_3.h5')

    # test save/reload model.
    # pred1 = model.predict(data_x)
    # model.save('model_3.h5')
    # model_h5 = load_model('model_3.h5', custom_objects={'Attention': Attention})
    # pred2 = model_h5.predict(data_x)
    # np.testing.assert_almost_equal(pred1, pred2)
    # print('Success.')


    if __name__ == '__main__':
        main()