import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import update

def LSTM(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size):

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
    valid_data = valid_data.reshape(valid_data.shape[0], 1, valid_data.shape[1])

    model = models.Sequential()
    model.add(layers.LSTM(32, return_sequences=True, input_shape=(None, 122)))
    model.add(layers.LSTM(16))
    model.add(layers.Dense(16, activation='ReLU'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))

    model.summary()

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])
    history = model.fit(train_data, train_label, batch_size = batch_size, epochs = epochs_num)

    result_1 = model.predict(train_data)
    result_2 = model.predict(test_data)
    result_3 = model.predict(valid_data)

    train_score = []
    test_score = []
    valid_score = []

    train_score = update.metrics(result_1, train_label, "NSL-KDDTrain+_LSTM", "LSTM", "train")
    test_score = update.metrics(result_2, test_label, "NSL-KDDTest+_LSTM", "LSTM", "test")
    valid_score = update.metrics(result_3, valid_label, "NSL-KDDTest21_LSTM", "LSTM", "valid")

    update.result_update(train_score, test_score , valid_score,"/home/wang/rong/wang/LU_tensor/2class/result/LSTM/")

    model.save('model_2.h5')