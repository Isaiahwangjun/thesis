import tensorflow as tf
import update
from tensorflow.keras import models, layers
import numpy as np
import sklearn
tf.compat.v1.disable_eager_execution()


def CNN(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size):
    
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
    valid_data = valid_data.reshape(valid_data.shape[0], 1, valid_data.shape[1])

    model = models.Sequential()
    model.add(layers.Conv1D(50, 3, activation='ReLU', padding='same', input_shape=(None, 122)))
    model.add(layers.Conv1D(100, 3, activation='ReLU', padding='same'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(50, 3, activation='ReLU', padding ='same', dilation_rate=2))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(1024, activation='sigmoid'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='sigmoid'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2)) # replace the second parameter of nn.Linear with classification type of your dataset
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

    train_score = update.metrics(result_1, train_label, "NSL-KDDTrain+_CNN", "CNN", "train")
    test_score = update.metrics(result_2, test_label, "NSL-KDDTest+_CNN", "CNN", "test")
    valid_score = update.metrics(result_3, valid_label, "NSL-KDDTest21_CNN", "CNN", "valid")

    update.result_update(train_score, test_score ,valid_score,"/home/wang/rong/wang/LU_tensor/2class/result/CNN/")

    model.save('model_1.h5')