import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from sklearn.metrics import accuracy_score
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
    model.add(layers.Dense(5)) # replace the second parameter of nn.Linear with classification type of your dataset
    model.add(layers.Activation('softmax'))

    model.summary()
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])
    history = model.fit(train_data, train_label, batch_size = batch_size, epochs = epochs_num)

    result_1 = model.predict(train_data)
    result_2 = model.predict(test_data)
    result_3 = model.predict(valid_data)

    result_1 = np.argmax(result_1, axis = 1)
    result_2 = np.argmax(result_2, axis = 1)
    result_3 = np.argmax(result_3, axis = 1)

    x = accuracy_score(test_label, result_2)

    print(x)
    
    fp = open('/home/wang/rong/wang/LU_tensor/5class/result/cnn.txt', "a")
    fp.write(str(x) + "\n")
    fp.close()

    np.savetxt('/home/wang/rong/wang/LU_tensor/5class/output/cnn_test.txt', result_2.astype(int), fmt='%i')
    
