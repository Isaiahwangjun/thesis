import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
tf.compat.v1.disable_eager_execution()
import pandas as pd

def CNN(train_data, train_label, test_data, test_label, test_21_data, test_21_label, learningrate, epochs_num, batch_size, name, dim):
    

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
    test_21_data = test_21_data.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

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
    result_3 = model.predict(test_21_data)
    
    if(dim == 15):
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_train_' + str(dim) + name + '.txt', result_1.astype(float), fmt='%f', delimiter=',')
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_' + str(dim) + name + '.txt', result_2.astype(float), fmt='%f', delimiter=',')
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_21_' + str(dim) + name + '.txt', result_3.astype(float), fmt='%f', delimiter=',')

    result_1 = np.argmax(result_1, axis = 1)
    result_2 = np.argmax(result_2, axis = 1)
    result_3 = np.argmax(result_3, axis = 1)

    # result_1 = pd.DataFrame(result_1)
    # xxx = result_1.value_counts()
    # fp = open('/home/wang/rong/wang/LU_tensor/augmentation/result/cnn_check.txt', "a")
    # fp.write(str(xxx) + '\n')
    # fp.close()

    train_acc = accuracy_score(train_label, result_1)
    test_acc = accuracy_score(test_label, result_2)
    test_21_acc = accuracy_score(test_21_label, result_3)

    train_ps = precision_score(train_label, result_1, average = 'macro')
    test_ps = precision_score(test_label, result_2, average = 'macro')
    test_21_ps = precision_score(test_21_label, result_3, average = 'macro')

    train_rs = recall_score(train_label, result_1, average = 'macro')
    test_rs = recall_score(test_label, result_2, average = 'macro')
    test_21_rs = recall_score(test_21_label, result_3, average = 'macro')

    train_f1 = f1_score(train_label, result_1, average = 'macro')
    test_f1 = f1_score(test_label, result_2, average = 'macro')
    test_21_f1 = f1_score(test_21_label, result_3, average = 'macro')

    if(dim == 3):
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_train_' + str(dim) + name + '.txt', result_1.astype(int), fmt='%i')
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_' + str(dim) + name + '.txt', result_2.astype(int), fmt='%i')
        np.savetxt('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_21_' + str(dim) + name + '.txt', result_3.astype(int), fmt='%i')
    
    
    fp = open('/home/wang/Desktop/thesis/LU_tensor/augmentation/result/cnn_' + str(dim) + name + '.txt', "a")
    fp.write('train\n')
    fp.write('acc: ' + str(train_acc) + ', ps: ' + str(train_ps) + ', rs: ' + str(train_rs) + ', f1: ' + str(train_f1) + '\n')
    fp.write('test\n')
    fp.write('acc: ' + str(test_acc) + ', ps: ' + str(test_ps) + ', rs: ' + str(test_rs) + ', f1: ' + str(test_f1) + '\n')
    fp.write('test_21\n')
    fp.write('acc: ' + str(test_21_acc) + ', ps: ' + str(test_21_ps) + ', rs: ' + str(test_21_rs) + ', f1: ' + str(test_21_f1) + '\n')
    fp.close()

