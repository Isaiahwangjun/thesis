import sys 
sys.path.append("/home/wang/rong/wang") 
import datetime
from CNN import CNN
from LSTM import LSTM
from AT import AT
from load_output import output_load
import get_data
import update
import tensorflow as tf
from tensorflow.keras import models, layers
import sklearn
import numpy as np
import pandas as pd

train_model = str(sys.argv[1])
learningrate = float(sys.argv[2])
epochs_num = int(sys.argv[3])
run_num = int(sys.argv[4])
batch_size = int(sys.argv[5])

train_data, train_label, test_data, test_label, valid_data, valid_label = get_data.load_2class()

train_data = sklearn.preprocessing.normalize(train_data)
test_data = sklearn.preprocessing.normalize(test_data)
valid_data = sklearn.preprocessing.normalize(valid_data)

train_label = np.array(train_label)
test_label = np.array(test_label)
valid_label = np.array(valid_label)

if(train_model == 'CNN'):
    CNN(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size)

if(train_model == 'LSTM'):
    LSTM(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size)

if(train_model == 'AT'):
    AT(train_data, train_label, test_data, test_label, valid_data, valid_label, learningrate, epochs_num, batch_size)

if(train_model == 'EN'):

    output_load()

    train_data_en = pd.read_csv('/home/wang/rong/wang/LU_tensor/2class/output/train.csv', header = None)
    test_data_en = pd.read_csv('/home/wang/rong/wang/LU_tensor/2class/output/test.csv', header = None)
    valid_data_en = pd.read_csv('/home/wang/rong/wang/LU_tensor/2class/output/valid.csv', header = None)

    train_data_en = np.array(train_data_en)
    test_data_en = np.array(test_data_en)
    valid_data_en = np.array(valid_data_en)

    #train_data_en = train_data_en.reshape(train_data_en.shape[0], 1, train_data_en.shape[1])
    #test_data_en = test_data_en.reshape(test_data_en.shape[0], 1, test_data_en.shape[1])
    #valid_data_en = valid_data_en.reshape(valid_data_en.shape[0], 1, valid_data_en.shape[1])

    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape = (3,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True, to_file='/home/wang/rong/wang/LU_tensor/model_graph.png')

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])
    history = model.fit(train_data_en, train_label, batch_size = batch_size, epochs = epochs_num)

    result_1 = model.predict(train_data_en)
    result_2 = model.predict(test_data_en)
    result_3 = model.predict(valid_data_en)
    
    train_score = []
    test_score = []
    valid_score = []

    train_score = update.metrics(result_1, train_label, "NSL-KDDTrain+_EN", "EN", "train")
    test_score = update.metrics(result_2, test_label, "NSL-KDDTest+_EN", "EN", "test")
    valid_score = update.metrics(result_3, valid_label, "NSL-KDDTest21_EN", "EN", "valid")

    update.result_update(train_score, test_score ,valid_score,"/home/wang/rong/wang/LU_tensor/2class/result/EN/")