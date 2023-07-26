import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
from cnn import CNN
from lstm import LSTM
from attention import AT
from load_output import output_load, output_load_3dim
import tensorflow as tf
from tensorflow.keras import models, layers
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN

train_model = str(sys.argv[1])
learningrate = float(sys.argv[2])
epochs_num = int(sys.argv[3])
batch_size = int(sys.argv[4])
method = int(sys.argv[5])
dim = int(sys.argv[6])

train_data, train_label, test_data, test_label, test_21_data, test_21_label = get_data.load_multi_class()

if method == 1:
    over_samples = SMOTE(random_state=1234)
    train_data,train_label = over_samples.fit_resample(train_data, train_label)
    au_name = 'smote'
elif method == 2:
    over_samples = ADASYN(random_state=1234)
    train_data,train_label = over_samples.fit_resample(train_data, train_label)
    au_name = 'adasyn'
else :
    au_name = 'original'

train_data = sklearn.preprocessing.normalize(train_data)
test_data = sklearn.preprocessing.normalize(test_data)
test_21_data = sklearn.preprocessing.normalize(test_21_data)

train_label = np.array(train_label)
test_label = np.array(test_label)
test_21_label = np.array(test_21_label)

if(train_model == 'CNN'):
    CNN(train_data, train_label, test_data, test_label, test_21_data, test_21_label, learningrate, epochs_num, batch_size, au_name, dim)

if(train_model == 'LSTM'):
    LSTM(train_data, train_label, test_data, test_label, test_21_data, test_21_label, learningrate, epochs_num, batch_size, au_name, dim)

if(train_model == 'AT'):
    AT(train_data, train_label, test_data, test_label, test_21_data, test_21_label, learningrate, epochs_num, batch_size, au_name, dim)

if(train_model == 'EN'):

    if(dim == 15):
        output_load(au_name, dim)
        train_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/train_' + str(dim) + au_name + '.csv', header = None)
        test_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_' + str(dim) + au_name + '.csv', header = None)
        test_21_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_21_' + str(dim) + au_name + '.csv', header = None)

    if(dim == 3):
        output_load_3dim(au_name, dim)
        train_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/train_' + str(dim) + au_name + '.csv', header = None)
        test_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_' + str(dim) + au_name + '.csv', header = None)
        test_21_data_en = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_21_' + str(dim) + au_name + '.csv', header = None)

    train_data_en = np.array(train_data_en)
    test_data_en = np.array(test_data_en)
    test_21_data_en = np.array(test_21_data_en)

    #train_data_en = train_data_en.reshape(train_data_en.shape[0], 1, train_data_en.shape[1])
    #test_data_en = test_data_en.reshape(test_data_en.shape[0], 1, test_data_en.shape[1])
    #valid_data_en = valid_data_en.reshape(valid_data_en.shape[0], 1, valid_data_en.shape[1])

    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape = (dim,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True, to_file='/home/wang/rong/wang/LU_tensor/model_graph.png')

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])
    history = model.fit(train_data_en, train_label, batch_size = batch_size, epochs = epochs_num)

    result_1 = model.predict(train_data_en)
    result_2 = model.predict(test_data_en)
    result_3 = model.predict(test_21_data_en)

    result_1 = np.argmax(result_1, axis = 1)
    result_2 = np.argmax(result_2, axis = 1)
    result_3 = np.argmax(result_3, axis = 1)

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
    
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_acc.txt', 'a') as file:
        file.write("%f\n" %test_acc)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_ps.txt', 'a') as file:
        file.write("%f\n" %test_ps)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_rs.txt', 'a') as file:
        file.write("%f\n" %test_rs)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_f1.txt', 'a') as file:
        file.write("%f\n" %test_f1)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_21_acc.txt', 'a') as file:
        file.write("%f\n" %test_21_acc)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_21_ps.txt', 'a') as file:
        file.write("%f\n" %test_21_ps)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_21_rs.txt', 'a') as file:
        file.write("%f\n" %test_21_rs)
    with open(r'/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_' + str(dim) + au_name + 'test_21_f1.txt', 'a') as file:
        file.write("%f\n" %test_21_f1)
