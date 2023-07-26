import pandas as pd
import numpy as np
import csv


def output_load(name, dim):

    ##### merge train data
    train_data_AT = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/at_train_' + str(dim) + name + '.txt', header = None)
    train_data_CNN = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_train_' + str(dim) + name + '.txt', header = None)
    train_data_LSTM = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/lstm_train_' + str(dim) + name + '.txt', header = None)

    train = pd.concat([train_data_AT, train_data_CNN, train_data_LSTM], axis = 1)
    train.to_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/train_' + str(dim) + name + '.csv', header=False, index=False)


    ##### merge test data
    test_data_AT = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/at_test_' + str(dim) + name + '.txt', header = None)
    test_data_CNN = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_' + str(dim) + name + '.txt', header = None)
    test_data_LSTM = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/lstm_test_' + str(dim) + name + '.txt', header = None)

    test = pd.concat([test_data_AT, test_data_CNN, test_data_LSTM], axis = 1)
    test.to_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_' + str(dim) + name + '.csv', header = False, index = False)


    ##### merge test_21 data
    test_21_data_AT = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/at_test_21_' + str(dim) + name + '.txt', header = None)
    test_21_data_CNN = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/cnn_test_21_' + str(dim) + name + '.txt', header = None)
    test_21_data_LSTM = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/lstm_test_21_' + str(dim) + name + '.txt', header = None)


    test_21 = pd.concat([test_21_data_AT, test_21_data_CNN, test_21_data_LSTM], axis = 1)
    test_21.to_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/output/test_21_' + str(dim) + name + '.csv', header = False, index = False)

def output_load_3dim(name, dim):

    ##### merge train data
    train_data_AT = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/at_train_' + str(dim) + name + '.txt', header = None)
    train_data_CNN = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/cnn_train_' + str(dim) + name + '.txt', header = None)
    train_data_LSTM = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/lstm_train_' + str(dim) + name + '.txt', header = None)

    train = pd.concat([train_data_AT, train_data_CNN, train_data_LSTM], axis = 1)
    train.to_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/train_' + str(dim) + name + '.csv', header=False, index=False)


    ##### merge test data
    test_data_AT = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/at_test_' + str(dim) + name + '.txt', header = None)
    test_data_CNN = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/cnn_test_' + str(dim) + name + '.txt', header = None)
    test_data_LSTM = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/lstm_test_' + str(dim) + name + '.txt', header = None)

    test = pd.concat([test_data_AT, test_data_CNN, test_data_LSTM], axis = 1)
    test.to_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/test_' + str(dim) + name + '.csv', header = False, index = False)


    ##### merge test_21 data
    test_21_data_AT = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/at_test_21_' + str(dim) + name + '.txt', header = None)
    test_21_data_CNN = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/cnn_test_21_' + str(dim) + name + '.txt', header = None)
    test_21_data_LSTM = pd.read_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/lstm_test_21_' + str(dim) + name + '.txt', header = None)


    test_21 = pd.concat([test_21_data_AT, test_21_data_CNN, test_21_data_LSTM], axis = 1)
    test_21.to_csv('/home/wang/rong/wang/LU_tensor/augmentation/output/test_21_' + str(dim) + name + '.csv', header = False, index = False)

