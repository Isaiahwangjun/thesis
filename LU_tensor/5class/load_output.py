import pandas as pd
import numpy as np


def output_load():

    # train_data_AT = []
    # train_data_CNN = []
    # train_data_LSTM = []

    # f = open('/home/wang/rong/wang/LU_tensor/output/AT_train.txt')
    # for line in f:
    #     train_data_AT.append(line.strip('\n'))
    
    # f = open('/home/wang/rong/wang/LU_tensor/output/CNN_train.txt')
    # for line in f:
    #     train_data_CNN.append(line.strip('\n'))

    # f = open('/home/wang/rong/wang/LU_tensor/output/LSTM_train.txt')
    # for line in f:
    #     train_data_LSTM.append(line.strip('\n'))

    test_data_AT = []
    test_data_CNN = []
    test_data_LSTM = []

    f = open('/home/wang/rong/wang/LU_tensor/5class/output/at_test.txt')
    for line in f:
        test_data_AT.append(line.strip('\n'))
    
    f = open('/home/wang/rong/wang/LU_tensor/5class/output/cnn_test.txt')
    for line in f:
        test_data_CNN.append(line.strip('\n'))

    f = open('/home/wang/rong/wang/LU_tensor/5class/output/lstm_test.txt')
    for line in f:
        test_data_LSTM.append(line.strip('\n'))

    # valid_data_AT = []
    # valid_data_CNN = []
    # valid_data_LSTM = []

    # f = open('/home/wang/rong/wang/LU_tensor/output/AT_valid.txt')
    # for line in f:
    #     valid_data_AT.append(line.strip('\n'))
    
    # f = open('/home/wang/rong/wang/LU_tensor/output/CNN_valid.txt')
    # for line in f:
    #     valid_data_CNN.append(line.strip('\n'))

    # f = open('/home/wang/rong/wang/LU_tensor/output/LSTM_valid.txt')
    # for line in f:
    #     valid_data_LSTM.append(line.strip('\n'))

    
    # train = pd.DataFrame([train_data_AT, train_data_CNN, train_data_LSTM])
    # train = train.T
    # train.to_csv('/home/wang/rong/wang/LU_tensor/output/train.csv', header = False, index = False)

    test = pd.DataFrame([test_data_AT, test_data_CNN, test_data_LSTM])
    test = test.T
    test.to_csv('/home/wang/rong/wang/LU_tensor/5class/output/test.csv', header = False, index = False)

    # valid = pd.DataFrame([valid_data_AT, valid_data_CNN, valid_data_LSTM])
    # valid = valid.T
    # valid.to_csv('/home/wang/rong/wang/LU_tensor/output/valid.csv', header = False, index = False)

