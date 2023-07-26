import pandas as pd

def load_2class():

    train_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/train_data.txt', header = None)
    train_label = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/2class/train_label', header = None)

    test_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/test_data.txt', header = None)
    test_label = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onrhot/2class/test_label', header = None)

    test_21_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/test_21_data.txt', header = None)
    test_21_label = pd.read_csv('/home/wang/Desktop/thesis/onrhot/2class/test_21_label', header = None)

    return train_data, train_label, test_data, test_label , test_21_data, test_21_label

def load_multi_class():

    train_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/train_data.txt', header = None)
    train_label = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/multi_class/train_label.txt', header = None)

    test_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/test_data.txt', header = None)
    test_label = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/multi_class/test_label.txt', header = None)

    test_21_data = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/test_21_data.txt', header = None)
    test_21_label = pd.read_csv('/home/wang/Desktop/thesis/NSLKDD/onehot/multi_class/test_21_label.txt', header = None)

    return train_data, train_label, test_data, test_label, test_21_data, test_21_label

