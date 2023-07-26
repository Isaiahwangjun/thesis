import csv


def load_NSLKDD():

    with open( "/home/wang/rong/wang/LU_tensor/train/x.csv") as txtfile:
        rows = csv.reader(txtfile)
        data = []
        for row in rows:
            data.append(row[0:41])  
    with open( "/home/wang/rong/wang/LU_tensor/train/y.csv") as txtfile:
        rows = csv.reader(txtfile)
        label = []
        for row in rows:
            label.append(row[0])

    train_data = data
    train_label = label

    with open( "/home/wang/rong/wang/LU_tensor/test/x.csv") as txtfile:
        rows = csv.reader(txtfile)
        data = []
        for row in rows:
            data.append(row[0:41])  
    with open( "/home/wang/rong/wang/LU_tensor/test/y.csv") as txtfile:
        rows = csv.reader(txtfile)
        label = []
        for row in rows:
            label.append(row[0])

    test_data = data
    test_label = label

    with open( "/home/wang/rong/wang/LU_tensor/valid/x.csv") as txtfile:
        rows = csv.reader(txtfile)
        data = []
        for row in rows:
            data.append(row[0:41])  
    with open( "/home/wang/rong/wang/LU_tensor/valid/y.csv") as txtfile:
        rows = csv.reader(txtfile)
        label = []
        for row in rows:
            label.append(row[0])

    valid_data = data
    valid_label = label


    return train_data, train_label, test_data, test_label, valid_data, valid_label
