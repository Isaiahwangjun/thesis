import pandas as pd


def encoder(train_label, valid_label, test_label, test_21_label):

    multi_class_train_label = pd.concat([train_label, train_label, train_label, train_label, train_label], axis = 1)

    multi_class_train_label.iloc[:, 0] = multi_class_train_label.iloc[:, 0].replace([1, 2, 3, 4], 5)
    multi_class_train_label.iloc[:, 0] = multi_class_train_label.iloc[:, 0].replace([0], 1)
    multi_class_train_label.iloc[:, 0] = multi_class_train_label.iloc[:, 0].replace([5], 0)
    multi_class_train_label.iloc[:, 1] = multi_class_train_label.iloc[:, 1].replace([0, 2, 3, 4], 0)
    multi_class_train_label.iloc[:, 1] = multi_class_train_label.iloc[:, 1].replace([1], 1)
    multi_class_train_label.iloc[:, 2] = multi_class_train_label.iloc[:, 2].replace([0, 1, 3, 4], 0)
    multi_class_train_label.iloc[:, 2] = multi_class_train_label.iloc[:, 2].replace([2], 1)
    multi_class_train_label.iloc[:, 3] = multi_class_train_label.iloc[:, 3].replace([0, 1, 2, 4], 0)
    multi_class_train_label.iloc[:, 3] = multi_class_train_label.iloc[:, 3].replace([3], 1)
    multi_class_train_label.iloc[:, 4] = multi_class_train_label.iloc[:, 4].replace([0, 1, 2, 3], 0)
    multi_class_train_label.iloc[:, 4] = multi_class_train_label.iloc[:, 4].replace([4], 1)


    multi_class_valid_label = pd.concat([valid_label, valid_label, valid_label, valid_label, valid_label], axis = 1)

    
    multi_class_valid_label.iloc[:, 0] = multi_class_valid_label.iloc[:, 0].replace([1, 2, 3, 4], 5)
    multi_class_valid_label.iloc[:, 0] = multi_class_valid_label.iloc[:, 0].replace([0], 1)
    multi_class_valid_label.iloc[:, 0] = multi_class_valid_label.iloc[:, 0].replace([5], 0)
    multi_class_valid_label.iloc[:, 1] = multi_class_valid_label.iloc[:, 1].replace([0, 2, 3, 4], 0)
    multi_class_valid_label.iloc[:, 1] = multi_class_valid_label.iloc[:, 1].replace([1], 1)
    multi_class_valid_label.iloc[:, 2] = multi_class_valid_label.iloc[:, 2].replace([0, 1, 3, 4], 0)
    multi_class_valid_label.iloc[:, 2] = multi_class_valid_label.iloc[:, 2].replace([2], 1)
    multi_class_valid_label.iloc[:, 3] = multi_class_valid_label.iloc[:, 3].replace([0, 1, 2, 4], 0)
    multi_class_valid_label.iloc[:, 3] = multi_class_valid_label.iloc[:, 3].replace([3], 1)
    multi_class_valid_label.iloc[:, 4] = multi_class_valid_label.iloc[:, 4].replace([0, 1, 2, 3], 0)
    multi_class_valid_label.iloc[:, 4] = multi_class_valid_label.iloc[:, 4].replace([4], 1)


    multi_class_test_label = pd.concat([test_label, test_label, test_label, test_label, test_label], axis = 1)

    multi_class_test_label.iloc[:, 0] = multi_class_test_label.iloc[:, 0].replace([1, 2, 3, 4], 5)
    multi_class_test_label.iloc[:, 0] = multi_class_test_label.iloc[:, 0].replace([0], 1)
    multi_class_test_label.iloc[:, 0] = multi_class_test_label.iloc[:, 0].replace([5], 0)
    multi_class_test_label.iloc[:, 1] = multi_class_test_label.iloc[:, 1].replace([0, 2, 3, 4], 0)
    multi_class_test_label.iloc[:, 1] = multi_class_test_label.iloc[:, 1].replace([1], 1)
    multi_class_test_label.iloc[:, 2] = multi_class_test_label.iloc[:, 2].replace([0, 1, 3, 4], 0)
    multi_class_test_label.iloc[:, 2] = multi_class_test_label.iloc[:, 2].replace([2], 1)
    multi_class_test_label.iloc[:, 3] = multi_class_test_label.iloc[:, 3].replace([0, 1, 2, 4], 0)
    multi_class_test_label.iloc[:, 3] = multi_class_test_label.iloc[:, 3].replace([3], 1)
    multi_class_test_label.iloc[:, 4] = multi_class_test_label.iloc[:, 4].replace([0, 1, 2, 3], 0)
    multi_class_test_label.iloc[:, 4] = multi_class_test_label.iloc[:, 4].replace([4], 1)
    

    multi_class_test_21_label = pd.concat([test_21_label, test_21_label, test_21_label, test_21_label, test_21_label], axis = 1)

    multi_class_test_21_label.iloc[:, 0] = multi_class_test_21_label.iloc[:, 0].replace([1, 2, 3, 4], 5)
    multi_class_test_21_label.iloc[:, 0] = multi_class_test_21_label.iloc[:, 0].replace([0], 1)
    multi_class_test_21_label.iloc[:, 0] = multi_class_test_21_label.iloc[:, 0].replace([5], 0)
    multi_class_test_21_label.iloc[:, 1] = multi_class_test_21_label.iloc[:, 1].replace([0, 2, 3, 4], 0)
    multi_class_test_21_label.iloc[:, 1] = multi_class_test_21_label.iloc[:, 1].replace([1], 1)
    multi_class_test_21_label.iloc[:, 2] = multi_class_test_21_label.iloc[:, 2].replace([0, 1, 3, 4], 0)
    multi_class_test_21_label.iloc[:, 2] = multi_class_test_21_label.iloc[:, 2].replace([2], 1)
    multi_class_test_21_label.iloc[:, 3] = multi_class_test_21_label.iloc[:, 3].replace([0, 1, 2, 4], 0)
    multi_class_test_21_label.iloc[:, 3] = multi_class_test_21_label.iloc[:, 3].replace([3], 1)
    multi_class_test_21_label.iloc[:, 4] = multi_class_test_21_label.iloc[:, 4].replace([0, 1, 2, 3], 0)
    multi_class_test_21_label.iloc[:, 4] = multi_class_test_21_label.iloc[:, 4].replace([4], 1)

    return multi_class_train_label, multi_class_valid_label, multi_class_test_label, multi_class_test_21_label