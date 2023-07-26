import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import copy

### deal train
data = pd.read_csv(r'/home/wang/rong/NSLKDD/origin/train.txt', header = None)

train_data = data.iloc[:, 0:41]
train_label = data.iloc[:, 41]

ct = ColumnTransformer([("1", OneHotEncoder(), [1, 2, 3])], remainder = 'passthrough')
train_data = ct.fit_transform(train_data)
train_data = pd.DataFrame(train_data)
train_data.to_csv('/home/wang/rong/NSLKDD/onehot/train_data.txt', header = None, index = None)


train_label = pd.DataFrame(train_label)
train_label = train_label.apply(LabelEncoder().fit_transform)
train_label = train_label.iloc[:, 0]
train_label = train_label.tolist()
train_label_copy = copy.deepcopy(train_label)

for i in range(len(train_label)):
    if train_label[i] == 0 or train_label[i] == 6 or train_label[i] == 9 or train_label[i] == 14 or train_label[i] == 18 or train_label[i] == 20:
        train_label[i] = 1
    elif train_label[i] == 17 or train_label[i] == 5 or train_label[i] == 10 or train_label[i] == 15:
        train_label[i] = 2
    elif train_label[i] == 2 or train_label[i] == 3 or train_label[i] == 4 or train_label[i] == 13 or train_label[i] == 8 or train_label[i] == 21 or train_label[i] == 22 or train_label[i] == 19:
        train_label[i] = 3
    elif train_label[i] == 1 or train_label[i] == 7 or train_label[i] == 16 or train_label[i] == 12:
        train_label[i] = 4
    else:
        train_label[i] = 0

with open(r'/home/wang/rong/NSLKDD/onehot/multi_class/train_label.txt', 'w') as file:
    for i in train_label:
        file.write("%d\n" %i)

### deal test
data_test = pd.read_csv(r'/home/wang/rong/NSLKDD/origin/test.txt', header = None)

test_data = data_test.iloc[:, 0:41]
test_label = data_test.iloc[:, 41]

test_data = ct.transform(test_data)
test_data = pd.DataFrame(test_data)
test_data.to_csv('/home/wang/rong/NSLKDD/onehot/test_data.txt', header = None, index = None)


test_label = pd.DataFrame(test_label)
test_label = test_label.apply(LabelEncoder().fit_transform)
test_label = test_label.iloc[:, 0]
test_label = test_label.tolist()
test_label_copy = copy.deepcopy(test_label)

for i in range(len(test_label)):
    if test_label[i] == 1 or test_label[i] == 8 or test_label[i] == 14 or test_label[i] == 19 or test_label[i] == 27 or test_label[i] == 31 or test_label[i] == 0 or test_label[i] == 32 or test_label[i] == 21 or test_label[i] == 10:
        test_label[i] = 1
    elif test_label[i] == 25 or test_label[i] == 7 or test_label[i] == 15 or test_label[i] == 20 or test_label[i] == 11 or test_label[i] == 24:
        test_label[i] = 2
    elif test_label[i] == 4 or test_label[i] == 3 or test_label[i] == 6 or test_label[i] == 18 or test_label[i] == 12 or test_label[i] == 33 or test_label[i] == 35 or test_label[i] == 36 or test_label[i] == 29 or test_label[i] == 28 or test_label[i] == 34 or test_label[i] == 26 or test_label[i] == 13:
        test_label[i] = 3
    elif test_label[i] == 2 or test_label[i] == 9 or test_label[i] == 23 or test_label[i] == 17 or test_label[i] == 30 or test_label[i] == 37 or test_label[i] == 22 or test_label[i] == 5:
        test_label[i] = 4
    else:
        test_label[i] = 0
    
with open(r'/home/wang/rong/NSLKDD/onehot/multi_class/test_label.txt', 'w') as file:
    for i in test_label:
        file.write("%d\n" %i)


### deal test_21
data_test_21 = pd.read_csv(r'/home/wang/rong/NSLKDD/origin/test_21.txt', header = None)

test_21_data = data_test_21.iloc[:, 0:41]
test_21_label = data_test_21.iloc[:, 41]

test_21_data = ct.transform(test_21_data)
test_21_data = pd.DataFrame(test_21_data)
test_21_data.to_csv('/home/wang/rong/NSLKDD/onehot/test_21_data.txt', header = None, index = None)


test_21_label = pd.DataFrame(test_21_label)
test_21_label = test_21_label.apply(LabelEncoder().fit_transform)
test_21_label = test_21_label.iloc[:, 0]
test_21_label = test_21_label.tolist()
test_21_label_copy = copy.deepcopy(test_21_label)

for i in range(len(test_21_label)):
    if test_21_label[i] == 1 or test_21_label[i] == 8 or test_21_label[i] == 14 or test_21_label[i] == 19 or test_21_label[i] == 27 or test_21_label[i] == 31 or test_21_label[i] == 0 or test_21_label[i] == 32 or test_21_label[i] == 21 or test_21_label[i] == 10:
        test_21_label[i] = 1
    elif test_21_label[i] == 25 or test_21_label[i] == 7 or test_21_label[i] == 15 or test_21_label[i] == 20 or test_21_label[i] == 11 or test_21_label[i] == 24:
        test_21_label[i] = 2
    elif test_21_label[i] == 4 or test_21_label[i] == 3 or test_21_label[i] == 6 or test_21_label[i] == 18 or test_21_label[i] == 12 or test_21_label[i] == 33 or test_21_label[i] == 35 or test_21_label[i] == 36 or test_21_label[i] == 29 or test_21_label[i] == 28 or test_21_label[i] == 34 or test_21_label[i] == 26 or test_21_label[i] == 13:
        test_21_label[i] = 3
    elif test_21_label[i] == 2 or test_21_label[i] == 9 or test_21_label[i] == 23 or test_21_label[i] == 17 or test_21_label[i] == 30 or test_21_label[i] == 37 or test_21_label[i] == 22 or test_21_label[i] == 5:
        test_21_label[i] = 4
    else:
        test_21_label[i] = 0
    
with open(r'/home/wang/rong/NSLKDD/onehot/multi_class/test_21_label.txt', 'w') as file:
    for i in test_21_label:
        file.write("%d\n" %i)

#2class label
for i in range(len(train_label_copy)):
    if train_label_copy[i] == 11:
        train_label_copy[i] = 0
    else:
        train_label_copy[i] = 1

with open(r'/home/wang/rong/NSLKDD/onehot/2class/train_label.txt', 'w') as file:
    for i in train_label_copy:
        file.write("%d\n" %i)

for i in range(len(test_label_copy)):
    if test_label_copy[i] == 16:
        test_label_copy[i] = 0
    else:
        test_label_copy[i] = 1

with open(r'/home/wang/rong/NSLKDD/onehot/2class/test_label.txt', 'w') as file:
    for i in test_label_copy:
        file.write("%d\n" %i)

for i in range(len(test_21_label_copy)):
    if test_21_label_copy[i] == 16:
        test_21_label_copy[i] = 0
    else:
        test_21_label_copy[i] = 1

with open(r'/home/wang/rong/NSLKDD/onehot/2class/test_21_label.txt', 'w') as file:
    for i in test_21_label_copy:
        file.write("%d\n" %i)