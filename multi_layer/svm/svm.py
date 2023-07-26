import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
import numpy as np
import sklearn
import copy
sys.path.append("/home/wang/Desktop/thesis/multi_layer") 
import deal_label
import random
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
tf.compat.v1.disable_eager_execution()

learningrate = float(sys.argv[1])
epochs_num = int(sys.argv[2])
batch_size = int(sys.argv[3])
method = int(sys.argv[4])

train_data, train_label, test_data, test_label, test_21_data, test_21_label = get_data.load_multi_class()
tradaboost_target_data = train_data.copy(deep = True)
tradaboost_target_data = sklearn.preprocessing.normalize(tradaboost_target_data)
tradaboost_target_label = train_label.copy(deep = True)

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

train_label_rename = train_label.rename(columns = {list(train_label)[0]:'label'})
class_pool = pd.concat([train_data, train_label_rename], axis = 1)

cluster_train_data = train_data.copy(deep = True)
cluster_test_data = test_data.copy(deep = True)
cluster_test_21_data = test_21_data.copy(deep = True)

valid_label = train_label.copy(deep = True) #讓程式可運行而已

multi_class_train_label, multi_class_valid_label,\
multi_class_test_label, multi_class_test_21_label = deal_label.encoder(train_label, valid_label, test_label, test_21_label)

train_data = sklearn.preprocessing.normalize(train_data)
test_data = sklearn.preprocessing.normalize(test_data)
test_21_data = sklearn.preprocessing.normalize(test_21_data)

dos_train_label = train_label.copy()
probe_train_label = train_label.copy()
r2l_train_label = train_label.copy()
u2r_train_label = train_label.copy()

train_data_re = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
test_data_re = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
test_21_data_re = test_21_data.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

original_train_data = np.array(train_data, copy = True)
original_test_data = np.array(test_data, copy = True)
#print(np.shares_memory(original_train_data.shape, train_data_re))
#print(id(original_train_data))
#print(id(train_data_re))
tradaboost_source = np.array(original_train_data, copy = True)
original_train_data = original_train_data.reshape(original_train_data.shape[0], 1, original_train_data.shape[1])
#(122,64,32,16,2) ()
model = models.Sequential()
model.add(Bidirectional(LSTM(122, return_sequences=True, dropout=0.2), input_shape=(None, 122)))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(16, dropout=0.2)))
model.add(Dense(16, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(2, kernel_initializer = 'normal', activation = 'softmax'))

model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=learningrate), metrics = ['accuracy'])

train_result = [5] * train_data.shape[0]
test_result = [5] * test_data.shape[0]
test_21_result = [5] * test_21_data.shape[0]

######### dos classifier

history_dos = model.fit(original_train_data, multi_class_train_label.iloc[:, 1], batch_size = batch_size, epochs = epochs_num)

result_dos_train = model.predict(train_data_re)
result_dos_test = model.predict(test_data_re)
result_dos_test_21 = model.predict(test_21_data_re)

result_dos_train = np.argmax(result_dos_train, axis = 1)
result_dos_test = np.argmax(result_dos_test, axis = 1)
result_dos_test_21 = np.argmax(result_dos_test_21, axis = 1)

train_dos_list = []
train_not_dos_list = []
test_dos_list = []
test_not_dos_list = []
test_21_dos_list = []
test_21_not_dos_list = []

for index, result in enumerate(result_dos_train):
    if result == 0:
        train_not_dos_list.append(index)
    else:
        train_dos_list.append(index)

for index, result in enumerate(result_dos_test):
    if result == 0:
        test_not_dos_list.append(index)
    else:
        test_dos_list.append(index)

for index, result in enumerate(result_dos_test_21):
    if result == 0:
        test_21_not_dos_list.append(index)
    else:
        test_21_dos_list.append(index)

train_data = pd.DataFrame(train_data)
train_data = train_data.iloc[train_not_dos_list]
train_data_re = train_data.values.reshape(train_data.shape[0], 1, train_data.shape[1])

test_data = pd.DataFrame(test_data)
test_data = test_data.iloc[test_not_dos_list]
test_data_re = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

test_21_data = pd.DataFrame(test_21_data)
test_21_data = test_21_data.iloc[test_21_not_dos_list]
test_21_data_re = test_21_data.values.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

for index in range(len(train_dos_list)):
    train_result[train_dos_list[index]] = 1

for index in range(len(test_dos_list)):
    test_result[test_dos_list[index]] = 1

for index in range(len(test_21_dos_list)):
    test_21_result[test_21_dos_list[index]] = 1

######### normal classifier

history_nor = model.fit(original_train_data, multi_class_train_label.iloc[:, 0], batch_size = batch_size, epochs = epochs_num)

result_nor_train = model.predict(train_data_re)
result_nor_test = model.predict(test_data_re)
result_nor_test_21 = model.predict(test_21_data_re)

result_nor_train = np.argmax(result_nor_train, axis = 1)
result_nor_test = np.argmax(result_nor_test, axis = 1)
result_nor_test_21 = np.argmax(result_nor_test_21, axis = 1)

train_nor_list = []
train_not_nor_list = []
test_nor_list = []
test_not_nor_list = []
test_21_nor_list = []
test_21_not_nor_list = []

for index, result in enumerate(result_nor_train):
    if result == 0:
        train_not_nor_list.append(index)
    else:
        train_nor_list.append(index)

for index, result in enumerate(result_nor_test):
    if result == 0:
        test_not_nor_list.append(index)
    else:
        test_nor_list.append(index)

for index, result in enumerate(result_nor_test_21):
    if result == 0:
        test_21_not_nor_list.append(index)
    else:
        test_21_nor_list.append(index)

train_data = pd.DataFrame(train_data)
train_data = train_data.iloc[train_not_nor_list]
train_data_re = train_data.values.reshape(train_data.shape[0], 1, train_data.shape[1])

test_data = pd.DataFrame(test_data)
test_data = test_data.iloc[test_not_nor_list]
test_data_re = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

test_21_data = pd.DataFrame(test_21_data)
test_21_data = test_21_data.iloc[test_21_not_nor_list]
test_21_data_re = test_21_data.values.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

for index in range(len(train_nor_list)):
    train_result[train_not_dos_list[train_nor_list[index]]] = 0

for index in range(len(test_nor_list)):
    test_result[test_not_dos_list[test_nor_list[index]]] = 0

for index in range(len(test_21_nor_list)):
    test_21_result[test_21_not_dos_list[test_21_nor_list[index]]] = 0

######### probe classifier

history_probe = model.fit(original_train_data, multi_class_train_label.iloc[:, 2], batch_size = batch_size, epochs = epochs_num)

result_probe_train = model.predict(train_data_re)
result_probe_test = model.predict(test_data_re)
result_probe_test_21 = model.predict(test_21_data_re)

result_probe_train = np.argmax(result_probe_train, axis = 1)
result_probe_test = np.argmax(result_probe_test, axis = 1)
result_probe_test_21 = np.argmax(result_probe_test_21, axis = 1)

train_probe_list = []
train_not_probe_list = []
test_probe_list = []
test_not_probe_list = []
test_21_probe_list = []
test_21_not_probe_list = []

for index, result in enumerate(result_probe_train):
    if result == 0:
        train_not_probe_list.append(index)
    else:
        train_probe_list.append(index)

for index, result in enumerate(result_probe_test):
    if result == 0:
        test_not_probe_list.append(index)
    else:
        test_probe_list.append(index)

for index, result in enumerate(result_probe_test_21):
    if result == 0:
        test_21_not_probe_list.append(index)
    else:
        test_21_probe_list.append(index)

train_data = pd.DataFrame(train_data)
train_data = train_data.iloc[train_not_probe_list]
train_data_re = train_data.values.reshape(train_data.shape[0], 1, train_data.shape[1])

test_data = pd.DataFrame(test_data)
test_data = test_data.iloc[test_not_probe_list]
test_data_re = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

test_21_data = pd.DataFrame(test_21_data)
test_21_data = test_21_data.iloc[test_21_not_probe_list]
test_21_data_re = test_21_data.values.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

for index in range(len(train_probe_list)):
    train_result[train_not_dos_list[train_not_nor_list[train_probe_list[index]]]] = 2

for index in range(len(test_probe_list)):
    test_result[test_not_dos_list[test_not_nor_list[test_probe_list[index]]]] = 2

for index in range(len(test_21_probe_list)):
    test_21_result[test_21_not_dos_list[test_21_not_nor_list[test_21_probe_list[index]]]] = 2

######### r2l classifier

history_r2l = model.fit(original_train_data, multi_class_train_label.iloc[:, 3], batch_size = batch_size, epochs = epochs_num)

result_r2l_train = model.predict(train_data_re)
result_r2l_test = model.predict(test_data_re)
result_r2l_test_21 = model.predict(test_21_data_re)

result_r2l_train = np.argmax(result_r2l_train, axis = 1)
result_r2l_test = np.argmax(result_r2l_test, axis = 1)
result_r2l_test_21 = np.argmax(result_r2l_test_21, axis = 1)

train_r2l_list = []
train_not_r2l_list = []
test_r2l_list = []
test_not_r2l_list = []
test_21_r2l_list = []
test_21_not_r2l_list = []

for index, result in enumerate(result_r2l_train):
    if result == 0:
        train_not_r2l_list.append(index)
    else:
        train_r2l_list.append(index)

for index, result in enumerate(result_r2l_test):
    if result == 0:
        test_not_r2l_list.append(index)
    else:
        test_r2l_list.append(index)

for index, result in enumerate(result_r2l_test_21):
    if result == 0:
        test_21_not_r2l_list.append(index)
    else:
        test_21_r2l_list.append(index)

train_data = pd.DataFrame(train_data)
train_data = train_data.iloc[train_not_r2l_list]
train_data_re = train_data.values.reshape(train_data.shape[0], 1, train_data.shape[1])

test_data = pd.DataFrame(test_data)
test_data = test_data.iloc[test_not_r2l_list]
test_data_re = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

test_21_data = pd.DataFrame(test_21_data)
test_21_data = test_21_data.iloc[test_21_not_r2l_list]
test_21_data_re = test_21_data.values.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

for index in range(len(train_r2l_list)):
    train_result[train_not_dos_list[train_not_nor_list[train_not_probe_list[train_r2l_list[index]]]]] = 3

for index in range(len(test_r2l_list)):
    test_result[test_not_dos_list[test_not_nor_list[test_not_probe_list[test_r2l_list[index]]]]] = 3

for index in range(len(test_21_r2l_list)):
    test_21_result[test_21_not_dos_list[test_21_not_nor_list[test_21_not_probe_list[test_21_r2l_list[index]]]]] = 3

######### u2r classifier

history_u2r = model.fit(original_train_data, multi_class_train_label.iloc[:, 4], batch_size = batch_size, epochs = epochs_num)

result_u2r_train = model.predict(train_data_re)
result_u2r_test = model.predict(test_data_re)
result_u2r_test_21 = model.predict(test_21_data_re)

result_u2r_train = np.argmax(result_u2r_train, axis = 1)
result_u2r_test = np.argmax(result_u2r_test, axis = 1)
result_u2r_test_21 = np.argmax(result_u2r_test_21, axis = 1)

train_u2r_list = []
train_not_u2r_list = []
test_u2r_list = []
test_not_u2r_list = []
test_21_u2r_list = []
test_21_not_u2r_list = []

for index, result in enumerate(result_u2r_train):
    if result == 0:
        train_not_u2r_list.append(index)
    else:
        train_u2r_list.append(index)

for index, result in enumerate(result_u2r_test):
    if result == 0:
        test_not_u2r_list.append(index)
    else:
        test_u2r_list.append(index)

for index, result in enumerate(result_u2r_test_21):
    if result == 0:
        test_21_not_u2r_list.append(index)
    else:
        test_21_u2r_list.append(index)

train_data = pd.DataFrame(train_data)
train_data = train_data.iloc[train_not_u2r_list]
train_data_re = train_data.values.reshape(train_data.shape[0], 1, train_data.shape[1])

test_data = pd.DataFrame(test_data)
test_data = test_data.iloc[test_not_u2r_list]
test_data_re = test_data.values.reshape(test_data.shape[0], 1, test_data.shape[1])

test_21_data = pd.DataFrame(test_21_data)
test_21_data = test_21_data.iloc[test_21_not_u2r_list]
test_21_data_re = test_21_data.values.reshape(test_21_data.shape[0], 1, test_21_data.shape[1])

for index in range(len(train_u2r_list)):
    train_result[train_not_dos_list[train_not_nor_list[train_not_probe_list[train_not_r2l_list[train_u2r_list[index]]]]]] = 4

for index in range(len(test_u2r_list)):
    test_result[test_not_dos_list[test_not_nor_list[test_not_probe_list[test_not_r2l_list[test_u2r_list[index]]]]]] = 4

for index in range(len(test_21_u2r_list)):
    test_21_result[test_21_not_dos_list[test_21_not_nor_list[test_21_not_probe_list[test_21_not_r2l_list[test_21_u2r_list[index]]]]]] = 4

##### find '5' class in result
train_cluster_index = [i for i,x in enumerate(train_result) if x == 5]
test_cluster_index = [i for i,x in enumerate(test_result) if x == 5]
test_21_cluster_index = [i for i,x in enumerate(test_21_result) if x == 5]

##### cluster train
train_data.reset_index(inplace = True)
train_data.drop('index', axis = 1, inplace = True)
test_data.reset_index(inplace = True)
test_data.drop('index', axis = 1, inplace = True)
test_21_data.reset_index(inplace = True)
test_21_data.drop('index', axis = 1, inplace = True)


from sklearn.linear_model import RidgeClassifier
from adapt.instance_based import TrAdaBoost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

tra_model = TrAdaBoost(SVC(kernel="poly",probability=True), n_estimators=10, Xt=tradaboost_target_data, yt=tradaboost_target_label)
tra_model.fit(tradaboost_source, train_label)

#result_1 = tra_model.predict(train_data)
result_2 = tra_model.predict(test_data)
result_3 = tra_model.predict(test_21_data)

#for i, row in train_data.iterrows():
#    first_match = train_result.index(5)
#    train_result[first_match] = result_1[i]

for i, row in test_data.iterrows():
    first_match = test_result.index(5)
    test_result[first_match] = result_2[i]

for i, row in test_21_data.iterrows():
    first_match = test_21_result.index(5)
    test_21_result[first_match] = result_3[i]
    

#train_acu = accuracy_score(train_result, train_label)
test_acu = accuracy_score(test_result, test_label)
test_21_acu = accuracy_score(test_21_result, test_21_label)

#train_ps = precision_score(train_result, train_label, average = 'macro')
test_ps = precision_score(test_result, test_label, average = 'macro')
test_21_ps = precision_score(test_21_result, test_21_label, average = 'macro')

#train_rs = recall_score(train_result, train_label, average = 'macro')
test_rs = recall_score(test_result, test_label, average = 'macro')
test_21_rs = recall_score(test_21_result, test_21_label, average = 'macro')

#train_f1 = f1_score(train_result, train_label, average = 'macro')
test_f1 = f1_score(test_result, test_label, average = 'macro')
test_21_f1 = f1_score(test_21_result, test_21_label, average = 'macro')

with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_acc.txt', 'a') as file:
    file.write("%f\n" %test_acu)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_ps.txt', 'a') as file:
    file.write("%f\n" %test_ps)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_rc.txt', 'a') as file:
    file.write("%f\n" %test_rs)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_f1.txt', 'a') as file:
    file.write("%f\n" %test_f1)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_21_acc.txt', 'a') as file:
    file.write("%f\n" %test_21_acu)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
    file.write("%f\n" %test_21_ps)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
    file.write("%f\n" %test_21_rs)
with open(r'/home/wang/Desktop/thesis/multi_layer/svm/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
    file.write("%f\n" %test_21_f1)
