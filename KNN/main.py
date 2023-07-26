import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN

method = int(sys.argv[1])

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

test_acc_list = []
test_21_acc_list = []

test_ps_list = []
test_21_ps_list = []

test_rc_list = []
test_21_rc_list = []

test_f1_list = []
test_21_f1_list = []

def KNN(k):

    model = KNeighborsClassifier(n_neighbors = k, n_jobs = -1)
    model.fit(train_data, train_label)

    result_1 = model.predict(test_data)
    result_2 = model.predict(test_21_data)

    test_acc = accuracy_score(test_label, result_1)
    test_21_acc = accuracy_score(test_21_label, result_2)
    
    test_ps = precision_score(test_label, result_1, average = 'macro')
    test_21_ps = precision_score(test_21_label, result_2, average = 'macro')

    test_rc = recall_score(test_label, result_1, average = 'macro')
    test_21_rc = recall_score(test_21_label, result_2, average = 'macro')

    test_f1 = f1_score(test_label, result_1, average = 'macro')
    test_21_f1 = f1_score(test_21_label, result_2, average = 'macro')

    test_acc_list.append(test_acc)
    test_21_acc_list.append(test_21_acc)

    test_ps_list.append(test_ps)
    test_21_ps_list.append(test_21_ps)

    test_rc_list.append(test_rc)
    test_21_rc_list.append(test_21_rc)

    test_f1_list.append(test_f1)
    test_21_f1_list.append(test_21_f1)

for i in range(1):
    KNN(10)


with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_acc.txt', 'a') as file:
    for i in test_acc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_ps.txt', 'a') as file:
    for i in test_ps_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_rc.txt', 'a') as file:
    for i in test_rc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_f1.txt', 'a') as file:
    for i in test_f1_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_21_acc.txt', 'a') as file:
    for i in test_21_acc_list:    
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
    for i in test_21_ps_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
    for i in test_21_rc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/KNN/result/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
    for i in test_21_f1_list:
        file.write("%f\n" %i)