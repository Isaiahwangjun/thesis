from xgboost import XGBClassifier
import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
import numpy as np 
import sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
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

model = XGBClassifier(n_estimators=120, learning_rate= 0.001, max_depth=75, random_state=7)
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

with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_acc.txt', 'a') as file:
    file.write("%f\n" %test_acc)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_ps.txt', 'a') as file:
    file.write("%f\n" %test_ps)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_rc.txt', 'a') as file:
    file.write("%f\n" %test_rc)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_f1.txt', 'a') as file:
    file.write("%f\n" %test_f1)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_21_acc.txt', 'a') as file:
    file.write("%f\n" %test_21_acc)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
    file.write("%f\n" %test_21_ps)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
    file.write("%f\n" %test_21_rc)
with open(r'/home/wang/Desktop/thesis/xgboost/result/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
    file.write("%f\n" %test_21_f1)
