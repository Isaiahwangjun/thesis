#只用tradaboost-rf

import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
import numpy as np 
import sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN

method = int(sys.argv[1])

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

train_data = sklearn.preprocessing.normalize(train_data)
test_data = sklearn.preprocessing.normalize(test_data)
test_21_data = sklearn.preprocessing.normalize(test_21_data)

from adapt.instance_based import TrAdaBoost
from sklearn.ensemble import RandomForestClassifier

tra_model = TrAdaBoost(RandomForestClassifier(n_estimators = 120, criterion = 'entropy', max_depth=75), n_estimators=10, Xt=tradaboost_target_data, yt=tradaboost_target_label)
tra_model.fit(train_data, train_label)

train_result = tra_model.predict(train_data)
test_result = tra_model.predict(test_data)
test_21_result = tra_model.predict(test_21_data)

train_acu = accuracy_score(train_result, train_label)
test_acu = accuracy_score(test_result, test_label)
test_21_acu = accuracy_score(test_21_result, test_21_label)

train_ps = precision_score(train_result, train_label, average = 'macro')
test_ps = precision_score(test_result, test_label, average = 'macro')
test_21_ps = precision_score(test_21_result, test_21_label, average = 'macro')

train_rs = recall_score(train_result, train_label, average = 'macro')
test_rs = recall_score(test_result, test_label, average = 'macro')
test_21_rs = recall_score(test_21_result, test_21_label, average = 'macro')

train_f1 = f1_score(train_result, train_label, average = 'macro')
test_f1 = f1_score(test_result, test_label, average = 'macro')
test_21_f1 = f1_score(test_21_result, test_21_label, average = 'macro')


with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_acc.txt', 'a') as file:
    file.write("%f\n" %test_acu)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_ps.txt', 'a') as file:
    file.write("%f\n" %test_ps)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_rc.txt', 'a') as file:
    file.write("%f\n" %test_rs)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_f1.txt', 'a') as file:
    file.write("%f\n" %test_f1)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_21_acc.txt', 'a') as file:
    file.write("%f\n" %test_21_acu)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_21_ps.txt', 'a') as file:
    file.write("%f\n" %test_21_ps)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_21_rc.txt', 'a') as file:
    file.write("%f\n" %test_21_rs)
with open(r'/home/wang/Desktop/thesis/ablation/tra/result/test_21_f1.txt', 'a') as file:
    file.write("%f\n" %test_21_f1)
