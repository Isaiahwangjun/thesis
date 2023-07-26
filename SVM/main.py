import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
import numpy as np
from sklearn.svm import SVC
import sklearn
from sklearn.preprocessing import LabelBinarizer
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

test_acc_list = []
test_21_acc_list = []

test_ps_list = []
test_21_ps_list = []

test_rc_list = []
test_21_rc_list = []

test_f1_list = []
test_21_f1_list = []

def SVM(Kernel):

    model = SVC(kernel=Kernel,probability=True)
    model.fit(train_data, train_label)
    
    result_1 = model.predict(test_data)
    result_2 = model.predict(test_21_data)

    # result_3 = model.predict_proba(test_data)
    # value = roc_auc_score(test_label, result_3, multi_class="ovr", average="macro")
    # with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/area_value.txt', 'a') as file:
    #     file.write(str(value)+',\n')
    # label_binarizer = LabelBinarizer().fit(train_label)
    # y_onehot_test = label_binarizer.transform(test_label)
    # n_classes = len(np.unique(train_label))
    # fpr, tpr, roc_auc = dict(), dict(), dict()
    # from sklearn.metrics import roc_curve, auc
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], result_3[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr_grid = np.linspace(0.0, 1.0, 1000)
    # mean_tpr = np.zeros_like(fpr_grid)
    # for i in range(n_classes):
    #     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # mean_tpr /= n_classes

    # fpr["macro"] = fpr_grid
    # tpr["macro"] = mean_tpr
    # with open('/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/auc.txt', "ab") as f:
    #     np.savetxt(f, mean_tpr, fmt='%.5f')

    # result_4 = model.predict_proba(test_21_data)
    # value = roc_auc_score(test_21_label, result_4, multi_class="ovr", average="macro")
    # with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/area_21_value.txt', 'a') as file:
    #     file.write(str(value)+',\n')
    # label_binarizer = LabelBinarizer().fit(train_label)
    # y_onehot_test_21 = label_binarizer.transform(test_21_label)
    # n_classes = len(np.unique(train_label))
    # fpr, tpr, roc_auc = dict(), dict(), dict()
    # from sklearn.metrics import roc_curve, auc
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_onehot_test_21[:, i], result_4[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr_grid = np.linspace(0.0, 1.0, 1000)
    # mean_tpr = np.zeros_like(fpr_grid)
    # for i in range(n_classes):
    #     mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # mean_tpr /= n_classes

    # fpr["macro"] = fpr_grid
    # tpr["macro"] = mean_tpr
    # with open('/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/auc_21.txt', "ab") as f:
    #     np.savetxt(f, mean_tpr, fmt='%.5f')
    
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
    SVM("poly")

with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_acc.txt', 'a') as file:
    for i in test_acc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_ps.txt', 'a') as file:
    for i in test_ps_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_rc.txt', 'a') as file:
    for i in test_rc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_f1.txt', 'a') as file:
    for i in test_f1_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_21_acc.txt', 'a') as file:
    for i in test_21_acc_list:    
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
    for i in test_21_ps_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
    for i in test_21_rc_list:
        file.write("%f\n" %i)
with open(r'/home/wang/Desktop/thesis/SVM/result/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
    for i in test_21_f1_list:
        file.write("%f\n" %i)