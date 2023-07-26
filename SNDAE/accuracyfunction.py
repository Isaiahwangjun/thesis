from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def accuracy(y_test_ture, y_test_predic, y_test_proba, method, data, train_label):
    # accuarcy
    accuracy = accuracy_score(y_test_ture, y_test_predic)

    # f1_score
    f1score = f1_score(y_test_ture, y_test_predic, average='macro')

    # precision_score
    precision = precision_score(y_test_ture, y_test_predic, average='macro')

    # recall_score
    recall = recall_score(y_test_ture, y_test_predic, average='macro')

    value = roc_auc_score(y_test_ture, y_test_proba, multi_class="ovr", average="macro")
    with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(method)+'/'+ str(data) + 'area_value.txt', 'a') as file:
        file.write(str(value)+',\n')
    label_binarizer = LabelBinarizer().fit(train_label)
    y_onehot_test = label_binarizer.transform(y_test_ture)
    n_classes = len(np.unique(train_label))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    from sklearn.metrics import roc_curve, auc
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    with open('/home/wang/Desktop/thesis/SNDAE/result/'+str(method)+'/'+ str(data) + 'auc.txt', "ab") as f:
        np.savetxt(f, mean_tpr, fmt='%.5f')

    # classification_report
    # print("The testing classification_report is")
    # print(classification_report(y_test_ture, y_test_predic))
    return accuracy, precision, recall, f1score