from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import csv

def result_output(predict_label, model_name, data):
    with open('./output/' + str(model_name) +  "_" + str(data) + ".txt", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(predict_label)):
            if(predict_label[i] > 0.5):
                writer.writerow("1")
            else:
                writer.writerow("0")

def AUC(predict, answer, model_name, data):
    predict_label = []
    answer = np.array(answer).astype(np.float)
    
    for i in range(len(predict)):
        predict_label.append(predict[i][1])

    result_output(predict_label, model_name, data)
    
    fpr,tpr,threshold = roc_curve(answer, predict_label)
    auc = roc_auc_score(answer, predict_label)
    
    pig1 = []
    pig2 = []
    for i in range(len(threshold)):
        if float(threshold[i]) <= 0.1 :
            pig1.append(float(fpr[i]))
            pig2.append(float(tpr[i]))
            break
    for i in range(len(threshold)):
        if float(threshold[i]) <= 0.3:
            pig1.append(float(fpr[i]))
            pig2.append(float(tpr[i]))
            break
    for i in range(len(threshold)):
        if float(threshold[i]) <= 0.5 :
            pig1.append(float(fpr[i]))
            pig2.append(float(tpr[i]))
            break
    for i in range(len(threshold)):
        if float(threshold[i]) <= 0.7 :
            pig1.append(float(fpr[i]))
            pig2.append(float(tpr[i]))
            break
    for i in range(len(threshold)):
        if float(threshold[i]) <= 0.9 :
            pig1.append(float(fpr[i]))
            pig2.append(float(tpr[i]))
            break
    
    return auc, fpr, tpr, pig1, pig2

def plot(fpr, tpr, auc, pig1, pig2, filename):
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC1 curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.plot(pig1[0], pig2[0], color = 'blue', marker = "o", label = '10%-')
    plt.plot(pig1[1], pig2[1], color = 'blue', marker = "v", label = '30%-')
    plt.plot(pig1[2], pig2[2], color = 'blue', marker = "D", label = '50%-') 
    plt.plot(pig1[3], pig2[3], color = 'blue', marker = "|", label = '70%-') 
    plt.plot(pig1[4], pig2[4], color = 'blue', marker = "x", label = '90%-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + filename)
    plt.legend(loc = "lower right")
    plt.savefig('/home/wang/rong/wang/LU_tensor/2class/fig/' + filename + "_" + '_ROC.png')

def metrics(predict, answer, filename, model_name, data):
    TP = 0 
    TN = 0 
    FP = 0 
    FN = 0
    auc, fpr, tpr, pig1, pig2 = AUC(predict, answer, model_name, data)
    plot(fpr, tpr, auc, pig1, pig2, filename)

    for i in range(len(answer)):
        if(int(answer[i]) == 0 and (predict[i][1] < 0.5)):
            TN += 1
        elif(int(answer[i]) == 1 and (predict[i][1] < 0.5)):
            FN += 1
        elif(int(answer[i]) == 1 and (predict[i][1] >= 0.5)):
            TP += 1
        elif(int(answer[i]) == 0 and (predict[i][1] >= 0.5)):
            FP += 1

    Accuracy = (TP + TN) / (TP + TN + FP + FN) 
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1_score = 2 * Recall * Precision / (Recall + Precision)

    score = []
    score.append(Accuracy)
    score.append(Recall)  
    score.append(Precision)
    score.append(F1_score)
    score.append(auc)
    score.append(TP)
    score.append(TN)
    score.append(FP)
    score.append(FN)
    return score

def result_update(train_score, test_score, test_21_score, filename):
        fp=open(str(filename) + "train_accuracy_"  + ".txt","a")
        fp.write(str(train_score[0]) + "\n")
        fp.close()
        fp=open(str(filename) + "train_recall_" + ".txt","a")
        fp.write(str(train_score[1]) + "\n")
        fp.close()
        fp=open(str(filename) + "train_precision_" + ".txt","a")
        fp.write(str(train_score[2]) + "\n")
        fp.close()
        fp=open(str(filename) + "train_f1score_" + ".txt","a")
        fp.write(str(train_score[3]) + "\n")
        fp.close()
        fp=open(str(filename) + "train_AUC_" + ".txt","a")
        fp.write(str(train_score[4]) + "\n")
        fp.close()

        fp=open(str(filename) + "test_accuracy_" + ".txt","a")
        fp.write(str(test_score[0]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_recall_" + ".txt","a")
        fp.write(str(test_score[1]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_precision_" + ".txt","a")
        fp.write(str(test_score[2]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_f1score_" + ".txt","a")
        fp.write(str(test_score[3]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_AUC_" + ".txt","a")
        fp.write(str(test_score[4]) + "\n")
        fp.close()

        fp=open(str(filename) + "test_21_accuracy_" + ".txt","a")
        fp.write(str(test_21_score[0]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_21_recall_" + ".txt","a")
        fp.write(str(test_21_score[1]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_21_precision_" + ".txt","a")
        fp.write(str(test_21_score[2]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_21_f1score_" + ".txt","a")
        fp.write(str(test_21_score[3]) + "\n")
        fp.close()
        fp=open(str(filename) + "test_21_AUC_" + ".txt","a")
        fp.write(str(test_21_score[4]) + "\n")
        fp.close()
        fp=open(str(filename) + "train_metrice_" + ".txt","a")
        fp.write(str(train_score[5])+", "+str(train_score[6])+", "+str(train_score[7])+", "+str(train_score[8])+"\n")
        fp.close()
        fp=open(str(filename) + "test_metrice_" + ".txt","a")
        fp.write(str(test_score[5])+", "+str(test_score[6])+", "+str(test_score[7])+", "+str(test_score[8])+"\n")
        fp.close()
        fp=open(str(filename) + "test_21_metrice_" + ".txt","a")
        fp.write(str(test_21_score[5])+", "+str(test_21_score[6])+", "+str(test_21_score[7])+", "+str(test_21_score[8])+"\n")
        fp.close()
