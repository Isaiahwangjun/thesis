from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
from neuralnetwork import SNDAE
import preprocess
from accuracyfunction import accuracy
import time
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import sklearn
# don't show warnings message
import warnings
warnings.filterwarnings("ignore")
#python ./main.py 5 100 0.02 0 1

tree_number = sys.argv[1]
learningrate = float(sys.argv[2])
method = int(sys.argv[3])

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

# set filename outputLayer testing iteration
#trainData = sys.argv[1]
#testData = sys.argv[2]

# load trainData
#train_temp = preprocess.loadDataset(trainData)
#train_noStringTemp_X = train_temp[0:,0:-1]

# start preprocess----------------------------------------------------------------------------------------------------------

# only trainData do this
#protocal_type_list = preprocess.saveProtocal_typeOrder(train_noStringTemp_X)
#service_list = preprocess.saveServiceOrder(train_noStringTemp_X)
#flag_list = preprocess.saveFlagOrder(train_noStringTemp_X)
#protocal_type_onehotencoded = preprocess.onehotencoded(protocal_type_list)
#service_list_onehotencoded = preprocess.onehotencoded(service_list)
#flag_list_onehotencoded = preprocess.onehotencoded(flag_list)

# load testData
#test_temp = preprocess.loadDataset(testData)
#test_noStringTemp_X = test_temp[0:,0:-1]

#train_noStringTemp_Y = []
#train_noStringTemp_Y = preprocess.distinguishNaturalAttack(train_temp, train_noStringTemp_Y, int(outputLayer))
#test_noStringTemp_Y = []
#test_noStringTemp_Y = preprocess.distinguishNaturalAttack(test_temp, test_noStringTemp_Y, int(outputLayer))

# np can't dynamic add number so use list
#train_noStringTemp_X = train_noStringTemp_X.tolist() 
#test_noStringTemp_X = test_noStringTemp_X.tolist()

# replace text with onehotencoded number
#preprocess.replace(train_noStringTemp_X, protocal_type_list, service_list, flag_list, protocal_type_onehotencoded, service_list_onehotencoded, flag_list_onehotencoded)
#preprocess.replace(test_noStringTemp_X, protocal_type_list, service_list, flag_list, protocal_type_onehotencoded, service_list_onehotencoded, flag_list_onehotencoded)

# list->np
train_noStringTemp_X = np.array(train_data)
train_noStringTemp_Y = np.array(train_label)
test_noStringTemp_X = np.array(test_data)
test_noStringTemp_Y = np.array(test_label)
test_21_noStringTemp_X = np.array(test_21_data)
test_21_noStringTemp_Y = np.array(test_21_label)
training_noStringTemp_X = np.array(train_data)
training_noStringTemp_Y = np.array(train_label)

# duration src_bytes
#for i in range(len(train_noStringTemp_X)):
#    train_noStringTemp_X[i][0] = preprocess.log(train_noStringTemp_X[i][0])
#    train_noStringTemp_X[i][4] = preprocess.log(train_noStringTemp_X[i][4])
#    train_noStringTemp_X[i][5] = preprocess.log(train_noStringTemp_X[i][5])

#for i in range(len(test_noStringTemp_X)):
#    test_noStringTemp_X[i][0] = preprocess.log(test_noStringTemp_X[i][0])
#    test_noStringTemp_X[i][4] = preprocess.log(test_noStringTemp_X[i][4])
#    test_noStringTemp_X[i][5] = preprocess.log(test_noStringTemp_X[i][5])

# normalize


train_resultNormalize = sklearn.preprocessing.normalize(train_noStringTemp_X)
test_resultNormalize = sklearn.preprocessing.normalize(test_noStringTemp_X)
test_21_resultNormalize =sklearn.preprocessing.normalize(test_21_noStringTemp_X)
training_resultNormalize = sklearn.preprocessing.normalize(training_noStringTemp_X)
#train_resultNormalize = preprocess.normalize(train_noStringTemp_X)
#test_resultNormalize = preprocess.normalize(test_noStringTemp_X)
#test_21_resultNormalize = preprocess.normalize(test_21_noStringTemp_X)
#training_resultNormalize = preprocess.normalize(training_noStringTemp_X)

# np->tensor
x_train_tensor = Variable(torch.from_numpy(train_resultNormalize)).float()
y_train_tensor = Variable(torch.from_numpy(train_noStringTemp_Y)).long()
x_test_tensor = Variable(torch.from_numpy(test_resultNormalize)).float()
y_test_tensor = Variable(torch.from_numpy(test_noStringTemp_Y)).long()
x_test_21_tensor = Variable(torch.from_numpy(test_21_resultNormalize)).float()
y_test_21_tensor = Variable(torch.from_numpy(test_21_noStringTemp_Y)).long()
x_training_tensor = Variable(torch.from_numpy(training_resultNormalize)).float()
y_training_tensor = Variable(torch.from_numpy(training_noStringTemp_Y)).long()

# end preprocess----------------------------------------------------------------------------------------------------------

# bulid stacked non-symmetric deep auto encoder model & random forest
sndae = SNDAE(np.size(train_resultNormalize,1))
random_forest = RandomForestClassifier(n_estimators=int(tree_number),n_jobs=-1)

# set optimizer and lossFunction
optimizer = optim.RMSprop(sndae.parameters(), lr=learningrate)
lossFunction = nn.CrossEntropyLoss()

# traning
sndae_train = sndae(x_train_tensor)
sndae_train = sndae_train.data.numpy() # tensor to numpy
random_forest = random_forest.fit(sndae_train, train_noStringTemp_Y)
 
# testing
#sndae_training = sndae(x_training_tensor)
#sndae_training = sndae_training.data.numpy()
#y_training_predic = random_forest.predict(sndae_training)
#accuracy_training, precision_training, recall_training = accuracy(training_noStringTemp_Y, y_training_predic)

sndae_test = sndae(x_test_tensor)
sndae_test = sndae_test.data.numpy()
y_test_predic = random_forest.predict(sndae_test)
y_test_proba = random_forest.predict_proba(sndae_test)
accuracy_test, precision_test, recall_test, f1_test = accuracy(test_noStringTemp_Y, y_test_predic, y_test_proba, au_name, 'test', train_label)

sndae_test_21 = sndae(x_test_21_tensor)
sndae_test_21 = sndae_test_21.data.numpy()
y_test_21_predic = random_forest.predict(sndae_test_21)
y_test_21_proba = random_forest.predict_proba(sndae_test_21)
accuracy_21, precision_21, recall_21, f1_21 = accuracy(test_21_noStringTemp_Y, y_test_21_predic, y_test_21_proba, au_name, 'test21', train_label)

with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_acc.txt', 'a') as file:
        file.write("%f\n" %accuracy_test)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_ps.txt', 'a') as file:
        file.write("%f\n" %precision_test)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_rc.txt', 'a') as file:
        file.write("%f\n" %recall_test)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_f1.txt', 'a') as file:
        file.write("%f\n" %f1_test)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_21_acc.txt', 'a') as file:   
        file.write("%f\n" %accuracy_21)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
        file.write("%f\n" %precision_21)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
        file.write("%f\n" %recall_21)
with open(r'/home/wang/Desktop/thesis/SNDAE/result/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
        file.write("%f\n" %f1_21)