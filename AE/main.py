import numpy as np
import pandas as pd
import tensorflow as tf
import sys 
sys.path.append("/home/wang/Desktop/wangjun/wang/") 
import get_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
import sklearn
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE, ADASYN

learningrate = float(sys.argv[1])
epochs_num = int(sys.argv[2])
batch_size = int(sys.argv[3])
method = int(sys.argv[4])

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

train_label = np.array(train_label)

class AE(Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(122, )),
            layers.Dense(32, kernel_initializer = 'normal', activation = 'relu'),
            layers.Dense(5, kernel_initializer = 'normal', activation = 'relu')])

        self.decoder = tf.keras.Sequential([
            layers.Dense(32, kernel_initializer = 'normal', activation = 'relu'),
            layers.Dense(122, kernel_initializer = 'normal', activation = 'softmax')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AE()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningrate), loss='sparse_categorical_crossentropy')
history = autoencoder.fit(train_data, train_label, batch_size = batch_size, epochs = epochs_num)

result_1 = autoencoder.predict(test_data)
result_2 = autoencoder.predict(test_21_data)

result_1 = np.argmax(result_1, axis = 1)
result_2 = np.argmax(result_2, axis = 1)

test_acc = accuracy_score(test_label, result_1)
test_21_acc = accuracy_score(test_21_label, result_2)

test_ps = precision_score(test_label, result_1, average = 'macro')
test_21_ps = precision_score(test_21_label, result_2, average = 'macro')

test_rc = recall_score(test_label, result_1, average = 'macro')
test_21_rc = recall_score(test_21_label, result_2, average = 'macro')

test_f1 = f1_score(test_label, result_1, average = 'macro')
test_21_f1 = f1_score(test_21_label, result_2, average = 'macro')

with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_acc.txt', 'a') as file:
    file.write("%f\n" %test_acc)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_ps.txt', 'a') as file:
    file.write("%f\n" %test_ps)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_rc.txt', 'a') as file:
    file.write("%f\n" %test_rc)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_f1.txt', 'a') as file:
    file.write("%f\n" %test_f1)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_21_acc.txt', 'a') as file:
    file.write("%f\n" %test_21_acc)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_21_ps.txt', 'a') as file:
    file.write("%f\n" %test_21_ps)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_21_rc.txt', 'a') as file:
    file.write("%f\n" %test_21_rc)
with open(r'/home/wang/Desktop/thesis/AE/result/'+str(au_name)+'/test_21_f1.txt', 'a') as file:
    file.write("%f\n" %test_21_f1)
