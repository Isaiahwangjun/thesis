import scipy.stats as stats
import pandas as pd

data1 = pd.read_csv('/home/wang/Desktop/thesis/multi_layer/result/adasyn/test_acc.txt', header=None)
data2 = pd.read_csv('/home/wang/Desktop/thesis/LU_tensor/augmentation/result/en_15adasyntest_acc.txt', header=None)
statistic, pvalue = stats.ranksums(data1[0], data2[0], alternative='greater')
if pvalue <= 0.005:
    print("大")
else :
    statistic, pvalue = stats.ranksums(data1[0], data2[0], alternative='less')
    if pvalue <= 0.005:
        print("小")
    else :
        print("無")