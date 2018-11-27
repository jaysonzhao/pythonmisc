from sklearn import svm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)



f = open('x_data.csv')
df = pd.read_csv(f)
x_data = np.array(df)

f = open('y_data.csv')
df = pd.read_csv(f)
#y_data = np.array(df)
y_data_temp = np.array(df)
y_data = np.zeros(1)
for temp in y_data_temp:
    # y_data.append(int(temp[0]))
    y_data = np.append(y_data, temp[0])

y_data = y_data[1:len(y_data)]

#y_data =str( np.loadtxt(open("y_data.csv", "rb"), delimiter=",", skiprows=0))

print("-------------1--------------")
print(x_data)
print(y_data)
clf = svm.SVR()
print("-------------0--------------")
clf.fit(x_data, y_data)
print("---------------------------")
res = clf.predict([[7, 57600000, 2, 9020, 16548.48],
                      [11, 892685.14, 1, 2050, 769556.15],
                      [7, 263.95, 1, 3090, 225.6],
                      [11, 4288.93, 1, 2050, 3697.35],
                      [7, 32916, 1, 2080, 28375.86],
                      [7, 7504889.99, 1, 2020, 7286301.01]])
print (res)

##预测结果如下，计算结果巨慢巨慢，且出现的结果非预想结果
##[2.1000353  3.21019461 3.21019461 3.20420949 3.21019461 3.21019461]

#X = [[0, 0], [2, 2]]
#y = [0.5, 2.5]
#clf = svm.SVR()
#clf.fit(X, y)
#res = clf.predict([[1, 1]])
#print res
#如果如下：
#1.5，1.5