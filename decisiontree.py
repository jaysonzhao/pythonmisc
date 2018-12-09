#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import tree


def oracle_data():

    f = open('x_data.csv')
    df = pd.read_csv(f)
    x_data = np.array(df)

    f = open('y_data.csv')
    df = pd.read_csv(f)
    y_data_temp = np.array(df)
    y_data = np.zeros(1)
    for temp in y_data_temp:
        #y_data.append(int(temp[0]))
        y_data = np.append(y_data, temp[0])

    y_data = y_data[1:len(y_data)]

    # 数据整理
   # y_data = np.delete(y_data, [len(y_data)-1])
   #  print(x_data)
   #  print(y_data)

    #np.savetxt('C:\\Users\\Thinkpad\\Desktop\\x_data.csv', x_data, delimiter=',')
    #np.savetxt('C:\\Users\\Thinkpad\\Desktop\\y_data.csv', y_data, delimiter=',')
    return x_data, y_data

if __name__ == '__main__':

    trainData, trainLabel = oracle_data()
    clf = tree.DecisionTreeClassifier()
    tree = clf.fit(trainData, trainLabel)
    print(clf.predict([[7,57600000,2,9020,16548.48], [11,892685.14,1,2050,769556.15], [7,263.95,1,3090,225.6]]))