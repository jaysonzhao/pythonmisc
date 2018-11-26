#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

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
    #价格经理    采购经理（音膜）  价格组经理/采购经理审批   价格总监(音膜)   财务审核  财务审核（零星）
    result=clf.predict([[7,57600000,2,9020,16548.48],
                       [11,892685.14,1,2050,769556.15],
                       [7,263.95,1,3090,225.6],
                       [11, 4288.93, 1, 2050, 3697.35],
                        [7, 32916, 1, 2080, 28375.86],
                        [9, 7504889.99, 1, 2020,7286301.01]])
    print(result)

    label_map = {1: '起草', 2:'采购经理', 3: '采购总监',4: '财务总监', 5: '采购VP', 6: 'CEO审批',
                 7: 'CEO手工单', 8: '特控部备案', 9: '采购经理(音膜)', '采购经理（音膜）': 9, 10: '财务审核（零星）',
                 11: '采购总监(音膜)', 12: '采购经理（标准）', 13: '价格组经理/采购经理审批）',
                 14: '价格总监(音膜)', 15: '财务审核', 16: '价格总监', 17: '价格组经理/采购经理审批'}
    for temp in result:
        print(label_map.get(int(temp)))

    from sklearn.externals import joblib
    #导出训练模型模板
    treefile = './tree.csv'
    joblib.dump(tree, treefile)

