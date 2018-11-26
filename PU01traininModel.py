#coding:utf-8
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

if __name__ == '__main__':

    d=joblib.load('tree.pkl')
    result=d.predict([[7,57600000,2,9020,16548.48],
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

