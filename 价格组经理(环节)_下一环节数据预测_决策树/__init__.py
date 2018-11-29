import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

import tensorflow as tf
from datetime import datetime

from sklearn.metrics import precision_score, recall_score
import cx_Oracle as oracle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

# 价格组经理/采购经理审批 网关环节的下一处理环节


label_map = {'起草': 1, '采购经理': 2, '采购总监': 3, '财务总监': 4, '采购VP': 5, 'CEO审批': 6,
             'CEO手工单': 7, '特控部备案': 8,'采购经理(音膜)' :9,'采购经理（音膜）' :9,'财务审核（零星）' :10,
            '采购总监(音膜)':11,'采购经理（标准）':12,'价格组经理/采购经理审批':13,
              '价格总监(音膜)':14,'价格总监（音膜）':14, '财务审核':15 , '价格总监':16 , '价格组经理/采购经理审批':13,'采购vp':18 ,'财务总监（标准）':19,
			  '采购总监(配件)':20,'采购VP（音膜）':21,
			  'CEO审批（标准）':22 ,'采购总监（标准）':23,'采购VP（标准）':24,'CEO手工单(标准采购订单)':25,'采购VP(配件)':26,'车间生产':27,'采购总监（音膜）':28,
			  'CEO审批(标准采购订单)':29,
			  '采购VP(音膜)':30,'车间生产（音膜）':31,'CEO手工单（标准）':32}



label_map1 = {'null': 1, '月结30天': 2, '月结45天': 3, '月结60天': 4, '月结90天': 5, '月结180天': 6,
             '入帐后30天': 7, '入帐后45天': 8, '入帐后60天':9, '入帐后75天': 10, '入帐后90天': 11, '入账后120天': 12,
             '货、票到7天付款': 13, '货、票到15天付款': 14, '货、票到30天付款': 15, '货、票到45天付款': 16,
              '(停用)月结30天': 17, '货到付款': 18, '90天远期L/C': 19, '预付': 20, 'Net 30 Days To HK YEC': 21}

label_map2 ={'国内': 1, '国外': 2}




def oracle_data():
    conn = oracle.connect('csmart/csmart@192.168.1.69:1521/smartformsdb')
    cursor = conn.cursor()
    sql = "select  f.text1 , f.document_id  ,f.zje,f.currentsapfactory,f.sapclientcode  , f.zjj_local2 " \
          "   from FORM_PU01 f  , PU01_EXPORT_TABLE k where  " \
          " f.document_id  =k.document_id " \
          " and f.currentsapfactory is not null and f.text1 is not null and f.zje is not null" \
          " and f.sapclientcode is not null and f.zjj_local2 is not null  " \
          " and k.task_name='价格组经理/采购经理审批'  "

    cursor.execute(sql)
    cursorV = conn.cursor()
    x_data = np.zeros(5)
    y_row = np.zeros(1)  # y目前只存放环节名称
    y_data = np.zeros(1)  # zeros(1)创建长度1的全0数组(这个是一维的只有一行)
    x_row = np.zeros(5)  # x目前只存放金额
    # 堆叠数组：stack()，hstack()，vstack()
    for result in cursor:  # 循环从游标获取每一行并输出该行。

        # 此处可优化，合并同一sql，后期可优化
        sql = "select  task_name,record_id  from  ( " \
              "select aa.* ,  RANK() OVER(PARTITION BY aa.document_id  ORDER BY  aa.create_time ) a  from (" \
              "select tt.*, RANK() OVER(PARTITION BY tt.document_id,  tt.SRC_NODE_ID" \
              " ORDER BY  tt.create_time ) sort2 from (" \
              "select task_name, SRC_NODE_ID, document_id ,record_id, create_time ," \
              "RANK() OVER(PARTITION BY document_id ORDER BY  create_time asc) sort " \
              "from PU01_EXPORT_TABLE where EXCHANGE_TYPE='submit' and document_id='"+result[1]+"'" \
              "order by document_id , create_time  asc) tt  order by document_id ,create_time asc ) aa " \
              " where sort2 = 1) where a=3"
        # a 为第几个环节 由于“价格组经理/采购经理审批” 为第二环节，下一环节为3

        cursorV.execute(sql)

        # for resultV in cursorV:
        # print(cursorV.fetchone()[0])
        # 目标只有一条，只取第一条
        all = cursorV.fetchall()
        if len(all) > 0:
            # 当经过第二个环节得文档才计入集合中
            x_row[0] = label_map1.get(result[0])
            x_row[1] = result[2]
            x_row[2] = label_map2.get(result[3])
            x_row[3] = result[4]
            x_row[4] = result[5]
            x_data = np.row_stack((x_data, x_row))  # 两个数组相加：加行

            y_row[0] = label_map.get(all[0][0])
            # y_row = hash(a[1])
            y_data = np.append(y_data, y_row)  # 一个数组扩展：加列

    x_data = x_data[1:len(x_data)]
    y_data = y_data[1:len(y_data)]
    # 关闭游标、oracle连接
    cursor.close()
    cursorV.close()
    conn.close()


    np.savetxt('x_data.csv', x_data, delimiter=',')
    np.savetxt('y_data.csv', y_data, delimiter=',')


    return x_data, y_data


if __name__ == '__main__':

    trainData, trainLabel = oracle_data()
    clf = tree.DecisionTreeClassifier()
    tree = clf.fit(trainData, trainLabel)
    #以下数据为数据中典型的数据，以此数据为预测数据
    #采购总监(配件)   特控部备案   采购总监
    result=clf.predict([[7,37800,1,5010,32307.69],
                       [11,9420,1,2050,8051.28],
                       [11,60529.51,1,2080,52180.61]
                    ])

    label_map = {1: '起草', 2: '采购经理', 3: '采购总监', 4: '财务总监', 5: '采购VP', 6: 'CEO审批', 7: 'CEO手工单', 8: '特控部备案',
                 9: '采购经理(音膜)', 10: '财务审核（零星）', 11: '采购总监(音膜)', 12: '采购经理（标准）', 13: '价格组经理/采购经理审批', 14: '价格总监(音膜)',
                 15: '财务审核', 16: '价格总监', 17: '', 18: '采购vp', 19: '财务总监（标准）', 20: '采购总监(配件)', 21: '采购VP（音膜）',
                 22: 'CEO审批（标准）', 23: '采购总监（标准）', 24: '采购VP（标准）', 25: 'CEO手工单(标准采购订单)', 26: '采购VP(配件)', 27: '车间生产',
                 28: '采购总监（音膜）', 29: 'CEO审批(标准采购订单)', 30: '采购VP(音膜)', 31: '车间生产（音膜）', 32: 'CEO手工单（标准）'}

    for temp in result:
        print(label_map.get(int(temp)))



    #输出结果符合预测预期
    #采购总监(配件)
    #特控部备案
    #采购总监

    from sklearn.externals import joblib

    # 导出训练模型模板
    treefile = '价格组经理.pkl'
    joblib.dump(tree, treefile)

