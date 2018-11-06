# -*- coding: UTF-8 -*-

import numpy as np
import cx_Oracle
import os
import xmltodict
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.cElementTree as et

from mpl_toolkits.mplot3d import axes3d


os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

connection = cx_Oracle.connect("csmart", "csmart", "mail.gzsolartech.com:1521/smartformsdb")
cursor = connection.cursor()
cursor.execute("SELECT d.DOCUMENT_ID,d.DOCUMENT_DATA.getClobVal(),d.FORM_NAME,d.APP_ID FROM DAT_DOCUMENT d "
               "where d.form_name='testwf3' and  rownum  <4  ")


df = pd.DataFrame()

resultall = cursor.fetchall()
arr2 = ['DOCUMENT_ID', 'FORM_NAME', 'APP_ID']
arr3 = []

for res in resultall:
    arr1 = []
    for i in range(len(res)):

        if i <> 1:
            a = res[i]
            arr1.append(a);

    arr3.append(arr1);

df = pd.DataFrame(arr3, columns=arr2)
print(df)



cursor.execute(" select  count(*) , TO_CHAR(d.create_time, 'yyyy') as years"
               "  from DAT_DOCUMENT d group by TO_CHAR(d.create_time, 'yyyy') ")

countall = cursor.fetchall()

x=[]
y=[]

for res in countall:
    x.append(res[1])
    y.append(res[0])




def plot2D():

    plt.plot(x, y)  # 画连线图
    plt.scatter(x, y)  # 画散点图

    plt.xlabel(u'年份', fontproperties='SimHei', fontsize=18)
    plt.ylabel(u'文档数', fontproperties='SimHei', fontsize=18)
    plt.show()


if __name__ == '__main__':
    plot2D()

