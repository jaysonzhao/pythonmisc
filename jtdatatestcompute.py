import csv
import pandas as pd
import numpy as np
from dfa import dfa


with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    counter=0
    # df = pd.DataFrame(columns={'time','temp'})
    arr = []
    for row in reader:
        # s = dict()
        # s['time']=counter
        # s['temp']=row['CS_MetBox_Copper_Temp']
        # counter+=1
        # print(s)
        #  dftemp = pd.DataFrame([s], columns={'time','temp'})
        #  df = df.append(dftemp)
        try:
            arr.append(float(row['CS_MetBox_Copper_Temp']))
        except ValueError:
            continue
    print(arr)
    #计算变化率
    result = []
    it = iter(arr)
    n1  = float(it.__next__())
    for n2 in it:
        result.append((n2-n1)/n1)
        n1 = float(n2)
    # 求均值
    arr_mean = np.mean(arr)
    # 求方差
    arr_var = np.var(arr)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("最大值为:", max(arr))
    print("最小值为:", min(arr))
    print("变化率:", result)

    scales, fluct, alpha = dfa(arr)
    print("去趋势波动分析指数: {}".format(alpha))
    # print(df)
    # print('平均值 ',df.mean(column='temp'))