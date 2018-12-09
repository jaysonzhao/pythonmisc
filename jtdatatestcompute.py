import csv
import pandas as pd
import numpy as np
from dfa import dfa
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist

def rollingDFA(narr):
    scales, fluct, alpha = dfa(narr)
    print("去趋势波动分析指数: {}", alpha)
    return alpha


with open('jtdataset\\speedupv1.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    counter=0

    arr = []
    for row in reader:
        try:
            # arr.append(float(row['Mill inlet temperature']))
             arr.append(float(row['Caster_pool_level']))
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
    arr_var = np.var(arr, ddof=1)
    # 求标准差
    arr_std = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)
    print("最大值为:", max(arr))
    print("最小值为:", min(arr))
    print("变化率方差:", np.var(result, ddof=1))
    print("变化率max:", np.max(result))
    logarr = np.log(arr)

    print("log Volatility", (np.std(logarr, ddof=1)/np.mean(logarr))/np.sqrt(1/len(arr)))
    c = pd.Series(arr).rolling(window=2400, center=False).var(ddof=1)
    # v = pd.Series(arr).rolling(window=2400, center=False).apply(rollingDFA, raw=True)
    # print('rolling dfa max: ', np.max(v))
    print('rolling var:', np.max(c))
    scales, fluct, alpha = dfa(arr)
    print("去趋势波动分析指数: {}", alpha)

    x = np.array(arr)
    # print(x)
    maxpointspos = argrelextrema(x, np.greater, order=2400)[0]
    print("极值点：", maxpointspos)
    # curverate = []
    # for i in maxpointspos:
    #     x = (i - (i - 1), arr[i] - arr[i - 1])
    #     y = (i + 1 - i, arr[i + 1] - arr[i])
    #     d = 1 - pdist([x, y], 'cosine')
    #     sin = np.sqrt(1 - d ** 2)
    #     dis = np.sqrt((-2) ** 2 + (arr[i - 1] - arr[i + 1]) ** 2)
    #     k = 2 * sin / dis
    #     curverate.append(k)
    # print("极值点圆周曲率 ", curverate)
