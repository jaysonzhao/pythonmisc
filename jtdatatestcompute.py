import csv

import numpy as np
import pandas as pd
from scipy import stats

from dfa import dfa

filename = 'stablev2.csv'
columnnames = ['BarTemperatureMillEntry', 'Caster_pool_level', 'Mill inlet temperature']

for columnname in columnnames:
    with open('jtdataset\\' + filename) as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0

        arr = []
        for row in reader:
            try:
                # arr.append(float(row['Mill inlet temperature']))
                arr.append(float(row[columnname]))
            except ValueError:
                continue
        # print(arr)

        # 计算变化率
        # result = []
        # it = iter(arr)
        # n1  = float(it.__next__())
        # for n2 in it:
        #     result.append((n2-n1)/n1)
        #     n1 = float(n2)
        # 求均值
        arr_mean = np.mean(arr)
        # 求方差
        arr_var = np.var(arr, ddof=1)
        # 求标准差
        arr_std = np.std(arr, ddof=1)
        # 偏度 衡量随机分布的不均衡性，偏度 = 0，数值相对均匀的分布在两侧

        skew = stats.skew(arr)
        # 峰度 概率密度在均值处峰值高低的特征
        kurtosis = stats.kurtosis(arr)

        # print("变化率方差:", np.var(result, ddof=1))
        # print("变化率max:", np.max(result))
        logarr = np.log(arr)

        logVolatility = np.std(logarr, ddof=1) / np.mean(logarr) / np.sqrt(1 / len(arr))
        c = pd.Series(arr).rolling(window=2400, center=False).var(ddof=1)
        # v = pd.Series(arr).rolling(window=2400, center=False).apply(rollingDFA, raw=True)
        # print('rolling dfa max: ', np.max(v))
        rollingVar = np.max(c)
        scales, fluct, alpha = dfa(arr)
        # print("去趋势波动分析指数: {}", alpha)

        x = np.array(arr)
        # print(x)
        # maxpointspos = argrelextrema(x, np.greater, order=2400)[0]
        # print("极值点：", maxpointspos)
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
        print("文件名，列名， 平均值, 方差， 标准差， 偏度，峰度, 对数振荡, 滚动方差，DFA")
        print(filename, columnname, arr_mean, arr_var, arr_std, skew, kurtosis, logVolatility, rollingVar, alpha)
        output = open('jtdataset\\dataresult.csv', 'a')
        output.write(str(filename) + ',' + str(columnname) + ',' + str(arr_mean) + ',' + str(arr_var) + ','
                     + str(arr_std) + ',' + str(skew) + ',' + str(kurtosis) + ',' + str(logVolatility) + ',' + str(
            rollingVar) + ',' + str(alpha) + '\n')
        output.close()
