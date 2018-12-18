import csv
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model  # GARCH(1,1)
from matplotlib import pyplot as plt
from datetime import timedelta
from dfa import dfa
from Hurst import *


# 计算得分value 为实际值 ，expect 是期望值， direction 是方向，也可用于权值，负数为越小越好，正数为越大越好.输入可以是NP ARRAY
def calpoint(value, expect, direction):
    return (value - expect) * direction / expect


columnnames = ['metbox_copperTemp', 'caster_pool_level', 'sr_position', 'Bartemp', 'caster_lineSpeed', 'millinletTemp']
columnnamestomark = ['caster_pool_level', 'caster_lineSpeed', 'millinletTemp', 'sr_position']
markexpect = dict()
markdir = dict()
#'DFA' 'violmax','rollingvarmax','std'
markexpect['millinletTemp']=np.array([1.1, 11, 15, 2.2])
markdir['millinletTemp'] = np.array([-0.3, -0.25, -0.25, -0.2])
markexpect['caster_lineSpeed'] = np.array([0.9, 1, 1, 0.002])
markdir['caster_lineSpeed'] = np.array([-0.8, 0, 0, -0.2])
markexpect['caster_pool_level'] = np.array([1.4, 2.0, 1.0, 0.6])
markdir['caster_pool_level'] = np.array([-0.35, -0.2, -0.35, -0.1])
markexpect['sr_position'] = np.array([1.5, 0.5, 0.01, 0.1])
markdir['sr_position'] = np.array([-0.3, -0.2, -0.2, -0.3])

dataresultname = 'dataresult1213'
rollingwindow = 600
for columnname in columnnames:
    resultfilename = dataresultname + columnname
    output = open('jtdataset\\' + resultfilename + '.csv', 'w')
    output.write("filename,column,avr,var,std,skew,kern,logviola,rollingvarmax,DFA,rollingvarmean,violmax,violmean," + '\n')
    output.close()

for i in range(3648, 3942, 1):
    filename = str(i) + '.csv'
    for columnname in columnnames:
        with open('jtdataset\\13\\' + filename) as csvfile:
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

            skew = stats.skew(np.array(arr))
            # 峰度 概率密度在均值处峰值高低的特征
            kurtosis = stats.kurtosis(arr)

            # print("变化率方差:", np.var(result, ddof=1))
            # print("变化率max:", np.max(result))
            logarr = np.log(arr)

            logVolatility = np.std(logarr, ddof=1) / np.mean(logarr) / np.sqrt(1 / len(arr))
            c = pd.Series(arr).rolling(window=rollingwindow, center=False).var(ddof=1)
            # v = pd.Series(arr).rolling(window=2400, center=False).apply(rollingDFA, raw=True)
            # print('rolling dfa max: ', np.max(v))
            rollingVarmax = np.max(c)
            rollingVarmean = np.mean(c)
            scales, fluct, alpha = dfa(arr)
            # 计算波动率 from GARCH(1,1)
            am = arch_model(arr)
            res = am.fit()
            sqt_h = res.conditional_volatility
            maxsqt = np.max(sqt_h)
            meansqt = np.mean(sqt_h)
            # print("去趋势波动分析指数: {}", alpha)

            # 去除波动性
            # f = arr / sqt_h
            # 计算hurst指数,函数来自自定义library
            # inter = 300  # 滑动时间窗口
            # hurst = Hurst(f, T=inter, step=1, q=2, Smin=10, Smax=50, Sintr=1)
            # print(hurst)

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

            print("文件名,列名,平均值,方差, 标准差, 偏度,峰度, 对数振荡,滚动方差最大,DFA,滚动方差平均,最大波动率,平均波动率")
            print(filename, columnname, arr_mean, arr_var, arr_std, skew, kurtosis, logVolatility, rollingVarmax, alpha,
                  rollingVarmean, maxsqt, meansqt)

            mark = 10
            # if(alpha<1.2):
            #     mark = mark-2
            # if(rollingVarmax>5 and rollingVarmax<8):
            #     mark = mark+1
            # if(logVolatility<0.1):
            #     mark = mark-1
            # if(rollingVarmax>8):
            #     mark = mark+2
            # if(columnname=='Caster_Pool_Level' and maxsqt>1):
            #     mark = mark+2
            # if(maxsqt>5):
            #     mark = mark+2

            output = open('jtdataset\\' + dataresultname + columnname + '.csv', 'a')
            output.write(str(filename) + ',' + str(columnname) + ',' + str(arr_mean) + ',' + str(arr_var) + ','
                         + str(arr_std) + ',' + str(skew) + ',' + str(kurtosis) + ',' + str(logVolatility) + ',' + str(
                rollingVarmax) + ',' + str(alpha) + ',' + str(rollingVarmean) + ',' + str(maxsqt) + ',' + str(
                meansqt) + ',' + str(mark) + '\n')
            output.close()



for columnname in columnnamestomark:
    resultfilename = dataresultname + columnname
    with open('jtdataset\\'+resultfilename+'.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        output = open('jtdataset\\'+resultfilename+'Mark.csv', 'w')
        for row in reader:
            try:
                arr = np.array([float(row['DFA']), float(row['violmax']), float(row['rollingvarmax']), float(row['std'])])

                output.write(row['filename'] + ',' + row['column'] + ',' + str(
                    np.sum(calpoint(arr, markexpect[columnname], markdir[columnname]))) + '\n')

            except ValueError:
                continue
        output.close()
