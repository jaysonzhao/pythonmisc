#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np


with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    arr = []
    for row in reader:
        try:
            arr.append(float(row['CS_MetBox_Copper_Temp']))
        except ValueError:
            continue
#定义x、y散点坐标
x = np.arange(1, len(arr)+1, 1)
y = np.array(arr)

#用3次多项式拟合
f1 = np.polyfit(x, y, 3)
p1 = np.poly1d(f1)
print(p1)

#也可使用yvals=np.polyval(f1, x)
yvals = p1(x)  #拟合y值

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()
plt.savefig('test.png')