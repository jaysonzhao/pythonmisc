import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import csv
filename = 'speedupv1.csv'
columnname = 'Caster_pool_level'
with open('jtdataset\\'+filename) as csvfile:
    reader = csv.DictReader(csvfile)
    counter=0

    arr = []
    for row in reader:
        try:
            arr.append(float(row[columnname]))
             # arr.append(float(row['Caster_pool_level']))
        except ValueError:
            continue
X = np.array(range(0, len(arr), 1)).reshape(-1, 1)
y = np.array(arr)
# y = np.array([300,500,0,-10,0,20,200,300,1000,800,4000,5000,10000,9000,22000]).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
rmses = []
degrees = np.arange(1, 10)
min_rmse, min_deg, score = 1e10, 0, 0

for deg in degrees:
    # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)

    # 多项式拟合
    poly_reg = LinearRegression()
    poly_reg.fit(x_train_poly, y_train)
    # print(poly_reg.coef_,poly_reg.intercept_) #系数及常数

    # 测试集比较
    x_test_poly = poly.fit_transform(x_test)
    y_test_pred = poly_reg.predict(x_test_poly)

    # mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
    poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmses.append(poly_rmse)
    # r2 范围[0，1]，R2越接近1拟合越好。
    r2score = r2_score(y_test, y_test_pred)

    # degree交叉验证
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg
        score = r2score
        print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse, r2score))


# 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
poly = PolynomialFeatures(degree=min_deg, include_bias=False)
x_poly = poly.fit_transform(X)

# 多项式拟合
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)

newy = poly_reg.predict(poly.fit_transform(X))

plt.figure() # 实例化作图变量
plt.title(filename+','+columnname) # 图像标题
plt.xlabel('x') # x轴文本
plt.ylabel('y') # y轴文本
plt.plot(X, newy, 'k.')
plt.show()