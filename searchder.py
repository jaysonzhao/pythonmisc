from scipy.optimize import leastsq
import numpy as np


class search(object):
    def __init__(self, filename):
        self.filename = filename

    def func(self, x, p):
        f = np.poly1d(p)
        return f(x)

    def residuals(self, p, x, y, reg):
        regularization = 0.1  # 正则化系数lambda
        ret = y - self.func(x, p)
        if reg == 1:
            ret = np.append(ret, np.sqrt(regularization) * p)
        return ret

    def LeastSquare(self, data, k=100, order=4, reg=1, show=1):  # k为求导窗口宽度,order为多项式阶数,reg为是否正则化
        l = self.len
        step = 2 * k + 1
        p = [1] * order
        for i in range(0, l, step):
            if i + step < l:
                y = data[i:i + step]
                x = np.arange(i, i + step)
            else:
                y = data[i:]
                x = np.arange(i, l)
            try:
                r = leastsq(self.residuals, p, args=(x, y, reg))
            except:
                print("Error - curve_fit failed")
            fun = np.poly1d(r[0])  # 返回拟合方程系数
            df_1 = np.poly1d.deriv(fun)  # 求得导函数
            df_2 = np.poly1d.deriv(df_1)
            df_3 = np.poly1d.deriv(df_2)
            df_value = df_1(x)
            df3_value = df_3(x)

    def gaussian(self, x, *param):
        fun = param[0] * np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.))) + param[1] * np.exp(
            -np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))
        return fun
