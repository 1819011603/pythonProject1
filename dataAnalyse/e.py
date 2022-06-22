import math

from sklearn import linear_model
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp
import numpy as np
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression




# 多项式
def polynomial(x1,y1,start = -15,end = 15, step = 0.01):
    # y = max_a * (x^max_coef)
    max_coef = [[0]]
    max_r2 = 0
    y_p = np.array([])



    lin_reg = [[0]]
    for a in np.arange(start, end, step, dtype=float):
        y = np.array(y1)
        x = np.array(x1)

        if a == 0:
            continue

        try:

            for i in range(len(x)):
                x[i][0] = math.pow(x[i][0], a)
            lin_reg = LinearRegression(fit_intercept=False)
            lin_reg.fit(x, y)
            y_pred = lin_reg.predict(x)
            y = np.array(y).reshape(-1, )
            R2 = sm.r2_score(y, y_pred)
            if R2 > max_r2:

                max_r2 = R2
                max_coef = lin_reg.coef_

                max_a = a
                y_p = lin_reg.predict(x)

            # ans = "y = {0:.2f} x ^ {1:.2f}".format(lin_reg.coef_[0][0], a)
            # print("前面系数为{0:.2f}, 多项式的值为{1:.2f}, R2得分为{2:.3f}, 预测的y为:{3}, 结果为: {4}".format(lin_reg.coef_[0][0], a,
            #                                                                               R2, y_p.ravel(),
            #                                                                               ans))
            # print('R2得分：', R2)

        except ValueError as e:

            continue
        except OverflowError as d:

            continue
    ans = "y = {0:.2f} x ^ {1:.2f}".format(max_coef[0][0], max_a)
    print("前面系数为{0:.2f}, 多项式的值为{1:.2f}, R2得分为{2:.3f}, 预测的y为:{3}, 结果为: {4}".format(max_coef[0][0], max_a, max_r2, y_p.ravel(),ans))
    return max_coef, max_a, max_r2


# 指数
def Exponential(x1,y1,start = -15,end = 15, step = 0.01):
    # y = max_a * (x^max_coef)
    max_coef = [[0]]
    max_r2 = 0
    y_p = np.array([])

    lin_reg = [[0]]
    for a in np.arange(start, end, step, dtype=float):
        y = np.array(y1)
        x = np.array(x1)

        if a == 0:
            continue

        try:

            for i in range(len(x)):
                x[i][0] = math.pow(math.e, a* x[i][0] )
            lin_reg = LinearRegression(fit_intercept=False)
            lin_reg.fit(x, y)
            y_pred = lin_reg.predict(x)
            y = np.array(y).reshape(-1, )
            R2 = sm.r2_score(y, y_pred)
            if  R2 > max_r2:
                max_r2 = R2
                max_coef = lin_reg.coef_
                max_a = a
                y_p = lin_reg.predict(x)
            # print('R2得分：', R2)

        except ValueError as e:

            continue
        except OverflowError as d:

            continue
    ans = "y = {0:.2f} e ^ {1:.2f}x".format(max_coef[0][0],max_a)
    print("前面系数为{0:.2f}, 指数的值为 {1:.2f}x, R2得分为{2:.3f}, 预测的y为:{3}, 结果为:{4}".format(max_coef[0][0], max_a, max_r2, y_p.ravel(),ans))
    return max_coef, max_a, max_r2

# 对数

def Logarithm(x1,y1):
    # y = max_a * (x^max_coef)

    max_coef = [[0]]
    max_r2 = 0
    y_p = np.array([])

    y = np.array(y1)
    x = np.array(x1)

    lin_reg = [[0]]
    try:

        for i in range(len(x)):
            x[i][0] = math.log(math.e,x[i][0])
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(x, y)
        y_pred = lin_reg.predict(x)
        y = np.array(y).reshape(-1, )
        R2 = sm.r2_score(y, y_pred)
        if  R2 > max_r2:
            max_r2 = R2
            max_coef = lin_reg.coef_

            y_p = lin_reg.predict(x)
        # print('R2得分：', R2)

    except ValueError as e:
        pass

    except OverflowError as d:
        pass

    ans = "y = {0:.2f} ln(x) {1:+.2f}".format(max_coef[0][0],lin_reg.intercept_[0])
    print("前面系数为{0:.2f}, 对数的截距为{1:.2f}, R2得分为{2:.2f}, 预测的y为:{3} 结果为: {4}".format(max_coef[0][0], lin_reg.intercept_[0] , max_r2, y_p.ravel(),ans))
    return max_coef, lin_reg, max_r2
if __name__ == '__main__':
    y = np.array([[23], [37], [100]], dtype=float)
    x = np.array([[10.96], [8.87], [5.28]], dtype=float)
    print("原始的x为: {0}".format(x.ravel()))
    print("原始的y为: {0}".format(y.ravel()))
    ans = []
    a = polynomial(x,y,start=-20,end=2,step=0.01)
    b = Exponential(x,y,start=-20,end=2,step=0.01)
    c = Logarithm(x,y)
    res = a
    ans.append(b)
    ans.append(c)
    j = 0
    k = 0
    for i in ans:
        k += 1

        if res[2] < i[2]:
            j = k
            res = i

    if k== 0:
        print("结果为多项式表示")
    elif k==1:
        print("结果为指数表示")
    else:
        print("结果为对数表示")

