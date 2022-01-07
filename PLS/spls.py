import numpy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from numpy import *
import scipy
from scipy import signal
# 数据读取-单因变量与多因变量
from sklearn.linear_model import LinearRegression


def loadDataSet01(filename, Separator='\t'):
    fr = open(filename)
    arrayLines = fr.readlines()
    assert len(arrayLines) != 0
    num = len(arrayLines[0].split(Separator)) - 1
    row = len(arrayLines)
    x = mat(zeros((row, num)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split(Separator)
        curLine = [float(i) for i in curLine]
        x[index, :] = curLine[0: -1]
        y[index, :] = curLine[-1]
        index += 1
    return np.array(x), np.array(y)


def data_Mean_Std(x0, y0):
    mean_x = mean(x0, 0)
    mean_y = mean(y0, 0)
    std_x = std(x0, axis=0, ddof=1)
    std_y = std(y0, axis=0, ddof=1)
    return mean_x, mean_y, std_x, std_y


# 数据标准化
from sklearn import preprocessing


def stardantDataSet(x0, y0):
    e0 = preprocessing.scale(x0)
    f0 = preprocessing.scale(y0)
    return e0, f0


def med_filtering(tea):
    ans = signal.medfilt(tea, 5)
    return ans


def wiener_filtering(tea):
    ans = signal.wiener(tea)
    # self.detrend(ans, humidity)
    return ans


def mean_filtering(tea):
    ans = np.convolve(tea, np.ones(5) / 5,mode="same")
    return ans


def gaussian_filtering(tea):
    ans = scipy.ndimage.filters.gaussian_filter(tea, sigma=0.85, mode="nearest")
    return ans


def MSC(data_x):  # 多元散射校正
    ## 计算平均光谱做为标准光谱
    mean = numpy.mean(data_x, axis=0)

    n, p = data_x.shape
    msc_x = numpy.ones((n, p))

    for i in range(n):
        y = data_x[i, :]
        lin = LinearRegression()
        lin.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = lin.coef_
        b = lin.intercept_
        msc_x[i, :] = (y - b) / k
    return msc_x


def savitzky_golay(ans):  # 实现曲线平滑

    return signal.savgol_filter(ans, 7, 3, mode="wrap")


def detrend(ans):
    ans = signal.detrend(ans)  # 去除非线性趋势 去除散射
    return ans


import pandas as pd


def D1(sdata):
    """
    一阶差分
    """
    temp1 = pd.DataFrame(sdata)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    return np.delete(temp3, 0, axis=1)


def D2(sdata):
    """
    二阶差分
    """
    temp2 = (pd.DataFrame(sdata)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


def filter(x0):
    # x0 = MSC(x0)
    # x0 = D1(x0)
    # x0 = preprocessing.scale(x0) # 标准化
    for x in range(len(x0)):
        x0[x] = gaussian_filtering(x0[x])
        x0[x] = savitzky_golay(x0[x])
        x0[x] = detrend(x0[x])
        x0[x] = med_filtering(x0[x])  # 提升0.2%

    return x0


def split10items(x0, y0, splits=10, random_state=1, extend=1):
    import random
    random.seed(random_state)
    len = np.shape(x0)[0]
    a = list(np.arange(0, len))
    random.shuffle(a)
    u = 0
    r = int(len / splits)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(splits - 1):
        x_test.append(x0[a[u:u + r]])
        y_test.append(y0[a[u:u + r]])

        train_x = list(a[u + r:])
        train_x.extend(a[0:u])

        if extend > 1:
            p = list(train_x)
            for i in range(1, extend):
                train_x.extend(p)
        x_train.append(x0[train_x])
        y_train.append(y0[train_x])
        u += r

    x_test.append(x0[a[u:]])
    y_test.append(y0[a[u:]])
    x_train.append(x0[a[0:u]])
    y_train.append(y0[a[0:u]])

    return x_train, x_test, y_train, y_test

def getRR_RMSE(y_test,y_predict):

    if isinstance(y_test,torch.Tensor):
        row = len(y_test)
        y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        RMSE = sqrt(SSE / row)
        return RR,RMSE
    y_mean = mean(y_test, 0)
    row = len(y_test)
    SSE = sum(sum(power((y_test - y_predict), 2), 0))
    SST = sum(sum(power((y_test - y_mean), 2), 0))
    SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    # print(SSE, SST, SSR)
    RR = 1 - (SSE / SST)
    RMSE = sqrt(SSE / row)
    return RR,RMSE


def beyesi(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import BayesianRidge
    svr = BayesianRidge(normalize=False)  # 贝叶斯回归
    svr.fit(x_train, y_train.ravel())

    y_predict = svr.predict(x_test)
    u = r2_score(y_test, y_predict)  # R2
    m = sqrt(mean_squared_error(y_test, y_predict))  # RMSE

    print(y_test - y_predict)
    RR,RMSE = getRR_RMSE(y_test,y_predict)
    # print(u"R2:", RR, u)
    # print(u"RMSE:", RMSE,m)
    # print(u"R^2", u)
    # print(u"RMSE.", m)
    return u, m


def RandomForestRegressor(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor
    svr = RandomForestRegressor()  # 贝叶斯回归
    svr.fit(x_train, y_train.ravel())

    y_predict = svr.predict(x_test)
    u = r2_score(y_test, y_predict)  # R2
    m = sqrt(mean_squared_error(y_test, y_predict))  # RMSE
    # print(u"R^2", u)
    # print(u"RMSE.", m)
    return u, m


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error


def test(x_train, x_test, y_train, y_test):
    # x0, y0 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')  # 单因变量与多因变量
    # x0 = filter(x0)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x0, y0, test_size=rate, random_state=11)
    # x_test,y_test = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')
    # x_test = filter(x_test)

    # from sklearn import feature_selection
    # # 筛选特征向量表现最好的前20%个特征，使用相同配置的决策树模型进行预测，并且评估性能
    # fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=10)
    #
    # # 注意此处的fit_transform参数，是两个值，不再只是X_train
    # X_train_fs = fs.fit_transform(x_train, y_train)
    # print(fs.scores_)
    max = 0
    j = 0
    r = 10000
    r_j = 0
    p = 0
    m = 0
    start = 11
    num = 1
    for i in range(start, start + num):
        pls2 = PLSRegression(n_components=i, max_iter=500, tol=1e-03, scale=True)
        pls2.fit(x_train, y_train)
        mean_y = mean(y_test, 0)
        row = shape(x_test)[0]
        y_mean = tile(mean_y, (row, 1))
        y_predict = pls2.predict(x_test)
        p = pls2.score(x_test, y_test)  # R2
        # u = r2_score(y_test, y_predict)  # R2
        m = sqrt(mean_squared_error(y_test, y_predict))  # RMSE
        # print(u"R^2",i, p)
        # print(u"RMSE.",m)

        #
        # for i in range(size(y_predict)):
        #     print(y_predict[i], y_test[i], y_predict[i] - y_test[i])
        # print(p)
        # SSE = sum(sum(power((y_test - y_predict), 2), 0))
        # SST = sum(sum(power((y_test- y_mean), 2), 0))
        # SSR = sum(sum(power((y_predict - y_mean), 2), 0))
        # RR = SSR / SST
        # RMSE = sqrt(SSE / row)
        # RR, RMSE = getRR_RMSE(y_test, y_predict)
        # print(u"R2:", RR,p,u)
        # print(u"RMSE:", RMSE,m)
        if p > max:
            max = p
            j = i
        # if RMSE<r:
        #     r = RMSE
        #     r_j = i
    # print(j,max)
    return p, m
    # print(r,r_j)

    # for i in range(size(y_predict)):
    #     print(y_predict[i], y_test[i], y_predict[i] - y_test[i])
    # plt.plot(y_predict)
    # plt.xticks(rotation=90)
    # plt.show()
    # print(numpy.concatenate((np.around(y_predict,1),y_test,np.around(y_predict-y_test,1)),axis=1))


def split10item(x0, y0, f_test, splits=10, random_state=11, extend=1):
    x0 = filter(x0)
    x_trains, x_tests, y_trains, y_tests = split10items(x0, y0, splits=splits, random_state=random_state, extend=extend)
    p = 0
    m = 0
    for i in range(len(x_trains)):
        a, b = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
        p += a
        m += b

    print(u"R^2 {0}%".format(np.round(p / len(x_trains) * 100, 2)))
    print(u"RMSE.", m / len(x_trains))


j = 0

import torch
import torch.optim as optim
from RegressionNet import Regression,LSTM
def regressionNet(x_train, x_test, y_train, y_test):

    net = Regression()
    global j

    y_mean = torch.tensor(mean(y_test, 0),dtype=torch.float64)
    x_train = torch.tensor(x_train,dtype=torch.float64)
    x_test = torch.tensor(x_test,dtype=torch.float64)
    y_train = torch.tensor(y_train,dtype=torch.float64)
    y_test = torch.tensor(y_test,dtype=torch.float64)

    row = len(y_test)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.937, 0.999))
    loss_func = torch.nn.MSELoss()

    for i in range(5000):
        y_predict = net(x_train)
        loss = loss_func(y_predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(y_predict)

    with torch.no_grad():
        y_predict = net(x_test)
        # print(type(y_predict))
        # print(list(y_predict.detach().numpy() - y_test.detach().numpy()))
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1- SSE / SST
        """
        RMSE实际上描述的是一种离散程度，不是绝对误差，其实就像是你射击，你开10枪，我也开10枪，你和我的总环数都是90环，你的子弹都落在离靶心差不多距离的地方,
        而我有几枪非常准，可是另外几枪又偏的有些离谱，这样的话我的RMSE就比你大，反映出你射击的稳定性高于我，但在仅以环数论胜负的射击比赛中，我们俩仍然是平手。
        这样说你应该能理解吧，基本可以理解为稳定性。那么对于你预报来说，在一个场中RMSE小说明你的模式对于所有区域的预报水平都相当，反之，RMSE较大，
        那么你的模式在不同区域的预报水平存在着较大的差异。
    
        """

        RMSE = sqrt(SSE / row)
        j += 1
        s = []
        for i, item in enumerate(net.l.named_children()):
            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
        print(round(RR*100,2), RMSE)
        # torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s),round(RR*100,2)))
    return RR, RMSE


def LSTM1(x_train, x_test, y_train, y_test):
    net = LSTM()
    global j
    print(net)
    y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
    x_train = torch.tensor(x_train, dtype=torch.float64)
    x_test = torch.tensor(x_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    row = len(y_test)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.937, 0.999))
    loss_func = torch.nn.MSELoss()

    for i in range(5000):
        y_predict = net(x_train)
        loss = loss_func(y_predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(y_predict)

    with torch.no_grad():
        y_predict = net(x_test)
        # print(list(y_predict.detach().numpy() - y_test.detach().numpy()))
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        """
        RMSE实际上描述的是一种离散程度，不是绝对误差，其实就像是你射击，你开10枪，我也开10枪，你和我的总环数都是90环，你的子弹都落在离靶心差不多距离的地方,
        而我有几枪非常准，可是另外几枪又偏的有些离谱，这样的话我的RMSE就比你大，反映出你射击的稳定性高于我，但在仅以环数论胜负的射击比赛中，我们俩仍然是平手。
        这样说你应该能理解吧，基本可以理解为稳定性。那么对于你预报来说，在一个场中RMSE小说明你的模式对于所有区域的预报水平都相当，反之，RMSE较大，
        那么你的模式在不同区域的预报水平存在着较大的差异。

        """

        RMSE = sqrt(SSE / row)
        j += 1
        s = []
        for i, item in enumerate(net.l.named_children()):
            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
        print(round(RR * 100, 2), RMSE)
        # torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s),round(RR*100,2)))
    return RR, RMSE

def testRR(x0,y0,file_path= "./PLS-master/data/Sigmoid_Sigmoid_99.36.pkl"):
    x0 = filter(x0)
    net = Regression()
    pre = torch.load(file_path)

    net.load_state_dict(pre, strict=False)
    with torch.no_grad():
        y_predict = net(torch.as_tensor(torch.from_numpy(x0), dtype=torch.float64))

        a = torch.round(torch.as_tensor(y_predict, dtype=torch.float64))
        b = torch.tensor(y0, dtype=torch.float64)



        y_predict = torch.concat((a, b, a - b), dim=1)
        RR, RMSE = getRR_RMSE(a.detach().numpy(),y0)
        y_predict = pd.DataFrame(y_predict.detach().numpy())
        y_predict.to_csv("./net3.csv")
        print("RR:", RR)
        print("RMSE:", RMSE)



x0, y0 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')  # 单因变量与多因变量
# import openpyxl
# import numpy
# t = openpyxl.load_workbook("test_all_reflect.xlsx", True).active
#
# y = []
# temp = [i for i in t.values]
#
#
# x = numpy.array(temp[0][1:-1],dtype=numpy.float)
# plt.figure(figsize=(12,9),dpi=130)
# plt.plot(x,x0[0],label=str("原始曲线"),ls='-',color="red")
# #
# x1 = filter(x0)
# plt.plot(x,x1[0],label=str("SG平滑+去趋势+中值滤波"),ls='-',color="green")

# x1 = wiener_filtering(x0[0])
# plt.plot(x,x1,label=str("wiener"),ls='-')
#
# x1 = mean_filtering(x0[0])
# plt.plot(x,x1,label=str("滑动平均法"),ls='-')
# x1 = MSC(x0)
#
# plt.plot(x,x1[0],label=str("多元反射矫正"),ls='-.')
# x1 = savitzky_golay(x0)
# plt.plot(x,x1[0],label=str("SG平滑"),ls='-',color= 'green')
# x1 = D1(x0)
# print(np.shape(x1))
# plt.plot(x[:-1],x1[0],label=str("一阶导数差值"),ls='-.')
# x1 = D2(x0)
# plt.plot(x[:-2],x1[0],label=str("二阶导数差值"),ls='-.')
#
# x1 = gaussian_filtering(x0)
# plt.plot(x,x1[0],label=str("高斯平滑"),ls='-.')
#
# x1 = detrend(x0)
# plt.plot(x,x1[0],label=str("去趋势"),ls='-',color="black")
# plt.legend()
# plt.xlabel(u"反射率")
# plt.ylabel(u"光谱波长")
# plt.show()



# testRR(x0,y0)

# split10item(x0, y0, LSTM1, splits=10, random_state=45, extend=1)
split10item(x0, y0, regressionNet, splits=10, random_state=5, extend=1)
# split10item(x0, y0, test, splits=10, random_state=11, extend=1)
# split10item(x0, y0, beyesi, splits=5, random_state=5, extend=1)
# split10item(x0,y0,RandomForestRegressor,splits=10,random_state=5,extend=1)


# x1, y1 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')
# x1 = filter(x1)
# mean_x, mean_y, std_x, std_y = data_Mean_Std(x1, y1)
# # x1,y1 = stardantDataSet(x1,y1)
# y_pred = pls2.predict(x1)
#
# for i in range( size(y_pred)):
#     print(y_pred[i],y1[i], y_pred[i]-y1[i])
