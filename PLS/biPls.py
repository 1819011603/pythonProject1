import numpy as np
from sklearn.model_selection import train_test_split

from utils import *


def get_iter(x0):
    global bb

    for i in range(len(bb)):
        f = list(bb[0:i])
        f.extend(bb[i + 1:])  # index
        ans = list(splits[f[0]])
        for v in f[1:]:
            ans.extend(splits[v])  # splits
        xx = []
        for m in x0:
            xx.append(m[ans])
        yield np.array(xx)  # x0


def getNext(j):
    global splits, bb
    bb.remove(j)
    if len(bb) == 0:
        return None
    ans = list(splits[bb[0]])
    for f in bb[1:]:
        ans.extend(splits[f])
    return ans

def getSplitsAndIndices(split_len=10):
    l = np.shape(x0)[1]
    len1 = int(np.floor(l / split_len))  # 剩余的尾巴不要了
    a = list(np.arange(l))
    b = list(np.arange(len1))
    splits = []
    u = 0
    for i in range(len1):
        splits.append(a[u:u + split_len])
        u += split_len
    # splits.append(a[u:])
    return splits, b
import pathlib
def testRR(index=None, file_path="./PLS-master/data/Sigmoid_Sigmoid_99.36.pkl"):
    if index is None:
        index = list(np.arange(256))
    x0, y0 = loadDataSet01('./PLS-master/data/test.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)
    x0 = getDataIndex(x0,index)
    net = Regression(len(index))
    pre = torch.load(file_path)

    net.load_state_dict(pre, strict=False)
    with torch.no_grad():
        y_predict = net(torch.as_tensor(torch.from_numpy(x0), dtype=torch.float64))

        # a = torch.round(torch.as_tensor(y_predict, dtype=torch.float64)) # 四舍五入
        a = torch.as_tensor(y_predict, dtype=torch.float64) #
        b = torch.tensor(y0, dtype=torch.float64)



        y_predict = torch.concat((a, b, a - b), dim=1)
        RR, RMSE = getRR_RMSE(a.detach().numpy(),y0)
        y_predict = pd.DataFrame(y_predict.detach().numpy())
        ff = get_log_name(pre="net", suff="csv", dir_path=str(pathlib.Path(file_path).parent))
        print("csv file save in {}".format(ff))
        y_predict.to_csv(ff)
        print("RR:", RR)
        print("RMSE:", RMSE)
import torch
from  RegressionNet import Regression
import torch.optim as optim
def regressionNet(x_train, x_test, y_train, y_test):
    global j,index
    net = Regression(len(index))
    print(net)
    y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
    x_train = torch.tensor(x_train, dtype=torch.float64)
    x_test = torch.tensor(x_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    row = len(y_test)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.937, 0.999))
    loss_func = torch.nn.MSELoss()

    for i in range(30000):
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
        # SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
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
        s = ["L"]
        for i, item in enumerate(net.l.named_children()):

            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
            else:
                s.append(str(item[1]).split("in_features=")[1].split(",")[0])
        print(round(RR * 100, 2), RMSE)
        torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s),round(RR*100,2)))
    return RR, RMSE

def regressionNet(x_train, x_test, y_train, y_test,x_val,y_val):
    global j,index
    net = Regression(len(index))
    print(net)
    y_mean = torch.tensor(mean(y_test, 0), dtype=torch.float64)
    y_val_mean = torch.tensor(mean(y_val,0),dtype=torch.float64)
    x_train = torch.tensor(x_train, dtype=torch.float64)
    x_test = torch.tensor(x_test, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    x_val = torch.tensor(x_val,dtype=torch.float64)
    y_val = torch.tensor(y_val,dtype=torch.float64)
    row1 = len(y_val)
    row = len(y_test)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.937, 0.999))
    loss_func = torch.nn.MSELoss()

    for i in range(30000):
        y_predict = net(x_train)
        loss = loss_func(y_predict, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(y_predict)

    with torch.no_grad():
        y_predict = net(x_test)
        y_val_ = net(x_val)
        # print(list(y_predict.detach().numpy() - y_test.detach().numpy()))
        SSE = sum(sum(power((y_test.detach().numpy() - y_predict.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_test.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = sum(sum(power((y_predict.detach().numpy() - y_mean.detach().numpy()), 2), 0))
        # SSR = SST-SSE
        RR = 1 - SSE / SST
        """
        RMSE实际上描述的是一种离散程度，不是绝对误差，其实就像是你射击，你开10枪，我也开10枪，你和我的总环数都是90环，你的子弹都落在离靶心差不多距离的地方,
        而我有几枪非常准，可是另外几枪又偏的有些离谱，这样的话我的RMSE就比你大，反映出你射击的稳定性高于我，但在仅以环数论胜负的射击比赛中，我们俩仍然是平手。
        这样说你应该能理解吧，基本可以理解为稳定性。那么对于你预报来说，在一个场中RMSE小说明你的模式对于所有区域的预报水平都相当，反之，RMSE较大，
        那么你的模式在不同区域的预报水平存在着较大的差异。

        """

        RMSE = sqrt(SSE / row)

        SSE = sum(sum(power((y_val.detach().numpy() - y_val_.detach().numpy()), 2), 0))
        SST = sum(sum(power((y_val.detach().numpy() - y_val_mean.detach().numpy()), 2), 0))
        RR1 = 1 - SSE / SST
        RMSE1 = sqrt(SSE / row1)
        j += 1
        s = ["L"]
        for i, item in enumerate(net.l.named_children()):

            if i % 2 == 1:
                s.append(str(item[1]).split('(')[0])
            else:
                s.append(str(item[1]).split("in_features=")[1].split(",")[0])
        print(round(RR * 100, 2), RMSE,round(RR1 * 100, 2), RMSE1)
        torch.save(net.state_dict(), "./{0}_{1}.pkl".format("_".join(s),round(RR*100,2)))
    return RR, RMSE,RR1,RMSE1
def main(f_test=PLS,s_len = 11,splitss = 10,random_state=11):
    import time
    start = time.time()
    global x0,bb,splits

    x0, y0 = loadDataSet01('./PLS-master/data/test_all_reflect1.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)


    splits, bb = getSplitsAndIndices(split_len=s_len)

    # print(getNext(1))
    # print(getNext(2))
    # print(getNext(3))
    # print(getNext(23))
    m = 0
    m_j = 0
    b_ = []

    rm = 10
    rm_j = 0
    mylog = open(get_log_name(), mode='a', encoding='utf-8')

    while len(bb) > ceil(11.0 / s_len):
        k = 0
        p = get_iter(x0)
        max = 0
        max_j = 0
        rmse = 10
        rmse_j = 0
        for x in p:
            x_trains, x_tests, y_trains, y_tests = split10items(x, y0, splitss=splitss, random_state=random_state)
            p = 0
            m1 = 0
            for i in range(len(x_trains)):
                RR, RMSE = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
                p += RR/len(x_trains)
                m1 += RMSE/len(x_trains)
            print("{} RR: {} RMSE: {}".format(k, p, m1), file=mylog)
            if max < p:
                max = p
                max_j = k
            if rmse > m1:
                rmse = m1
                rmse_j = k
            k += 1

        if m < max or abs(max-m) < 0.001:


            b_ = list(bb)
            if m < max:
                m_j = len(bb)
                m = max

        if rm>rmse:
            rm =rmse
            rm_j = len(bb)
        # print("max_RR: {}, delete group is {}".format(max,bb[max_j]), file=mylog)
        print("max_RR: {}, delete group is {}".format(max,bb[rmse_j]), file=mylog)
        # bb.remove(bb[max_j])
        bb.remove(bb[rmse_j])
        print(bb, file=mylog)

    print(file=mylog)
    print('the best groups: {}'.format(b_), file=mylog)
    print("R2_max:{}, b_len: {}".format(m, m_j), file=mylog)
    print("rmse_min: {}, b_len: {}".format(rm,rm_j),file=mylog)
    # print(bb,file=mylog)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)),file=mylog)
    mylog.close()



# index = [0, 2, 3, 4, 5, 7, 13, 17, 28, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 52, 55, 56, 57, 58, 61, 62, 64, 65, 66, 70, 72, 73, 74, 77, 78, 79, 81, 84, 91, 93, 96, 98, 99, 102, 108, 109, 111, 115, 120, 121, 128, 131, 136, 137, 138, 144, 145, 146, 147, 158, 159, 165, 166, 167, 172, 178, 179, 180, 183, 187, 193, 196, 204, 209, 210, 212, 217, 224, 227, 228, 230, 231, 240]
# index = [0, 3, 8, 13, 29, 30, 32, 33, 34, 35, 39, 40, 42, 47, 52, 55, 57, 60, 61, 64, 66, 69, 72, 74, 77, 79, 91, 92, 93, 95, 102, 107, 108, 109, 115, 120, 121, 128, 136, 137, 138, 145, 146, 158, 159, 166, 167, 170, 172, 178, 179, 180, 183, 187, 191, 193, 204, 209, 210, 212, 224, 227, 230, 231, 240, 241, 246, 252]
index = [0, 1, 2, 3, 4, 7, 8, 13, 15, 29, 30, 31, 32, 33, 34, 35, 39, 40, 42, 47, 52, 55, 56, 57, 58, 60, 61, 63, 64, 66, 69, 72, 74, 77, 78, 79, 91, 92, 93, 95, 102, 104, 106, 107, 108, 109, 110, 111, 115, 116, 118, 120, 121, 122, 128, 131, 136, 137, 138, 139, 141, 144, 145, 146, 157, 158, 159, 166, 167, 170, 172, 176, 178, 179, 180, 183, 184, 186, 187, 190, 191, 193, 196, 204, 208, 209, 210, 212, 222, 224, 227, 228, 230, 231, 234, 235, 240, 241, 246, 252]
def cross(f_test,random_state=11,splitss=10):
    global index
    # index =[0, 2, 3, 4, 5, 7, 8, 13, 17, 28, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 47, 52, 55, 56, 57, 58, 61, 62, 64, 65, 66, 70, 72, 73, 74, 77, 78, 79, 81, 84, 91, 93, 96, 98, 99, 102, 104, 106, 108, 109, 110, 111, 115, 118, 120, 121, 128, 131, 136, 137, 138, 141, 144, 145, 146, 147, 158, 159, 165, 166, 167, 172, 176, 178, 179, 180, 183, 187, 191, 193, 196, 204, 209, 210, 212, 217, 224, 227, 228, 230, 231, 234, 240]
    # index = list(range(0,256))
    x0, y0 = loadDataSet01('./PLS-master/data/train.txt', ', ')  # 单因变量与多因变量
    x0 = filter(x0)
    mm = getDataIndex(x0,index)
    x_trains, x_tests, y_trains, y_tests = split10items(mm, y0, splitss=splitss, random_state=random_state)
        # print(p)
    p = 0
    m = 0

    x1,y1 = loadDataSet01("./PLS-master/data/test.txt",", ")
    x1 = filter(x1)
    mm1 = getDataIndex(x1,index)
    p1 = 0
    m1 = 0

    for i in range(len(x_trains)):
        a, b,a1,b1 = f_test(x_trains[i], x_tests[i], y_trains[i], y_tests[i],mm1,y1)
        p += a
        m += b
        p1 += a1
        m1 += b1

    print(u"R2 {0}%".format(np.round(p / len(x_trains) * 100, 2)))
    print(u"RMSECV: {0}".format(m / len(x_trains)))
    print(u"r2 {0}%".format(np.round(p1 / len(x_trains) * 100, 2)))
    print(u"RMSEP: {0}".format(m1 / len(x_trains)))



if __name__ == '__main__':
    import time
    start = time.time()
    random_state=11
    # main(s_len=1)
    # testRR(index,"./PLS-master/data/L_100_Sigmoid_32_Sigmoid_11_99.44.pkl")


    cross(regressionNet,random_state=random_state)  # 30000epochs random = 11 spilits=10  R^2 98.23% RMSE. 1.5524621198000335
    # cross(PLS,random_state=random_state)
    end = time.time()
    print("the spent time is {} seconds".format((end - start)))