
import numpy as np
xpoints = np.array(
            [919.8078, 923.907776, 927.984375, 932.037781, 936.068176, 940.075684, 944.060486, 948.022827, 951.96283,
             955.880676, 959.776611, 963.650696, 967.503235, 971.33429, 975.144104, 978.9328, 982.700623, 986.447754,
             990.174316, 993.880432, 997.566467, 1001.232483, 1004.878601, 1008.505066, 1012.112122, 1015.699829,
             1019.268433, 1022.817993, 1026.348755, 1029.861206, 1033.354858, 1036.830566, 1040.288208, 1043.727661,
             1047.14978, 1050.554077, 1053.94104, 1057.311157, 1060.66394, 1063.999878, 1067.319458, 1070.622314,
             1073.908936, 1077.179688, 1080.434204, 1083.673096, 1086.896362, 1090.104248, 1093.296875, 1096.474609,
             1099.637207, 1102.7854, 1105.918945, 1109.038086, 1112.143188, 1115.234253, 1118.311523, 1121.375122,
             1124.425293, 1127.462402, 1130.486206, 1133.496948, 1136.495239, 1139.480713, 1142.453979, 1145.415161,
             1148.364014, 1151.301147, 1154.226685, 1157.140503, 1160.043213, 1162.934692, 1165.815308, 1168.684937,
             1171.544189, 1174.392822, 1177.231323, 1180.059692, 1182.878052, 1185.686768, 1188.486084, 1191.275879,
             1194.056519, 1196.828125, 1199.590942, 1202.345093, 1205.09082, 1207.828247, 1210.557373, 1213.278687,
             1215.99231, 1218.69812, 1221.396606, 1224.087891, 1226.771973, 1229.449341, 1232.120117, 1234.784058,
             1237.441772, 1240.093384, 1242.738892, 1245.37854, 1248.012817, 1250.641357, 1253.264648, 1255.883057,
             1258.496216, 1261.104736, 1263.708862, 1266.30835, 1268.903687, 1271.494995, 1274.082397, 1276.66626,
             1279.24646, 1281.823364, 1284.397095, 1286.967773, 1289.535889, 1292.101196, 1294.664062, 1297.224731,
             1299.783325, 1302.339722, 1304.894775, 1307.447998, 1310, 1312.550659, 1315.100342, 1317.649292,
             1320.197388, 1322.745117, 1325.292358, 1327.839722, 1330.386841, 1332.934326, 1335.4823, 1338.030762,
             1340.579834, 1343.130005, 1345.681396, 1348.233643, 1350.787598, 1353.343262, 1355.900391, 1358.459961,
             1361.021606, 1363.585449, 1366.151611, 1368.720947, 1371.292725, 1373.867676, 1376.446045, 1379.02771,
             1381.612915, 1384.202026, 1386.794922, 1389.391968, 1391.993408, 1394.599243, 1397.209717, 1399.825195,
             1402.445312, 1405.070801, 1407.70166, 1410.338135, 1412.980103, 1415.628296, 1418.282227, 1420.942627,
             1423.609375, 1426.282837, 1428.962769, 1431.649902, 1434.344238, 1437.045532, 1439.754761, 1442.471558,
             1445.196045, 1447.928345, 1450.669312, 1453.418335, 1456.176025, 1458.942505, 1461.717773, 1464.502197,
             1467.295776, 1470.098877, 1472.911743, 1475.734253, 1478.56665, 1481.409302, 1484.262329, 1487.125732,
             1489.999878, 1492.88501, 1495.781006, 1498.688232, 1501.607056, 1504.537354, 1507.479248, 1510.433105,
             1513.39917, 1516.377441, 1519.368164, 1522.371704, 1525.387695, 1528.417114, 1531.459473, 1534.515137,
             1537.584229, 1540.667236, 1543.764038, 1546.874878, 1550, 1553.139648, 1556.293457, 1559.462402,
             1562.64624, 1565.845093, 1569.059326, 1572.289062, 1575.53418, 1578.795654, 1582.072632, 1585.365967,
             1588.675659, 1592.001953, 1595.344849, 1598.704834, 1602.081787, 1605.47583, 1608.887573, 1612.316772,
             1615.763672, 1619.228638, 1622.711914, 1626.213013, 1629.73291, 1633.271729, 1636.828979, 1640.405273,
             1644.000977, 1647.616089, 1651.25061, 1654.905029, 1658.578979, 1662.273193, 1665.988037, 1669.722778,
             1673.478271, 1677.254883, 1681.052368, 1684.870728, 1688.710693, 1692.572144], dtype=np.float)

def get_x_by_moisture(moisture, data_path):


    from PLS.utils import loadDataSet01
    x0, y0 = loadDataSet01(data_path, ', ')

    index = []

    for (i,y) in enumerate(y0):
        if y == moisture:
            index.append(i)

    return x0[index],y0[index]

def get_x_by_moisture_and_x_y(moisture, x0, y0):
    index = []
    for (i,y) in enumerate(y0):
        if y == moisture:
            index.append(i)
    return x0[index],y0[index]

import matplotlib.pyplot as plt
def line_chart(ypoints, humidity):
    global xpoints
    """
    ????????????:ls  '???' ?????????'??????' ????????????'???.' ????????????':' ?????????
    lw linewidth ??????
    color ??????

    """

    # plt.plot(xpoints[1:-1], ypoints, ls='-', lw=1, label="")
    plt.plot(xpoints, ypoints, ls='-', lw=1, label="")
    # plt.plot(xpoints, ypoints, ls='-', lw=1, label=str(humidity[0]))




def paintsMoisture(x0,y0,dir_path="./picture",filter="SG",picture_name="picture",suff = "jpg"):

    #  (???,???) = (6.4*200, 4.8*200)  ????????????
    fig = plt.figure(figsize=(6.4, 4.8), dpi=200)
    # print(fig)   # Figure(1280x960)  == (6.4 * 200, 4.8*200)
    print(len(y0))
    if len(y0) == 0:
        return None
    for i in range(len(y0)):
        line_chart(x0[i],y0[i])
    plt.legend()
    plt.xlabel(u"????????????")
    plt.ylabel(u"????????????")

    from PLS.utils import get_log_name
    picture_path=get_log_name(picture_name,suff=suff,dir_path=dir_path)
    f = picture_path.split(".")

    f[0] += "_{}({})".format(filter,int(y0[0][0]))
    picture_path = "{}.{}".format(f[0],f[1])
    #  ????????????  dpi = 500 ??????????????????
    plt.savefig(picture_path,dpi=600) # (3840,2880) == (6.4*600, 4.8*600)
    fig = plt.gcf()
    print(fig)
    plt.show()

from PLS.utils import filter
import pathlib
def getAllPictures(filter_name="SG"):
    data_path = "../PLS-master/data/train.txt"
    from PLS.utils import loadDataSet01
    x0, y0 = loadDataSet01(data_path, ', ')
    x0 = filter(x0)
    p = pathlib.Path("./picture/"+filter_name)
    if ~p.exists():
        p.mkdir(parents=True,exist_ok=True)
    for i in range(1,100):
        x,y = get_x_by_moisture_and_x_y(i, x0,y0)
        paintsMoisture(x,y,dir_path="./picture/"+filter_name,filter=filter_name,suff="jpg")
def getAllPictures1(filter_name="SG"):
    data_path = "../PLS-master/data/test.txt"
    from PLS.utils import loadDataSet01
    x0, y0 = loadDataSet01(data_path, ', ')

    p = pathlib.Path("./picture/{}/test".format(filter_name))
    if ~p.exists():
        # ???????????????
        p.mkdir(parents=True, exist_ok=True)
    x0 = filter(x0)
    for i in range(1,100):
        x,y = get_x_by_moisture_and_x_y(i, x0,y0)
        paintsMoisture(x,y,dir_path="./picture/{}/test".format(filter_name),filter=filter_name,suff="jpg")

def getAllPictrueByFilter(filter_name):
    getAllPictures1(filter_name)
    getAllPictures(filter_name)

if __name__ == '__main__':
    getAllPictrueByFilter("SD[1:256]")


