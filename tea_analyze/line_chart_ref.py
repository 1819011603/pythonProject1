import matplotlib.pyplot as plt
import pandas
import xlrd
import openpyxl
import numpy
t = openpyxl.load_workbook("test_all_reflect.xlsx", True).active

y = []
temp = [i for i in t.values]


x = numpy.array(temp[0][1:-1],dtype=numpy.float)

for i in range(1,2):

    p = numpy.array(temp[i][1:-1],dtype=numpy.float)

    plt.plot(x,p,ls='-.',lw=1,label=str(temp[i][-1]))

plt.legend()
plt.xlabel(u"微波波长")
plt.ylabel(u"光照强度")
plt.show()
