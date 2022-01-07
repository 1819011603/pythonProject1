
import numpy as np
import matplotlib.pyplot as plt
with open("./recode20.log","r") as f:
    lines = f.readlines()
t = []
for line in lines:
    if line.find("max_RR:") != -1:
        t.append(float(line[line.find(" "):line.find(",")]) * 100)

p = np.arange(255,10,-1)
print(len(t))
print(len(p))
print(p)
plt.figure(figsize=(12,9),dpi=100)

plt.plot(p,t,)
plt.legend()
plt.xlabel(u"主成分数")
plt.ylabel(u"决定系数R2")
plt.show()