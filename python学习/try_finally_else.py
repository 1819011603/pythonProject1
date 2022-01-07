import math


def f():
    try:
        print("from try")
        u = 1.0/0
        return "from try"
    except:
        print("from except")
        return "from except"
    else:
        print("ppp")
        return  "ppp"
    finally:
        print("from finally")
        return "from finally"

print(f())
array = [1, 8, 15]
g = (x for x in array if array.count(x) > 0)

array = [2, 8, 22]
print(list(g))

a = 1270
c = 1111
b = 1270
print(a is b)

d = 1111
print(d is c)
print(id(a),id(b))

row = [""]
board = row * 3



A = [0,1,2,3,4]

print(id(A))
print(id(A[0:]))
print(id(A[1:]))
print(id(A[2:]))

print(id(A[1:2]))
B = A[0:]
B[0] = ["hhhh"]
print(A)
print(B)

print("A : {}".format(A))
import time
t0 = time.time()
time.sleep(1)
name = 'processing'
print(f'{name} done in {time.time() - t0:.2f} s')
print('{} done in {} s'.format(name,round(time.time() - t0,2)))