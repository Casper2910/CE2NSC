import time
import numpy as np

def non_unrolled(x, y):
    i = 0
    result = 0
    while i < len(x):
        result += (x[i]*y[i])
        i += 1
    return result, i

def unrolled(x, y):
    i = 0
    j = 0 #loops
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    while i < len(x):
        sum1 += (x[i]*y[i])
        sum2 += (x[i+1]*y[i+1])
        sum3 += (x[i+2]*y[i+2])
        sum4 += (x[i+3]*y[i+3])
        i += 4
        j += 1
    return sum1 + sum2 + sum3 + sum4, j

x = np.random.normal(size=int(1e6))
y = np.random.normal(size=int(1e6))

s = time.time()

result, loops = non_unrolled(x,y)

e = time.time()

print('non unrolled loops and time')
print(loops, result)
print(e-s)

s = time.time()

result, loops = unrolled(x,y)

e = time.time()

print('unrolled loops and time')
print(loops, result)
print(e-s)



