import time
import numpy as np

x = np.random.normal(size=int(1e6))
y = np.random.normal(size=int(1e6))


start = time.time()
print(x*y)
end = time.time()
print(end-start)


start = time.time()
print(x/y)
end = time.time()
print(end-start)


start = time.time()
print(x+y)
end = time.time()
print(end-start)


start = time.time()
print(x-y)
end = time.time()
print(end-start)

k = 3

start = time.time()
print(x/k)
end = time.time()
print(end-start)


start = time.time()
print(x*y)
end = time.time()
print(end-start)