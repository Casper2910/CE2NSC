import numpy as np
import random
from math import sqrt
import time

def calc_distance_to_zero(x, y):
    distance = sqrt(x**2 + y**2)

    return distance

def monte_carlo(runs = 10_000_000):
    inside = 0
    for i in range(runs):

        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = calc_distance_to_zero(x, y)

        if distance <= 1:
            inside += 1

    estimate = (4 * inside) / (runs)
    
    return estimate

def monte_carlo_alt(runs = 10_000_000):
    outside = 0
    for i in range(runs):

        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = calc_distance_to_zero(x, y)

        if distance > 1:
            outside += 1

    estimate = (4 * (runs - outside)) / (runs)
    
    return estimate

start = time.time()
print(monte_carlo())
end = time.time()

print(end-start)

start = time.time()
print(monte_carlo_alt())
end = time.time()

print(end-start)