import numpy as np
import random
from math import sqrt, pi

def calc_distance_to_zero(x, y):
    distance = sqrt(x**2 + y**2)

    return distance

def monte_carlo(runs = 1000000):
    inside = 0
    for i in range(runs):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = calc_distance_to_zero(x, y)

        if distance <= 1:
            inside += 1

        estimate = (4 * inside) / (i+1)

        # print(estimate)
    
    return estimate

monte_carlo()


