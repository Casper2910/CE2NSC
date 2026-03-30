from functions import Mandelbrot
import timeit
import numpy as np

mandelbrot = Mandelbrot()

# Miniproject 1 implementations tests:
result_naive =      timeit.repeat(lambda: mandelbrot.naive(), number=1, repeat=10)
results_vector =    timeit.repeat(lambda: mandelbrot.vectorized(), number=1, repeat=10)
results_njit =      timeit.repeat(lambda: mandelbrot.njit(), number=1, repeat=10)

# Miniproject 2 implementations tests:
results_dask =      timeit.repeat(lambda: mandelbrot.dask(), number=1, repeat=10)
results_paralel =      timeit.repeat(lambda: mandelbrot.paralel(), number=1, repeat=10)