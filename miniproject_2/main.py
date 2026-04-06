from functions import Mandelbrot
import timeit
import numpy as np
import pandas as pd

test_sizes = [1024, 2048, 4096, 8192]

for size in test_sizes:
    
    mandelbrot = Mandelbrot(width = size, height = size)

    # Miniproject 1 implementations tests:
    result_naive =      timeit.repeat(lambda: mandelbrot.naive(), number=1, repeat=10)
    results_vector =    timeit.repeat(lambda: mandelbrot.vectorized(), number=1, repeat=10)
    results_njit =      timeit.repeat(lambda: mandelbrot.njit(), number=1, repeat=10)

    # Miniproject 2 implementations tests:
    results_parallel =   timeit.repeat(lambda: mandelbrot.parallel(), number=1, repeat=10)
    results_dask =      timeit.repeat(lambda: mandelbrot.dask(), number=1, repeat=10)


results = [('result_naive', result_naive, str(size)),
           ('results_vector', results_vector, str(size)),
           ('results_njit', results_njit, str(size)),
           ('results_parallel', results_parallel, str(size)),
           ('results_dask',results_dask, str(size))]

# format to dataframe
df = pd.DataFrame(results, columns=['method', 'time'])

# unnest time column
df = df.explode('time')

# Save to CSV
df.to_csv('results.csv', index=False)
