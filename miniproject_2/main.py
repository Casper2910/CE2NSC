from functions import Mandelbrot
import timeit
import numpy as np
import pandas as pd

test_sizes = [1024, 2048, 4096, 8192]

results = []

for size in test_sizes:
    
    mandelbrot = Mandelbrot(width = size, height = size)

    # Miniproject 1 implementations tests:
    result_naive =      timeit.repeat(lambda: mandelbrot.naive(), number=1, repeat=10)
    results_vector =    timeit.repeat(lambda: mandelbrot.vectorized(), number=1, repeat=10)
    results_njit =      timeit.repeat(lambda: mandelbrot.njit(), number=1, repeat=10)

    # Miniproject 2 implementations tests:
    results_parallel =   timeit.repeat(lambda: mandelbrot.parallel(), number=1, repeat=10)
    results_dask =      timeit.repeat(lambda: mandelbrot.dask(), number=1, repeat=10)

    # save results
    for name, time in [('naive', result_naive), 
                        ('vector', results_vector), 
                        ('njit', results_njit), 
                        ('parallel', results_parallel), 
                        ('dask', results_dask)]:
            
            results.append({'method': name, 'size': size, 'time': time})


# format to dataframe
df = pd.DataFrame(results)

# unnest time column
df = df.explode('time')

# Save to CSV
df.to_csv('results.csv', index=False)
