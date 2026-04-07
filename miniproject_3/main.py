from functions import Mandelbrot
import timeit
import numpy as np
import pandas as pd

test_sizes = [1024, 2048, 4096, 8192
              ]

for size in test_sizes:

    results = []

    print(f'Testing grid size: {size}x{size}:\n')
    
    mandelbrot = Mandelbrot(width = size, height = size)

    test_runs = [

    # Miniproject 1 implementations tests:
    #('naive',        timeit.repeat(lambda: mandelbrot.naive(), number=1, repeat=10)),
    #('vectorized',      timeit.repeat(lambda: mandelbrot.vectorized(), number=1, repeat=10)),
    ('njit',        timeit.repeat(lambda: mandelbrot.njit(), number=1, repeat=10)),

    # Miniproject 2 implementations tests:
    ('parallel',    timeit.repeat(lambda: mandelbrot.parallel(), number=1, repeat=10)),
    #('dask',        timeit.repeat(lambda: mandelbrot.dask(), number=1, repeat=10)),

    # ('results_dask_distributed', timeit.repeat(mandelbrot.dask_distributed(), number=1, repeat = 10))

    ('cupy',        timeit.repeat(lambda: mandelbrot.cupy(), number=1, repeat=10))
    
    ]


    # save results
    print('Savig results in df:\n')
    for name, time in test_runs: 
        # results.append({'method': name, 'size': size, 'time': time})

        print(name, size, time)

        # dataframe convertion
        df = pd.DataFrame({'name': name, 'size': size, 'time': time})

        #unnest time column and save to CSV
        df.explode('time').to_csv('benchmarks.csv', index=False, mode='a')
