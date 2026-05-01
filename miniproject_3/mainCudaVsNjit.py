from functions import Mandelbrot, plot_mandelbrot
import timeit
import numpy as np
import pandas as pd
import os

test_sizes = [1024, 2048, 4096, 8192, 8192*2]

max_iters = [100, 200, 300, 400, 500, 600]

for size in test_sizes:
    for max_iter in max_iters:

        results = []

        print(f'Grid size: {size}x{size}:\nMax iterations: {max_iter}')
        
        mandelbrot = Mandelbrot(width = size, height = size, max_iter=max_iter)
        
        # warmup for both functions:
        #njit_result = mandelbrot.njit()
        #cuda_result = mandelbrot.cuda_numba()
        
        #plot_mandelbrot(njit_result, size, max_iter, "njit")
        #plot_mandelbrot(cuda_result, size, max_iter, "cuda-numba")
        test_runs = [

        # Miniproject 3 implementations tests:
        #('njit',        timeit.repeat(lambda: mandelbrot.njit(), number=1, repeat=10)),
        ('cupy',        timeit.repeat(lambda: mandelbrot.cupy(), number=1, repeat=10)),
        #('cuda-numba',        timeit.repeat(lambda: mandelbrot.cuda_numba(), number=1, repeat=10))
        ]


        # save results
        print('Savig results in df:\n')
        for name, time in test_runs: 
            # results.append({'method': name, 'size': size, 'time': time})
            
            output_file = 'cupy.csv'

            print(name, size, time, max_iter)

            # dataframe convertion
            df = pd.DataFrame({'name': name, 'size': size, 'time': time, 'max_iter': max_iter})
            
            # check if file already exists
            if os.path.exists(output_file):
                #unnest time column and save to CSV
                df.explode('time').to_csv(output_file, index=False, header=False, mode='a')

            else:
                # print headers for first row:
                df.explode('time').to_csv(output_file, index=False, header=True, mode='a')