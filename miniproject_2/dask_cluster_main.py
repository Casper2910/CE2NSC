import timeit
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.array as da
import distributed
from functions import Mandelbrot

test_sizes = [1024,
              2048,
              4096
            #,8192
]

results = []

client = Client("tcp://10.92.0.112:8786")
print(client)

for size in test_sizes:

    mandelbrot = Mandelbrot(width=size, height=size)

    results_dask = timeit.repeat(lambda: mandelbrot.dask_distributed(client), number=1, repeat=3) # only 3 repeats

    results.append({'method': 'dask_distributed', 'size': size, 'time': results_dask})
    
    # format to dataframe
    df = pd.DataFrame(results)

    # unnest time column
    df = df.explode('time')

    # Save to CSV
    df.to_csv(f'dask_distributed_results_{size}x{size}.csv', index=False)
    

client.close()

# format to dataframe
# df = pd.DataFrame(results)

# unnest time column
# df = df.explode('time')

# Save to CSV
# df.to_csv('dask_distributed_results.csv', index=False)