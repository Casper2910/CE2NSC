import timeit
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.array as da

test_sizes = [1024, 2048, 4096, 8192]

results = []

class Mandelbrot:

    def __init__(self, xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5, width = 1024, height = 1024, max_iter = 100):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.x_values = np.linspace(self.xmin, self.xmax, self.width)
        self.y_values = np.linspace(self.ymin, self.ymax, self.height)
        self.array = np.zeros((self.height, self.width), dtype=int)
        self.max_iter = max_iter

    def dask_distributed(self, scheduler_address="tcp://172.30.54.140:8786"):
        # Connect to the distributed cluster
        client = Client(scheduler_address)

        # loading variables
        array = self.array
        x_values = self.x_values
        y_values = self.y_values
        max_iter = self.max_iter
        height = self.height
        width = self.width
        array = self.array

        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = x_values[None, :] + 1j * y_values[:, None]

        # > Initialize 𝑧0 = 0 (for all points)
        z = np.zeros_like(c)
        array = np.zeros(c.shape, dtype=int)

        # mask to keep track on updated indexes
        mask = np.ones(c.shape, dtype=bool)

        # Convert to dask arrays — chunks are distributed across cluster workers
        chunk = 'auto'
        z     = da.from_array(z,     chunks=chunk)
        c     = da.from_array(c,     chunks=chunk)
        array = da.from_array(array, chunks=chunk)
        mask  = da.from_array(mask,  chunks=chunk)

        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(max_iter):

            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z = da.where(mask, z**2 + c, z)

            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛
            escaped       = da.abs(z) > 2
            newly_escaped = escaped & mask
            array         = da.where(newly_escaped, n, array)
            mask          = mask & ~newly_escaped

        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array = da.where(mask, max_iter, array)

        # compute to trigger lazy operations across the cluster
        result = array.compute()

        client.close()
        return result

for size in test_sizes:
    
    mandelbrot = Mandelbrot(width = size, height = size)

    results_dask =      timeit.repeat(lambda: mandelbrot.dask_distributed(), number=1, repeat=10)
    
    results.append({'method': 'dask_distributed', 'size': size, 'time': results_dask})
    
# format to dataframe
df = pd.DataFrame(results)

# unnest time column
df = df.explode('time')

# Save to CSV
df.to_csv('results.csv', index=False)
