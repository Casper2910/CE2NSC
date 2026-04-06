import timeit
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.array as da

test_sizes = [1024, 2048, 4096, 8192]

results = []

class Mandelbrot:

    def __init__(self, xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=1024, height=1024, max_iter=100):
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

    def dask_distributed(self, client):
        # loading variables
        x_values = self.x_values
        y_values = self.y_values
        max_iter = self.max_iter

        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = x_values[None, :] + 1j * y_values[:, None]

        # > Initialize z0 = 0 (for all points)
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
        
        # reduce graph size for sending operations:
        # constantly sends
        c = client.persist(c)

        # > For n=0 to max_iter:
        for n in range(max_iter):

            # > Compute zn+1 = zn2 + c
            z = da.where(mask, z**2 + c, z)

            # > If zn+1 > 2: Point escapes! Store n
            escaped       = da.abs(z) > 2
            newly_escaped = escaped & mask
            array         = da.where(newly_escaped, n, array)
            mask          = mask & ~newly_escaped

        # > If loop completes: Point is in set, store max_iter
        array = da.where(mask, max_iter, array)

        # compute to trigger lazy operations across the cluster
        return array.compute()


client = Client("tcp://10.92.0.112:8786")
print(client)

for size in test_sizes:

    mandelbrot = Mandelbrot(width=size, height=size)

    # Warm up before timing
    mandelbrot.dask_distributed(client)

    results_dask = timeit.repeat(lambda: mandelbrot.dask_distributed(client), number=1, repeat=10)

    results.append({'method': 'dask_distributed', 'size': size, 'time': results_dask})

client.close()

# format to dataframe
df = pd.DataFrame(results)

# unnest time column
df = df.explode('time')

# Save to CSV
df.to_csv('results.csv', index=False)