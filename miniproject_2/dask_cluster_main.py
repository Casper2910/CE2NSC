import timeit
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.array as da
import distributed

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
        x_da = da.from_array(x_values, chunks=128)
        y_da = da.from_array(y_values, chunks=128)
        c = x_da[None, :] + 1j * y_da[:, None]

        # > Initialize z0 = 0 (for all points)
        z     = da.zeros_like(c)
        array = da.zeros(c.shape, dtype=int, chunks=c.chunks)

        # mask to keep track on updated indexes
        mask  = da.ones(c.shape, dtype=bool, chunks=c.chunks)

        # persist all arrays on the cluster upfront to avoid sending large graphs
        c, z, array, mask = client.persist([c, z, array, mask])
        distributed.wait([c, z, array, mask])

        # > For n=0 to max_iter:
        for n in range(max_iter):

            # > Compute zn+1 = zn2 + c
            z = da.where(mask, z**2 + c, z)

            # > If zn+1 > 2: Point escapes! Store n
            escaped       = da.abs(z) > 2
            newly_escaped = escaped & mask
            array         = da.where(newly_escaped, n, array)
            mask          = mask & ~newly_escaped

            # persist every iteration to keep the task graph small
            z, array, mask = client.persist([z, array, mask])
            distributed.wait([z, array, mask])

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