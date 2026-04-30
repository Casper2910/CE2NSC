import numpy as np
from numba import njit

class Mandelbrot:

    def __init__(self, xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5, width = 1024, height = 1024, max_iter = 100):
        """
    Class for generating Mandelbrot set images using different computational strategies.

    Parameters
    ----------
    xmin : float
        Minimum real value.
    xmax : float
        Maximum real value.
    ymin : float
        Minimum imaginary value.
    ymax : float
        Maximum imaginary value.
    width : int
        Number of points along the x-axis.
    height : int
        Number of points along the y-axis.
    max_iter : int
        Maximum number of iterations for divergence.
    """
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
    
    def naive(self):
        """
        Compute the Mandelbrot set using a naive double for-loop.

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        print('Naive:\n')
        # loading variables
        array = self.array
        x_values = self.x_values
        y_values = self.y_values
        max_iter = self.max_iter
        height = self.height
        width = self.width
        array = self.array
        
        for i in range(height):
            for j in range(width):
                
                # 3. For each point c in grid:
                # note to self: 1j is imaginary unit in python
                c = x_values[j] + 1j * y_values[i]
                
                # > Initialize 𝑧0 =0
                z = 0
                
                # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
                for n in range(max_iter):
                    
                    # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
                    z = z**2 + c
                    
                    # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛, break to next point
                    if abs(z) > 2:
                        array[i, j] = n
                        break
                else:
                    # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
                    array[i, j] = max_iter
        return array
    
    def vectorized(self):
        """
        Compute the Mandelbrot set using NumPy vectorized operations.

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        print('Vectorized:\n')
        # loading variables
        array = self.array
        x_values = self.x_values
        y_values = self.y_values
        max_iter = self.max_iter
        height = self.height
        width = self.width
        
        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = x_values[None, :] + 1j * y_values[:, None]
        
        # > Initialize 𝑧0 = 0 (for all points) (complex256 to prevent overflow of z)
        z = np.zeros_like(c)
        
        # mask to keep track on updated indexes
        mask = np.ones(c.shape, dtype=bool)
        
        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(max_iter):
            
            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z[mask] = z[mask]**2 + c[mask]
            
            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛 
            escaped = np.abs(z) > 2
            newly_escaped = escaped & mask
            array[newly_escaped] = n
            mask[newly_escaped] = False
            
        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array[mask] = max_iter

        return array
    
    # wrapper:
    def njit(self):
        """
        Compute the Mandelbrot set using a Numba JIT-compiled function.
        Function out of scope as njit can't read variables from object self

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        print('Njit:\n')
        return njit(
            self.array, self.x_values, self.y_values,
            self.max_iter, self.height, self.width
        )
    
    def parallel(self, num_threads=8):
        """
        Compute the Mandelbrot set using multithreading.

        Parameters
        ----------
        num_threads : int, optional
            Number of threads to use.

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        from threading import Thread
        print('Parallel:\n')
        # loading variables
        array = self.array
        x_values = self.x_values
        y_values = self.y_values
        max_iter = self.max_iter
        height = self.height
        width = self.width

        # Worker function processes a block of rows in a vectorized way
        def worker(start_row, end_row):
            # 3. For each point c in grid (for the block of rows):
            # note to self: 1j is imaginary unit in python
            c = x_values[None, :] + 1j * y_values[start_row:end_row, None]
            
            # > Initialize z0 = 0 for all points in this block
            z = np.zeros_like(c)
            
            # mask to track which points have not escaped
            mask = np.ones(c.shape, dtype=bool)
            
            # local array to store iteration counts for this block
            local = np.zeros(c.shape, dtype=int)

            # > For n = 0 to max_iter
            for n in range(max_iter):
                # > Compute zn+1 = zn^2 + c (only for points that haven't escaped)
                z[mask] = z[mask]**2 + c[mask]

                # > Check which points escaped (abs(z) > 2)
                escaped = np.abs(z) > 2
                newly_escaped = escaped & mask

                # > Store the iteration count for newly escaped points
                local[newly_escaped] = n
                
                # > Update mask so escaped points are ignored in next iterations
                mask[newly_escaped] = False

            # > If loop completes: Point is in set, store max_iter
            local[mask] = max_iter

            # > Write back the block's results to the main array
            array[start_row:end_row, :] = local

        # Split rows among threads
        threads = []
        rows_per_thread = height // num_threads

        for t in range(num_threads):
            start = t * rows_per_thread
            # last thread takes any remaining rows
            end = (t + 1) * rows_per_thread if t != num_threads - 1 else height

            # create and start thread
            thread = Thread(target=worker, args=(start, end))
            threads.append(thread)
            thread.start()

        # wait for all threads to finish
        for thread in threads:
            thread.join()

        # > Return the full Mandelbrot array
        return array

    def dask(self):
        """
        Compute the Mandelbrot set using Dask for parallel array processing.

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        import dask.array as da
        print('Dask:\n')
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
        
        # > Initialize 𝑧0 = 0 (for all points) (complex256 to prevent overflow of z)
        z = np.zeros_like(c)
        
        # mask to keep track on updated indexes
        mask = np.ones(c.shape, dtype=bool)

        # Convert to dask arrays for parallel chunk processing
        z = da.from_array(z, chunks='auto')
        c = da.from_array(c, chunks='auto')
        mask = da.from_array(mask, chunks='auto')
        
        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(max_iter):
            
            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z = da.where(mask, z**2 + c, z)
            
            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛 
            escaped = da.abs(z) > 2
            newly_escaped = escaped & mask
            array = da.where(newly_escaped, n, array)
            mask = da.where(newly_escaped, False, mask)
        
        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array = da.where(mask, max_iter, array)

        # compute to trigger lazy operations
        return array.compute()


    def dask_distributed(self, client):
        """
        Compute the Mandelbrot set using Dask distributed across a cluster.

        Parameters
        ----------
        client : dask.distributed.Client
            Dask client connected to a cluster.

        Returns
        -------
        np.ndarray
            2D array with iteration counts for each point.
        """
        import dask.array as da
        from dask.distributed import Client
        import distributed
        print('dask_distributed:\n')
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
    
    def cupy(self):
        """
        Compute the Mandelbrot set using CuPy on a GPU.

        Returns
        -------
        cupy.ndarray
            2D array with iteration counts for each point.
        """
        import cupy as cp
        print('Cupy:\n')
        # loading variables
        max_iter = self.max_iter
        height = self.height
        width = self.width
        
        # load as cp instead of np:
        x_values = cp.linspace(self.xmin, self.xmax, self.width)
        y_values = cp.linspace(self.ymin, self.ymax, self.height)
        array = cp.zeros((self.height, self.width), dtype=int)
        
        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = x_values[None, :] + 1j * y_values[:, None]
        
        # > Initialize 𝑧0 = 0 (for all points) (complex256 to prevent overflow of z)
        z = cp.zeros_like(c)
        
        # mask to keep track on updated indexes
        mask = cp.ones(c.shape, dtype=bool)
        
        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(max_iter):
            
            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z[mask] = z[mask]**2 + c[mask]
            
            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛 
            escaped = cp.abs(z) > 2
            newly_escaped = escaped & mask
            array[newly_escaped] = n
            mask[newly_escaped] = False
            
        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array[mask] = max_iter

        return array

# cant use self variables of class, standalone function:
@njit
def njit(array, x_values, y_values, max_iter, height, width):
    for i in range(height):
        for j in range(width):
                
                # 3. For each point c in grid:
                # note to self: 1j is imaginary unit in python
                c = x_values[j] + 1j * y_values[i]
                
                # > Initialize 𝑧0 =0
                z = 0
                
                # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
                for n in range(max_iter):
                    
                    # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
                    z = z**2 + c
                    
                    # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛, break to next point
                    if abs(z) > 2:
                        array[i, j] = n
                        break
                else:
                    # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
                    array[i, j] = max_iter
        return array
    
if __name__ == "__main__":
    m = Mandelbrot()
    