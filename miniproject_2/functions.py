import numpy as np
from numba import njit
import dask
import threading
import dask.array as da

class Mandelbrot:

    def __init__(self):
        self.xmin = -2,
        self.xmax = 1,
        self.ymin = -1.5,
        self.ymax = 1.5,
        self.width = 1024,
        self.height = 1024,
        self.x_values = np.linspace(self.xmin, self.xmax, self.width)
        self.y_values = np.linspace(self.ymin, self.ymax, self.height)
        self.array = np.zeros((self.height, self.width), dtype=int)
    
    def mandlebrot_naive(self):
        array = self.array
        for i in range(self.height):
            for j in range(self.width):
                
                # 3. For each point c in grid:
                # note to self: 1j is imaginary unit in python
                c = self.x_values[j] + 1j * self.y_values[i]
                
                # > Initialize 𝑧0 =0
                z = 0
                
                # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
                for n in range(self.max_iter):
                    
                    # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
                    z = z**2 + c
                    
                    # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛, break to next point
                    if abs(z) > 2:
                        array[i, j] = n
                        break
                else:
                    # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
                    array[i, j] = self.max_iter
        return array
    
    def mandlebrot_vectorized(self):
        
        array = self.array
        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = self.x_values[None, :] + 1j * self.y_values[:, None]
        
        # > Initialize 𝑧0 = 0 (for all points) (complex256 to prevent overflow of z)
        z = np.zeros_like(c)
        
        # mask to keep track on updated indexes
        mask = np.ones(c.shape, dtype=bool)
        
        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(self.max_iter):
            
            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z[mask] = z[mask]**2 + c[mask]
            
            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛 
            escaped = np.abs(z) > 2
            newly_escaped = escaped & mask
            array[newly_escaped] = n
            mask[newly_escaped] = False
            
        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array[mask] = self.max_iter

        return array
    
    @njit
    def mandlebrot_njit(self):
        array = self.array
        for i in range(self.height):
            for j in range(self.width):
                
                # 3. For each point c in grid:
                # note to self: 1j is imaginary unit in python
                c = self.x_values[j] + 1j * self.y_values[i]
                
                # > Initialize 𝑧0 =0
                z = 0
                
                # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
                for n in range(self.max_iter):
                    
                    # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
                    z = z**2 + c
                    
                    # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛, break to next point
                    if abs(z) > 2:
                        array[i, j] = n
                        break
                else:
                    # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
                    array[i, j] = self.max_iter
        return array
    
    def mandelbroot_paralel(self):
        print('not implemented')
        return None

    def mandelbroot_dask(self):
        
        array = self.array
        # 3. For each point c in grid (perform operation on each element in array):
        # note to self: 1j is imaginary unit in python
        c = self.x_values[None, :] + 1j * self.y_values[:, None]
        
        # > Initialize 𝑧0 = 0 (for all points) (complex256 to prevent overflow of z)
        z = np.zeros_like(c)
        
        # mask to keep track on updated indexes
        mask = np.ones(c.shape, dtype=bool)

        # Convert to dask arrays for parallel chunk processing
        z = da.from_array(z, chunks='auto')
        c = da.from_array(c, chunks='auto')
        mask = da.from_array(mask, chunks='auto')
        
        # > For 𝑛=0 to 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟:
        for n in range(self.max_iter):
            
            # > Compute 𝑧𝑛+1 =𝑧𝑛2 +𝑐
            z = da.where(mask, z**2 + c, z)
            
            # > If 𝑧𝑛+1 >2: Point escapes! Store 𝑛 
            escaped = da.abs(z) > 2
            newly_escaped = escaped & mask
            array = da.where(newly_escaped, n, array)
            mask = da.where(newly_escaped, False, mask)
        
        # > If loop completes: Point is in set, store 𝑚𝑎𝑥_𝑖𝑡𝑒𝑟
        array = da.where(mask, self.max_iter, array)

        # compute to trigger lazy operations
        return array.compute()

    
if __name__ == "__main__":
    m = Mandelbrot()