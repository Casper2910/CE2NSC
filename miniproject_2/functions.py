import numpy as np
import dask

def generate_array_shape(xmin, xmax, ymin, ymax, width, height):
    # 2. Create grid of complex numbers 𝑐 over this region
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)

    # empty array for unit storage
    return np.zeros((height, width), dtype=int)

def mandelbroot_dask(array, max_iter):
    height, width = array.shape

    
if __name__ == "__main__":
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100