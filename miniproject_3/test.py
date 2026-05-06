import numpy as np
import pytest
from functions import Mandelbrot  # change to your filename

# small grid for fast tests
@pytest.fixture
def small_mandelbrot():
    return Mandelbrot(width=50, height=50, max_iter=50)


def test_naive_shape(small_mandelbrot):
    result = small_mandelbrot.naive()
    assert result.shape == (50, 50)


def test_vectorized_matches_naive(small_mandelbrot):
    m1 = Mandelbrot(width=50, height=50, max_iter=50)
    m2 = Mandelbrot(width=50, height=50, max_iter=50)

    naive = m1.naive()
    vect = m2.vectorized()

    assert np.array_equal(naive, vect)


def test_parallel_matches_naive(small_mandelbrot):
    m1 = Mandelbrot(width=50, height=50, max_iter=50)
    m2 = Mandelbrot(width=50, height=50, max_iter=50)

    naive = m1.naive()
    parallel = m2.parallel(num_threads=4)

    assert np.array_equal(naive, parallel)


def test_njit_matches_naive(small_mandelbrot):
    m1 = Mandelbrot(width=50, height=50, max_iter=50)
    m2 = Mandelbrot(width=50, height=50, max_iter=50)

    naive = m1.naive()
    njit_result = m2.njit()

    assert np.array_equal(naive, njit_result)


def test_values_within_bounds(small_mandelbrot):
    result = small_mandelbrot.vectorized()

    assert result.min() >= 0
    assert result.max() <= small_mandelbrot.max_iter


def test_known_point_inside_set():
    m = Mandelbrot(width=10, height=10, max_iter=50)

    # center ~ (0,0) is inside Mandelbrot set
    result = m.naive()
    center_value = result[5, 5]

    assert center_value == m.max_iter


def test_known_point_outside_set():
    m = Mandelbrot(width=10, height=10, max_iter=50)

    result = m.naive()

    # corner ~ (-2, -1.5) escapes quickly
    corner_value = result[0, 0]

    assert corner_value < m.max_iter


# Optional (skip if GPU not available)
@pytest.mark.skip(reason="Requires CUDA GPU")
def test_cuda_matches_naive():
    m1 = Mandelbrot(width=50, height=50, max_iter=50)
    m2 = Mandelbrot(width=50, height=50, max_iter=50)

    naive = m1.naive()
    cuda_result = m2.cuda_numba()

    assert np.array_equal(naive, cuda_result)