"""Test Numba JIT compilation"""
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def test_jit(x):
 return x ** 2

arr = np.arange(10.0)
result = test_jit(arr)
print(f"Numba JIT working: {result.sum}")
print(f"Result: {result}")
