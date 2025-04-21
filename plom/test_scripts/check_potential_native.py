import os
import ctypes
import math
import numpy as np
from math import exp, sqrt
import time
import sys

n_iters = 100

# Load the dll/so
lib_name = "potential_native.dll" if os.name == "nt" else "potential_native.so"
#lib_path = f"../pre-compiled/{lib_name}"
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lib", lib_name)

try:
    potential_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Failed to load library {lib_name}: {e}")
    sys.exit(1)

# Define the C++ function's argument and return types
potential_lib.get_L.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # H
    ctypes.POINTER(ctypes.c_double),  # u
    ctypes.POINTER(ctypes.c_double),  # pot (output)
    ctypes.c_int,  # nu
    ctypes.c_int,  # N
]
potential_lib.get_L.restype = None

# Input matrices
# H = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
# u = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
H = np.random.rand(3, 300)
u = np.random.rand(3, 300)

nu, N = H.shape  # nu = 3, N = 300

# Flatten H and u
H_flat = H.flatten()
u_flat = u.flatten()

# Prepare output array
pot = np.zeros((nu, N), dtype=np.float64)

# Convert arrays to ctypes pointers
H_ptr = H_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
u_ptr = u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
pot_ptr = pot.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# Call the C++ function
s = time.time()
for i in range(max(n_iters, 1)):
    potential_lib.get_L(H_ptr, u_ptr, pot_ptr, nu, N)
cpp_time = time.time() - s

# Python function
def get_L(H, u):
    nu, N = H.shape
    s = (4 / (N * (2 + nu))) ** (1 / (nu + 4))
    shat = s / np.sqrt(s**2 + (N - 1) / N)
    scaled_H = H * shat / s

    dist_mat_list = [(scaled_H.T - x).T for x in u.T]

    norms_list = np.exp((-1 / (2 * shat**2)) * np.array(list(map(
        lambda x: np.linalg.norm(x, axis=0)**2, dist_mat_list))))

    q_list = np.array(list(map(np.sum, norms_list))) / N

    product = np.array(list(map(np.dot, dist_mat_list, norms_list)))

    dq_list = product / shat**2 / N
    pot = (dq_list / q_list[:, None]).transpose()

    return pot

s = time.time()
for i in range(max(n_iters, 1)):
    pot_py = get_L(H, u)
py_time = time.time() - s

# Compare results
if np.allclose(pot, pot_py):
    print(f"\nTest passed for Potential library (native C++): Results match\nC++ run time: {cpp_time:.4f}s\nPython run time: {py_time:.4f}s")
    sys.exit(0)
else:
    print("\nTest failed: Results do not match")
    print("\nC++ result (first 3x3):\n", pot[:3, :3])
    print("\nPython result (first 3x3):\n", pot_py[:3, :3])
    sys.exit(1)
