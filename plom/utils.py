# plom/utils.py

import os
import time
import ctypes
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import cdist
from contextlib import contextmanager

# Global backend reference
_potential_lib = None
_get_L_cpp_func = None

class Timer:
    """
    Context manager for timing execution blocks.
    
    Parameters
    ----------
    name : str
        Label for the timing block.
    user_verbosity : int
        The current verbosity setting of the model (e.g., self.verbose).
    message_level : int, default=1
        The verbosity level required to see this message.
    
    Usage:
        with Timer("PCA", verbose_level=1, current_level=1):
            pca.fit(X)
    """
    def __init__(self, name, user_verbosity=1, message_level=1):
        self.name = name
        self.user_verbosity = user_verbosity
        self.message_level = message_level
        self.start = 0

    def __enter__(self):
        self.start = time.time()
        if self.user_verbosity >= self.message_level:
            print(f"\n--- {self.name} ---")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start
        if self.user_verbosity >= self.message_level:
            print(f"   > {self.name} finished in {elapsed:.4f}s")

def load_backend(method="auto", verbose=False) -> bool:
    """
    Load the C++ library based on the requested method.
    method options: 'auto', 'cpp_eigen', 'cpp_native', 'python'
    """
    global _potential_lib, _get_L_cpp_func
    
    if method == "python":
        return False

    # Return true if already loaded
    if _potential_lib is not None:
        return True
    
    lib_ext = ".dll" if os.name == "nt" else ".so"
    base_path = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(base_path, "lib")
    
    # Determine search order
    if method == "auto":
        variants = ["potential_eigen", "potential_native"]
    elif method == "cpp_eigen":
        variants = ["potential_eigen"]
    elif method == "cpp_native":
        variants = ["potential_native"]
    else:
        # Fallback/Default
        variants = ["potential_eigen", "potential_native"]

    for variant in variants:
        lib_name = variant + lib_ext
        lib_path = os.path.join(lib_dir, lib_name)
        
        if os.path.exists(lib_path):
            try:
                _potential_lib = ctypes.CDLL(lib_path)
                _get_L_cpp_func = _potential_lib.get_L
                _get_L_cpp_func.argtypes = [
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
                ]
                _get_L_cpp_func.restype = None
                if verbose: 
                    print(f"PLoM C++ backend loaded: {lib_name}")
                return True
            except Exception as e:
                if verbose: 
                    print(f"Failed to load {lib_name}: {e}")
    
    if verbose and method != "auto":
        print(f"Warning: Requested PLoM backend '{method}' not found. Falling back to Python.")
    return False

def compute_potential(H: np.ndarray, u: np.ndarray, kde_bw_factor: float = 1.0, 
                      method: str = "auto") -> np.ndarray:
    """
    Computes the gradient of the potential (L).
    
    Dispatches to C++ or Python based on 'method'.
    
    Parameters
    ----------
    H : (nu, N) normalized data
    u : (nu, N) current state in Ito loop
    """
    nu, N = H.shape
    
    # Check if we should use C++
    use_cpp = (method != "python") and (_potential_lib is not None)

    if use_cpp:
        H_flat = np.ascontiguousarray(H, dtype=np.float64).flatten()
        u_flat = np.ascontiguousarray(u, dtype=np.float64).flatten()
        pot = np.zeros((nu, N), dtype=np.float64)
        
        _get_L_cpp_func(
            H_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nu, N
        )
        return pot

    # Optimized Python Implementation
    s = (4 / (N * (2 + nu))) ** (1 / (nu + 4)) * kde_bw_factor
    shat = s / np.sqrt(s**2 + (N - 1) / N)
    scaled_H = H * shat / s

    # # Efficient Distance Matrix Calculation (Vectorized)
    # u_sq = np.sum(u**2, axis=0)          
    # H_sq = np.sum(scaled_H**2, axis=0)   
    # interaction = np.dot(u.T, scaled_H)
    # sq_dists = u_sq[:, None] + H_sq[None, :] - 2 * interaction
    # sq_dists = np.maximum(sq_dists, 0.0)
    
    # Calculate squared distances between columns of u and scaled_H
    # u is (nu, N), scaled_H is (nu, N).
    # cdist expects (N_samples, n_features), so we transpose.
    # Result is (N, N) matrix where [i,j] is dist(u[:,i], scaled_H[:,j])
    sq_dists = cdist(u.T, scaled_H.T, metric='sqeuclidean')

    weights = np.exp(-sq_dists / (2 * shat**2))
    sum_weights = np.sum(weights, axis=1) 
    
    weighted_H = np.dot(scaled_H, weights.T)
    mean_shift = (weighted_H / sum_weights[None, :]) - u
    
    pot = mean_shift / shat**2
    return pot