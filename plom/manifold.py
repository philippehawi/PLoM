# plom/manifold.py

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, List, Union

class DiffusionMaps:
    """
    Computes the Diffusion Maps basis for the manifold.
    
    Parameters
    ----------
    epsilon : float or str, default='auto'
        Kernel width. If 'auto', an optimal value is estimated.
    kappa : int, default=1
        Diffusion time (power of the transition matrix).
    L : float, default=0.1
        Spectral gap cutoff criteria for dimension estimation.
    m_override : int, default=0
        If > 0, forces the manifold dimension to this value.
    dist_method : str, default='standard'
        Distance metric ('standard' or 'periodic').
    """
    def __init__(self, 
                 epsilon: Union[float, str] = 'auto', 
                 kappa: int = 1,
                 L: float = 0.1,
                 m_override: int = 0,
                 dist_method: str = 'standard',
                 verbose: int = 1):
        self.epsilon = epsilon
        self.kappa = kappa
        self.L = L
        self.m_override = m_override
        self.dist_method = dist_method
        self.verbose = verbose
        
        # Results
        self.basis_ = None       # The reduced basis (g)
        self.full_basis_ = None  # All eigenvectors
        self.eigenvalues_ = None
        self.dimension_ = None   # m
        self.epsilon_fitted_ = None
        self.eps_search_history_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'DiffusionMaps':
        """
        Fit the Diffusion Maps basis to the data X.
        X is typically the PCA-reduced data (nu x N).
        """
        # Note: Original code expects X to be (N, nu). 
        # If X is the output of PlomPCA, it is (N, nu).
        
        if self.epsilon == 'auto':
            self.epsilon_fitted_, self.dimension_, self.eps_search_history_ = \
                self._find_optimal_epsilon(X)
        else:
            # Handle list of epsilons if provided, or single float
            eps_list = np.atleast_1d(self.epsilon)
            # In original logic, if a list is provided, it iterates but only keeps the last one.
            # We will simplify: take the last one if a list is passed.
            self.epsilon_fitted_ = float(eps_list[-1])
            self.dimension_ = self._estimate_dimension(X, self.epsilon_fitted_)
        
        # If user overrides dimension
        if self.m_override > 0:
            self.dimension_ = self.m_override

        # Compute Final Basis
        self.basis_, self.eigenvalues_, self.full_basis_ = \
            self._compute_basis(X, self.epsilon_fitted_)
        
        self.is_fitted = True
        return self

    def _compute_basis(self, X: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core computation of diffusion coordinates."""        
        # 1. Pairwise squared euclidean distances (memory efficient)
        # metric='sqeuclidean' returns squared distances directly
        dists_sq = cdist(X, X, metric='sqeuclidean')
    
        # 2. Kernel
        diffusions = np.exp(-dists_sq / epsilon)
        
        # 3. Normalize (Graph Laplacian normalization)
        scales = np.sum(diffusions, axis=0)**0.5
        # P = D^-1 * K * D^-1
        normalized = diffusions / (scales[:, None] * scales[None, :])
        
        # 4. Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(normalized)
        
        # Sort descending
        # Original code uses np.flip because eigh returns ascending
        eigvals = np.flip(eigvals)
        eigvecs = np.flip(eigvecs, axis=1)
        
        # Convert back to basis vectors: psi = phi / scale
        basis_vectors = eigvecs / scales[:, None]
        
        # Apply time parameter kappa
        basis = basis_vectors * (eigvals[None, :] ** self.kappa)
        
        # Return reduced basis (excluding first trivial eigenvector)
        # s=1, e=m+1
        m = self.dimension_ if self.dimension_ is not None else len(eigvals)-1
        reduced_basis = basis[:, 1 : m + 1]
        
        return reduced_basis, eigvals, basis

    def _estimate_dimension(self, X: np.ndarray, epsilon: float) -> int:
        """Estimate m based on spectral gap L."""
        _, eigvals, _ = self._compute_basis(X, epsilon)
        
        # Logic: if lambda[a] / lambda[1] < L, then cut.
        # Note: eigvals[0] is always 1.0 (trivial). We compare to eigvals[1].
        m = len(eigvals) - 1
        lambda_1 = eigvals[1]
        
        for a in range(2, len(eigvals)):
            ratio = eigvals[a] / lambda_1
            if ratio < self.L:
                m = a - 1
                break
        return m
    
    def _find_optimal_epsilon(self, X: np.ndarray) -> Tuple[float, int, np.ndarray]:
        """
        Search for epsilon that minimizes manifold dimension.
        Replicates the exact search logic from the original PLoM script:
        1. Coarse search to find target min dimension.
        2. Linear refinement to bracket the transition.
        3. Binary search to pinpoint epsilon.
        4. Final linear adjustment.
        """
        if self.verbose >= 1:
            print("   > Searching for optimal epsilon...")
        
        # 1. Initialization and Coarse Search
        epsilon_list = [0.1, 1, 2, 8, 16, 32, 64, 100, 10000]
        eps_for_m_target = [1, 10, 100, 1000, 10000]
        
        history = []
        
        # Find target dimension (min possible m)
        m_targets = []
        for eps in eps_for_m_target:
            m = self._estimate_dimension(X, eps)
            m_targets.append(m)
        
        m_target = min(m_targets)
        
        # Initial bounds based on coarse search
        # upper_bound starts at the epsilon that gave us the best m
        best_idx = np.argmin(m_targets)
        upper_bound = eps_for_m_target[best_idx]
        lower_bound = epsilon_list[0]
        
        # 2. Linear Refinement (Bracket the transition)
        for eps in epsilon_list[1:]:
            m = self._estimate_dimension(X, eps)
            history.append([eps, m])
            
            if m > m_target:
                lower_bound = eps
            else:
                upper_bound = eps
                break
        
        # 3. Binary Search
        while (upper_bound - lower_bound) > 0.5:
            middle_bound = (lower_bound + upper_bound) / 2
            m = self._estimate_dimension(X, middle_bound)
            history.append([middle_bound, m])
            
            if m > m_target:
                lower_bound = middle_bound
            else:
                upper_bound = middle_bound
                
        # 4. Final Adjustment (Ensure we satisfy the target condition)
        # Check current lower_bound
        m = self._estimate_dimension(X, lower_bound)
        
        # If lower_bound still yields too high dimension, nudge it up until it fits
        while m > m_target:
            lower_bound += 0.1
            m = self._estimate_dimension(X, lower_bound)
            history.append([lower_bound, m])
            
        epsilon = lower_bound
        
        # Clean up history array
        history_arr = np.unique(np.array(history), axis=0) if history else np.array([])
        
        return epsilon, m_target, history_arr