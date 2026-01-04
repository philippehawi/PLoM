# plom/preprocessing.py

import numpy as np
from typing import Optional, Tuple, Union

class Scaler:
    """
    Scales data to a specified range or distribution.
    
    Parameters
    ----------
    method : str, default='Normalization'
        The scaling method to use. 
        - 'Normalization': Zero mean and unit variance.
        - 'MinMax': Scale to [0, 1] interval.
    """
    def __init__(self, method: str = 'Normalization'):
        self.method = method
        self.centers_ = None
        self.scales_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'Scaler':
        """Compute the mean and scale to be used for later scaling."""
        if self.method == 'MinMax':
            self.centers_ = np.min(X, axis=0)
            self.scales_ = np.max(X, axis=0) - self.centers_
        elif self.method == 'Normalization':
            self.centers_ = np.mean(X, axis=0)
            self.scales_ = np.std(X, axis=0)
        
        # Handle constant features to avoid division by zero
        if self.scales_ is not None:
            self.scales_[self.scales_ == 0.0] = 1.0
            
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling to X."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        return (X - self.centers_) / self.scales_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform.")
        return X * self.scales_ + self.centers_


class PCA:
    """
    Custom PCA implementation for PLoM.
    
    Supports truncation by cumulative energy, eigenvalue cutoff, or fixed dimension.
    Crucially, it supports scaling eigenvectors by 1/sqrt(eigenvalue), required for PLoM.

    Parameters
    ----------
    method : str, default='cum_energy'
        Method to select dimension ('cum_energy', 'eigv_cutoff', 'pca_dim').
    cumulative_energy : float, default=1-1e-7
        Threshold for cumulative energy truncation.
    eigenvalues_cutoff : float, default=0.0
        Threshold for eigenvalue truncation.
    n_components : int, default=1
        Fixed number of components (used if method='pca_dim').
    scale_evecs : bool, default=True
        If True, scaled eigenvectors by inverse sqrt of eigenvalues.
    """
    def __init__(self, 
                 method: str = 'cum_energy', 
                 cumulative_energy: float = 1 - 1e-7,
                 eigenvalues_cutoff: float = 0.0,
                 n_components: int = 1,
                 scale_evecs: bool = True):
        self.method = method
        self.cumulative_energy = cumulative_energy
        self.eigenvalues_cutoff = eigenvalues_cutoff
        self.n_components = n_components
        self.scale_evecs = scale_evecs
        
        # Attributes set during fit
        self.mean_ = None
        self.components_ = None # The projection matrix (V)
        self.inverse_components_ = None # Matrix for reconstruction
        self.eigenvalues_ = None
        self.eigenvalues_trunc_ = None
        self.n_features_in_ = None
        self.explained_variance_ratio_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'PCA':
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # 1. Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Covariance and Eigendecomposition
        cov = np.cov(X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues descending (Largest to Smallest)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        self.eigenvalues_ = eigvals
        tot_energy = np.sum(eigvals)
        self.explained_variance_ratio_ = eigvals / tot_energy
        
        # 3. Truncation Logic
        if self.method == 'eigv_cutoff':
            mask = eigvals > self.eigenvalues_cutoff
            nu = np.sum(mask)
            
        elif self.method == 'cum_energy':
            tot_energy = np.sum(eigvals)
            
            # Logic: We keep summing largest eigenvalues until we hit the target energy.
            current_energy = 0.0
            nu = len(eigvals)
            for i in range(len(eigvals)):
                current_energy += eigvals[i]
                if (current_energy / tot_energy) >= self.cumulative_energy:
                    nu = i + 1
                    break

        elif self.method == 'pca_dim':
            nu = self.n_components
               
        else:
            raise ValueError(f"Unknown PCA method: {self.method}")
        
        self.n_components_ = nu
        
        self.eigenvalues_trunc_ = eigvals[:nu]
        eigvecs_trunc = eigvecs[:, :nu]

        # 4. Scaling Eigenvectors (Specific to PLoM)
        # scaled_eigvecs = eigvecs / sqrt(lambda)
        sqrt_eigvals = np.sqrt(self.eigenvalues_trunc_)
        
        if self.scale_evecs:
            self.components_ = eigvecs_trunc / sqrt_eigvals[None, :]
            self.inverse_components_ = eigvecs_trunc * sqrt_eigvals[None, :]
        else:
            self.components_ = eigvecs_trunc
            self.inverse_components_ = eigvecs_trunc
        
        # --- LEGACY COMPATIBILITY PATCH ---
        # Reverse to Ascending Order (Smallest -> Largest Eigenvalue)
        # to match original PLoM script behavior.
        self.eigenvalues_trunc_ = self.eigenvalues_trunc_[::-1]
        self.components_ = self.components_[:, ::-1]
        self.inverse_components_ = self.inverse_components_[:, ::-1]
        # ----------------------------------

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted before calling transform.")
        X_centered = X - self.mean_
        # Project: X @ V
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Project back to original space."""
        if not self.is_fitted:
            raise RuntimeError("PCA must be fitted before calling inverse_transform.")
        # Reconstruct: Z @ V_inv.T + mean
        return np.dot(Z, self.inverse_components_.T) + self.mean_