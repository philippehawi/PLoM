# plom/core.py

import numpy as np
import time
from datetime import datetime
from typing import Optional, Union, Tuple, List
import pickle

from .preprocessing import Scaler, PCA
from .manifold import DiffusionMaps
from .sampling import ItoSampler
from .utils import Timer


class PLoM:
    """
    Probabilistic Learning on Manifolds (PLoM).
    
    A comprehensive class that orchestrates scaling, PCA, Diffusion Maps, 
    and Ito SDE sampling to generate new samples that respect the 
    underlying manifold geometry of the training data.

    Parameters
    ----------
    # Pipeline Control
    ------------------
    use_scaling : bool, default=True
        Whether to apply feature scaling (normalization or min-max).
    
    use_pca : bool, default=True
        Whether to apply PCA reduction before manifold learning.
    
    use_dmaps : bool, default=True
        Whether to perform Diffusion Maps analysis to learn the manifold basis.
    
    
    # Projection Configuration
    --------------------------
    projection_source : str, default='pca'
        The data matrix 'H' used for the initial projection onto the manifold (H -> Z).
        Options: 'pca', 'scaling', 'data'.
    
    projection_target : str, default='dmaps'
        The basis 'g' used for the projection. Z is computed such that H ~ Z g^T.
        Options: 'dmaps', 'pca', 'scaling', 'data'.

    
    # Hyperparameters (Scaling)
    ---------------------------
    scaling_method : str, default='Normalization'
        Method for feature scaling.
        - 'Normalization': Zero mean, unit variance.
        - 'MinMax': Scales data to [0, 1].
    
    
    # Hyperparameters (PCA)
    -----------------------
    pca_method : str, default='cum_energy'
        Method for PCA truncation ('cum_energy', 'eigv_cutoff', 'pca_dim').
    
    pca_cum_energy : float, default=1-1e-7
        Cumulative energy threshold (used if pca_method='cum_energy').
    
    pca_cutoff : float, default=0.0
        Eigenvalue cutoff threshold (used if pca_method='eigv_cutoff').
    
    pca_dim : int, default=1
        Fixed number of components (used if pca_method='pca_dim').
    
    pca_scale_evecs : bool, default=True
        Whether to scale PCA eigenvectors by 1/sqrt(lambda).
    
    
    # Hyperparameters (DMAPS)
    -------------------------
    dmaps_epsilon : float or str, default='auto'
        Kernel bandwidth for Diffusion Maps. If 'auto', it is estimated via binary search.
    
    dmaps_kappa : int, default=1
        Diffusion time parameter.
    
    dmaps_L : float, default=0.1
        Spectral gap cutoff for estimating the manifold dimension.
    
    dmaps_m_override : int, default=0
        If > 0, forces the manifold dimension to this value, ignoring dmaps_L.    
    
    
    # Hyperparameters (Sampling)
    ----------------------------
    ito_f0 : float, default=1.0
        Dissipation (damping) term for the SDE.
    
    ito_dr : float, default=0.1
        Integration step size for the SDE.
    
    ito_steps : int or str, default='auto'
        Number of integration steps. If 'auto', calculated based on f0 and dr.
    
    ito_kde_bw : float, default=1.0
        Multiplier for the Kernel Density Estimation bandwidth in the potential calculation.
    
    ito_potential_method : {'auto', 'cpp_eigen', 'cpp_native', 'python'}, default='auto'
        Backend for computing the potential gradient. 
        'auto' attempts to load the C++ library and falls back to Python.
    
    
    # General
    ---------
    n_jobs : int, default=-1
        Number of parallel jobs for sampling. -1 uses all available processors.
    
    verbose : int, default=1
        Verbosity level. 0=Silent, 1=Progress, 2=Detailed debug info.
    
    random_state : int, optional
        Seed for reproducibility.
    
    """
    def __init__(self, 
                 # Pipeline Control
                 use_scaling: bool = True,
                 use_pca: bool = True,
                 use_dmaps: bool = True,
                 projection_source: str = 'pca',
                 projection_target: str = 'dmaps',
                 
                 # Scaling Params
                 scaling_method: str = 'Normalization',
                 
                 # PCA Params
                 pca_method: str = 'cum_energy',
                 pca_cum_energy: float = 1 - 1e-7,
                 pca_cutoff: float = 0.0,
                 pca_dim: int = 1,
                 pca_scale_evecs: bool = True,
                 
                 # DMAPS Params
                 dmaps_epsilon: Union[float, str] = 'auto',
                 dmaps_kappa: int = 1,
                 dmaps_L: float = 0.1,
                 dmaps_m_override: int = 0,
                 
                 # Sampling Params
                 ito_f0: float = 1.0,
                 ito_dr: float = 0.1,
                 ito_steps: Union[int, str] = 'auto',
                 ito_kde_bw: float = 1.0,
                 ito_potential_method: str = 'auto',
                 
                 # General
                 n_jobs: int = 1,
                 verbose: int = 1,
                 random_state: Optional[int] = None):
        
        self._version = "2.0.0" # Metadata
        
        self.use_scaling = use_scaling
        self.use_pca = use_pca
        self.use_dmaps = use_dmaps
        self.projection_source = projection_source
        self.projection_target = projection_target
        self.verbose = int(verbose)
        self.random_state = random_state
        
        # Validation
        if not use_scaling and (projection_source == 'scaling' or projection_target == 'scaling'):
            raise ValueError("Cannot use 'scaling' for projection when use_scaling=False")
        if not use_pca and (projection_source == 'pca' or projection_target == 'pca'):
            raise ValueError("Cannot use 'pca' for projection when use_pca=False")
        if not use_dmaps and projection_target == 'dmaps':
            raise ValueError("Cannot use 'dmaps' for projection when use_dmaps=False")

        # Instantiate Sub-modules
        self.scaler = Scaler(method=scaling_method)
        
        self.pca = PCA(method=pca_method, 
                       cumulative_energy=pca_cum_energy,
                       eigenvalues_cutoff=pca_cutoff,
                       n_components=pca_dim,
                       scale_evecs=pca_scale_evecs)
        
        self.dmaps = DiffusionMaps(epsilon=dmaps_epsilon,
                                   kappa=dmaps_kappa,
                                   L=dmaps_L,
                                   m_override=dmaps_m_override,
                                   verbose=self.verbose)
        
        self.sampler = ItoSampler(f0=ito_f0,
                                  dr=ito_dr,
                                  steps=ito_steps,
                                  kde_bw_factor=ito_kde_bw,
                                  potential_method=ito_potential_method,
                                  n_jobs=n_jobs,
                                  random_state=random_state,
                                  verbose=self.verbose)
        
        # Fitted Data Storage
        self.X_train_ = None
        self.X_scaled_ = None
        self.H_pca_ = None
        self.g_dmaps_ = None
        
        # Projection Internals
        self.Z_init_ = None
        self.a_ = None
        
        self.is_fitted = False
    
    
    # --- Properties (Getters) ---
    @property
    def training_data(self):
        """The original training data."""
        return self.X_train_
    
    @property
    def dmaps_basis(self):
        """The reduced Diffusion Maps basis vectors (g)."""
        return self.dmaps.basis_
    
    @property
    def dmaps_eigenvalues(self):
        """The Diffusion Maps eigenvalues."""
        return self.dmaps.eigenvalues_

    @property
    def pca_components(self):
        """The PCA projection components."""
        return self.pca.components_
    
    
    # --- Methods ---
    def fit(self, X: np.ndarray) -> 'PLoM':
        """
        Fit the PLoM model to the training data.
        
        Executes the configured pipeline stages (Scaling -> PCA -> DMAPS -> Projection).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        
        Returns
        -------
        self : PLoM
            The fitted model.
        """
        start_time = time.time()
        
        X = self._validate_data(X)
        self.X_train_ = X
        current_data = X
        
        if self.verbose >= 1:
            print(f"\nPLoM v{self._version} Fitting... ({str(datetime.now()).split('.')[0]})")
            print("\nInput data dimensions:", self.X_train_.shape)
        
        # 1. Scaling
        if self.use_scaling:
            with Timer(f"Scaling ({self.scaler.method})", self.verbose):
                self.X_scaled_ = self.scaler.fit_transform(current_data)
                current_data = self.X_scaled_
        else: self.X_scaled_ = None

        # 2. PCA
        if self.use_pca:
            with Timer(f"PCA ({self.pca.method})", self.verbose):
                self.H_pca_ = self.pca.fit_transform(current_data)
                
            if self.verbose >= 1:
                print(f"   > Dimensions: {current_data.shape} -> {self.H_pca_.shape}")
                print(f"   > Retained eigenvalues: {self.pca.eigenvalues_trunc_[::-1]}")
            if self.verbose >= 2 and self.pca.explained_variance_ratio_ is not None:
                # print(f"   > Full eigenvalues (top 20): {self.pca.eigenvalues_[:20]}")
                print(f"   > Full eigenvalues: {self.pca.eigenvalues_}")
                kept = np.sum(self.pca.eigenvalues_trunc_) / np.sum(self.pca.eigenvalues_)
                print(f"   > Energy Kept: {kept:.6f}")
                
            current_data = self.H_pca_
        else: self.H_pca_ = None
        
        # 3. Diffusion Maps
        if self.use_dmaps:
            with Timer("Diffusion Maps", self.verbose):
                self.dmaps.fit(current_data)
                self.g_dmaps_ = self.dmaps.basis_
            
            if self.verbose >= 1:
                print(f"   > Epsilon: {self.dmaps.epsilon_fitted_:.4f}")
                print(f"   > Manifold dim (m): {self.dmaps.dimension_}")
                dmaps_top_eigenvalues = self.dmaps.eigenvalues_[1:self.dmaps.dimension_+1]
                print(f"   > Manifold top eigenvalues: {dmaps_top_eigenvalues} \
[{self.dmaps.eigenvalues_[self.dmaps.dimension_+1]:.8f} ...]")
            
            if self.verbose >= 2:
                print(f"   > Search history (eps -> m):")
                # Determine padding based on the largest epsilon string length
                # or just use a safe fixed width (e.g. 10 chars)
                print(f"     {'Epsilon':>10} | {'Dim (m)':<5}")
                print(f"     {'-'*10}-+-{'-'*5}")
                
                for eps, m in self.dmaps.eps_search_history_:
                    # .4f rounds epsilon to 4 decimal places
                    # .0f prints m as an integer
                    print(f"     {eps:10.4f} | {m:<5.0f}")
                    
        else: self.g_dmaps_ = None
        
        # 4. Projection
        with Timer("Projection (Z)", self.verbose):
            self._compute_projection()
        
        if self.verbose >= 1:
            print(f"   > Projected data (Z) dimensions: {self.Z_init_.shape}")
        
        self.is_fitted = True
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\nFitting Complete ({elapsed:.3f}s)")
            print(str(datetime.now()).split('.')[0])
            
        return self

    def sample(self, n_samples: int = 1, burn_in_steps: int = 0, 
               n_jobs: Optional[int] = None, rng_override=None) -> np.ndarray:
        """
        Generate new samples from the learned manifold.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of distinct PLoM sets to generate. 
            Note: Since PLoM is a particle method, one "sample" returns a set of 
            particles equal in size to the training set (N).
            If n_samples=1, returns (N, n_features).
            If n_samples>1, returns (n_samples * N, n_features).
        
        burn_in_steps : int, default=0
            Number of initial Ito steps to discard.
        
        n_jobs : int, optional
            If provided, overrides n_jobs specified at the PLoM-class level.

        rng_override : object, optional
            Internal use for testing (forces a specific RNG adapter).
        
        
        Returns
        -------
        X_new : np.ndarray
            Generated samples in the original data space.
        """
        start_time = time.time()
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling.")
            
        if self.verbose >= 1:
            print(f"\nPLoM v{self._version} Sampling... ({str(datetime.now()).split('.')[0]})")
        
        # Get Reference Matrices
        H_ref = self._get_projection_matrix(self.projection_source)
        g_ref = self._get_projection_matrix(self.projection_target)

        # 1. Ito SDE Sampling
        with Timer(f"Ito Sampling ({n_samples} samples)", self.verbose):
            Zs_new, steps = self.sampler.sample(self.Z_init_, 
                                                H_ref, 
                                                g_ref, 
                                                n_samples,
                                                burn_in_steps,
                                                n_jobs,
                                                rng_override)
        
        if self.verbose >= 1:
            print(f"   > Steps taken: {steps}")
            print(f"   > F_ac = {self.sampler.f_ac:.3f}")
            
        # 2. Inverse Projection & Reconstruction
        X_generated = []
        
        for Z in Zs_new:
            # Reconstruct H (Projection Source space)
            # H = g Z^T
            H_rec = np.dot(g_ref, Z.T) # (N, nu)
            
            # Unwind the pipeline
            current_rec = H_rec
            
            # If source was PCA, Inverse PCA
            if self.projection_source == 'pca' and self.use_pca:
                current_rec = self.pca.inverse_transform(current_rec)
            
            # If source was Scaling (or result of Inverse PCA), Inverse Scaling
            if self.use_scaling:
                # We check logical flow: 
                # If source='pca', we just inverted PCA, now we are in Scaled space.
                # If source='scaling', we are already in Scaled space.
                # If source='data', we are already in Data space (no inversion needed).
                
                # We apply inverse scaling if we are currently in the scaled domain.
                # This logic depends on what H_ref represented.
                if self.projection_source in ['pca', 'scaling']:
                     current_rec = self.scaler.inverse_transform(current_rec)

            X_generated.append(current_rec)
            
        final_output = np.vstack(X_generated)
        
        if self.verbose >= 1:
            print(f"   > Generated {final_output.shape[0]} points")
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\nSampling Complete in {elapsed:.2f}s ({str(datetime.now()).split('.')[0]})")
        
        return final_output

    def save(self, filepath: str):
        """Save the fitted model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        if self.verbose >= 1:
            print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'PLoM':
        """Load a fitted model from a file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _compute_projection(self):
        H = self._get_projection_matrix(self.projection_source)
        g = self._get_projection_matrix(self.projection_target)
        
        if H is None or g is None:
            raise RuntimeError("Projection matrices not computed. Check pipeline flags.")

        gram_inv = np.linalg.inv(np.dot(g.T, g))
        self.a_ = np.dot(g, gram_inv)
        self.Z_init_ = np.dot(H.T, self.a_)

    def _get_projection_matrix(self, source_name):
        if source_name == 'pca': return self.H_pca_
        if source_name == 'scaling': return self.X_scaled_
        if source_name == 'dmaps': return self.g_dmaps_
        if source_name == 'data': return self.X_train_
        raise ValueError(f"Unknown source: {source_name}")
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input data contains NaNs or Infs.")
        return X

    def __repr__(self):
        return (f"PLoM(use_pca={self.use_pca}, "
                f"dmaps_epsilon={self.dmaps.epsilon}, "
                f"ito_steps={self.sampler.steps})")