# plom/sampling.py

import numpy as np
import time
from typing import Tuple, List, Optional, Union
from joblib import Parallel, delayed
from .utils import compute_potential, load_backend

class ItoSampler:
    """
    Generates new samples on the manifold by solving an Ito SDE.
    
    Parameters
    ----------
    f0 : float, default=1.0
        Dissipation term (damping).
    dr : float, default=0.1
        Integration step size.
    steps : int or str, default='auto'
        Number of integration steps.
    kde_bw_factor : float, default=1.0
        Bandwidth modifier for the KDE.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=-1
        Number of parallel jobs.
    """
    def __init__(self, 
                 f0: float = 1.0, 
                 dr: float = 0.1, 
                 steps: Union[int, str] = 0, 
                 kde_bw_factor: float = 1.0, 
                 potential_method: str = "auto",
                 n_jobs: int = 1, 
                 random_state: Optional[int] = None,
                 verbose: int = 1):
        
        self.f0 = f0
        self.dr = dr
        self.steps = steps
        self.kde_bw_factor = kde_bw_factor
        self.potential_method = potential_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Try loading C++ backend on initialization
        load_backend(method=self.potential_method, verbose=self.verbose>=2)

    def sample(self, Z0: np.ndarray, H: np.ndarray, basis: np.ndarray, 
               n_samples: int = 1, burn_in_steps: int = 0, 
               n_jobs: Optional[int] = None, 
               rng_override: Optional[object] = None) -> Tuple[List[np.ndarray], int]:
        """
        Run the sampling process.
        
        Z0: (nu, m) Reduced random matrix (Initial projection)
        H:  (nu, N) Reference data (usually PCA data)
        basis: (N, m) Reduced DMAPS basis
        
        Parameters
        ----------
        ...
        n_jobs : int, optional
            If provided, overrides n_jobs specified at the PLoM-class level.
            
        rng_override : object, optional
            For testing only. If provided, this specific random generator 
            (or adapter) is passed to all workers, bypassing the internal 
            SeedSequence logic.
        """
        # Setup Randomness
        if rng_override is not None:
            # TESTING MODE: Use the provided object for everything.
            # Note: This passes the SAME object to all parallel workers.
            # This is only safe if the object is stateless (like LegacyRNGAdapter)
            # or if n_jobs=1.
            worker_rngs = [rng_override] * n_samples
        else:
            # PRODUCTION MODE: Proper parallel seeding
            # We use SeedSequence to spawn reliable, independent child seeds for parallel workers.
            # This is better than just random.randint() because it ensures statistical independence.
            ss = np.random.SeedSequence(self.random_state)
            child_seeds = ss.spawn(n_samples)
            worker_rngs = [np.random.default_rng(s) for s in child_seeds]
        
        # H is (N, nu) coming from PCA, we need (nu, N) for the math kernels
        H_t = H.T
        nu, N = H_t.shape
        m = basis.shape[1]
        
        s = (4 / (N * (2 + nu))) ** (1 / (nu + 4)) * self.kde_bw_factor
        s_hat = s / np.sqrt(s**2 + (N - 1) / N)
        self.f_ac = 2.0 * np.pi * s_hat / self.dr
        
        # Calculate reduction matrix 'a'
        # a = g (g^T g)^-1
        gram = np.dot(basis.T, basis)
        a_matrix = np.dot(basis, np.linalg.inv(gram)) # (N, m)
        
        # Auto-calculate steps if needed
        if self.steps == 0 or self.steps == 'auto':
            # Logic: steps = 4 * log(100) / f0 / dr
            self.steps = int(4 * np.log(100) / self.f0 / self.dr) + 1
            
        # 1. Burn-in (Transient response)
        Z_curr = Z0
        if burn_in_steps > 0:
            # Handle burn-in RNG
            if rng_override is not None:
                burn_rng = rng_override
            else:
                burn_rng = np.random.default_rng(ss.spawn(1)[0])
                
            Z_curr, _ = self._simulate_walk(Z_curr, burn_in_steps, H_t, basis, a_matrix, rng=burn_in_rng)
            # Original code logic: If M0>0, generate one sample, then loop remainder.
            # We will return the burn-in result as the starting point for all parallel chains
            # to ensure they start from the stationary distribution.

        # 2. Parallel Generation
        # We pass a specific RNG instance (seeded uniquely) to each worker.
        joblib_verbose = 11 if self.verbose >= 2 else 0
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        results = Parallel(n_jobs=effective_n_jobs, verbose=joblib_verbose)(
            delayed(self._simulate_walk)(
                Z_curr, self.steps, H_t, basis, a_matrix, rng_obj
            ) for rng_obj in worker_rngs
        )
        
        # Unpack results
        Zs = [res[0] for res in results]
        
        return Zs, self.steps

    def _simulate_walk(self, Z: np.ndarray, t: int, H: np.ndarray, 
                       basis: np.ndarray, a: np.ndarray, 
                       rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, List]:
        """Runs a single Ito chain."""
        if rng is None:
            rng = np.random.default_rng()
        
        nu, N = H.shape
        # Initialize velocity Y
        # Y = randn(nu, N) . a
        Y = np.dot(rng.standard_normal((nu, N)), a)
        
        # steps_history = []
        for _ in range(t):
            Z, Y = self._simulate_step(Z, Y, H, basis, a, rng)
            # steps_history.append(Z) # Optional: uncomment if you need full history
            
        return Z, []

    def _simulate_step(self, Z, Y, H, basis, a, rng: Optional[np.random.Generator] = None):
        """Single integration step."""
        if rng is None:
            rng = np.random.default_rng()
        
        nu, N = H.shape
        b = self.f0 * self.dr / 4.0
        
        # Wiener process
        noise = rng.standard_normal((nu, N))
        dW = np.dot(noise * (self.dr**0.5), a)
        
        Zhalf = Z + (self.dr / 2.0) * Y
        
        # Compute Potential Gradient (L)
        # u = Zhalf * g^T
        u = np.dot(Zhalf, basis.T)
        
        L_full = compute_potential(H, u, self.kde_bw_factor)
        L_proj = np.dot(L_full, a) # Project back
        
        # Update Y and Z
        Ynext = ((1 - b) * Y + self.dr * L_proj + np.sqrt(self.f0) * dW) / (1 + b)
        Znext = Zhalf + (self.dr / 2.0) * Ynext
        
        return Znext, Ynext