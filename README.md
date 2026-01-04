# PLoM: Probabilistic Learning on Manifolds

**A modern Python library for generative modeling on manifolds.**

This package is the object-oriented Python implementation of the PLoM algorithm, accompanying the research by [Soize and Ghanem, 2016](https://doi.org/10.1016/j.jcp.2016.05.044) and [Soize and Ghanem, 2020](https://doi.org/10.48550/arXiv.2002.12653).

## Overview

PLoM generates replicas of a given dataset (training data) where the small data challenge (small $N$) is exacerbated by high-dimensional data (large $n$). The training set is construed as a graph in $R^n$ with $N$-vertices, and each replica itself a graph that shares key features with the training set.

PLoM constraints the generated samples to the **diffusion manifold** of the training data. The diffusion coordinates provide an embedding of the initial data into an m-dimensional Euclidean space that preserves geometric structure over multiple scales. The samples are generated through a projected Ito stochastic differential equation whose invariant measure is constrained to the diffusion manifold (the span of the diffusion coordinates).

**New in version 2.0:**
* **Object-Oriented API**: Familiar `fit` / `sample` interface compatible with scikit-learn.
* **Efficiency**: Optimized C++ backend for potential gradients and memory-efficient distance calculations.
* **Reproducibility**: Robust random state management for parallel execution.

---

## Installation

You can install the package directly using `pip`:

```bash
pip install pyplom
```

Or install from source in editable mode (recommended for development):

```bash
git clone https://github.com/philippe-hawi/PLoM.git
cd PLoM
pip install -e .
```

*Note: This package includes compiled C++ extensions for performance.*

---

## Quick Start

The new API replaces the old dictionary-based workflow with a cleaner class-based approach.

### 1. Basic Usage

```python
import numpy as np
from plom import PLoM

# 1. Load Data (N samples x n_features)
X_train = np.loadtxt('data/training_data.txt')

# 2. Initialize Model
model = PLoM(
    use_pca=True,              # Enable/Disable pipeline stages
    use_dmaps=True,
    pca_method='cum_energy',   # PCA configuration
    pca_cum_energy=0.99,
    dmaps_epsilon='auto',      # Auto-tune kernel bandwidth
    ito_steps='auto',          # Number of SDE integration steps
    n_jobs=1,                  # Parallel processing option
    verbose=1                  # 0, 1, 2
)

# 3. Fit the Manifold
model.fit(X_train)

# 4. Generate New Samples
# Returns (n_samples * N) points
X_new = model.sample(n_samples=1)

print(f"Generated data shape: {X_new.shape}")

```

### 2. Saving and Loading Models

```python
# Save the trained model
model.save("my_plom_model.pkl")

# Load it later
from plom import PLoM
loaded_model = PLoM.load("my_plom_model.pkl")
samples = loaded_model.sample(n_samples=5)

```

---

## Key Parameters

The `PLoM` class manages the entire pipeline. Key arguments include:

* **Pipeline Control**: `use_scaling`, `use_pca`, `use_dmaps` (bools) to toggle specific stages.
* **PCA**: `pca_method` ('cum_energy', 'eigv_cutoff', 'pca_dim') controls dimensionality reduction.
* **Diffusion Maps**: `dmaps_epsilon` ('auto' or float) sets the kernel bandwidth; `dmaps_m_override` forces a specific manifold dimension.
* **Sampling**: `ito_steps` (int) controls the length of the random walk; `n_jobs` sets the number of CPU cores.

---

## Legacy Code

If you are looking for the original dictionary-based implementation (v0.7.0 and earlier), it has been moved to the **[PLoM-Legacy](https://github.com/philippe-hawi/PLoM-Legacy)** repository.

To install the legacy version alongside this new version:

```bash
git clone https://github.com/philippe-hawi/PLoM-Legacy.git
cd PLoM-Legacy
pip install -e 
```

---

## Dependencies

* Python 3.8+
* `numpy` >= 1.20
* `scipy` >= 1.7
* `joblib` >= 1.1
* `matplotlib` (optional, for plotting helpers)

---

## References

1. Soize, C., & Ghanem, R. (2016). Data-driven probability concentration and sampling on manifold. *Journal of Computational Physics*, 321, 242-258.
2. Soize, C., & Ghanem, R. (2020). Physics-constrained non-Gaussian probabilistic learning on manifolds. *arXiv preprint arXiv:2002.12653*.


## Examples
* [Example 1: 2 circles in 2 dimensions](examples/circles2d/circles_script.ipynb)
* [Example 2: spiral in 3 dimensions](examples/spiral3d/spiral_script.ipynb)
