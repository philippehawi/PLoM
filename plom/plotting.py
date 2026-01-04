# plom/plotting.py

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def plot_2d_scatter(X, Y=None, title="Scatter Plot", labels=("Dim 1", "Dim 2")):
    """
    Simple 2D scatter comparison.
    If Y is provided, overlays X (blue) and Y (red).
    """
    if not HAS_MATPLOTLIB:
        print("Error: Matplotlib not installed.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Original')
    
    if Y is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c='red', alpha=0.5, label='Generated')
        plt.legend()
        
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_3d_scatter(X, Y=None, title="3D Scatter Plot"):
    """
    3D scatter comparison.
    """
    if not HAS_MATPLOTLIB:
        print("Error: Matplotlib not installed.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', alpha=0.4, label='Original')
    
    if Y is not None:
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='red', alpha=0.4, label='Generated')
        ax.legend()
        
    ax.set_title(title)
    plt.show()

def plot_eigenvalues(model, n_show=20):
    """
    Plots the Diffusion Maps eigenvalues.
    
    Parameters
    ----------
    model : PLoM instance
        A fitted PLoM model.
    n_show : int
        Number of eigenvalues to display.
    """
    if not HAS_MATPLOTLIB:
        print("Error: Matplotlib not installed.")
        return

    if not model.is_fitted or model.dmaps_eigenvalues is None:
        print("Model is not fitted or DMAPS was not used.")
        return

    evals = model.dmaps_eigenvalues
    n = min(len(evals), n_show)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n+1), evals[:n], 'o-', markerfacecolor='white')
    plt.title("Diffusion Maps Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()