# File: plom.py
# File Created: Monday, 8th July 2019 10:25:45 am
# Author: Philippe Hawi (hawi@usc.edu)

"""
Tools for learning an intrinsic manifold using Diffusion-Maps, sampling new 
data on the manifold by solving an Ito SDE, and conditioning using 
non-parametric density estimations.
"""

import pickle
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
import ctypes
import os


def initialize(training=None,
               
               scaling=True,
               scaling_method='Normalization',
               
               pca=True,
               pca_method='cum_energy',
               pca_cum_energy=1-1e-5,
               pca_eigv_cutoff=0,
               pca_dim=1,
               pca_scale_evecs=True,
               
               dmaps=True,
               dmaps_epsilon='auto',
               dmaps_kappa=1,
               dmaps_L=0.1,
               dmaps_first_evec=False,
               dmaps_m_override=0,
               dmaps_dist_method='standard',
               
               projection=True,
               projection_source='pca',
               projection_target='dmaps',
               
               sampling=True,
               num_samples=1,
               ito_f0=1,
               ito_dr=0.1,
               ito_steps='auto',
               ito_pot_method=3,
               ito_kde_bw_factor=1,
               
               parallel=False,
               n_jobs=-1,
               
               save_samples=True,
               samples_fname=None,
               samples_fmt='txt',
               
               job_desc="Job 1",
               verbose=True
               ):
    
    inp_params_dict = dict()
    inp_params_dict['pca_cum_energy']        = pca_cum_energy
    inp_params_dict['pca_eigv_cutoff']       = pca_eigv_cutoff
    inp_params_dict['pca_dim']               = pca_dim
    inp_params_dict['dmaps_epsilon']         = dmaps_epsilon
    inp_params_dict['dmaps_kappa']           = dmaps_kappa
    inp_params_dict['dmaps_L']               = dmaps_L
    inp_params_dict['ito_f0']                = ito_f0
    inp_params_dict['ito_dr']                = ito_dr
    inp_params_dict['ito_steps']             = ito_steps
    inp_params_dict['ito_kde_bw_factor']     = ito_kde_bw_factor
    inp_params_dict['ito_num_samples']       = num_samples
    
    options_dict = dict()
    options_dict['scaling']           = scaling
    options_dict['scaling_method']    = scaling_method
    options_dict['pca']               = pca   
    options_dict['pca_method']        = pca_method
    options_dict['pca_scale_evecs']   = pca_scale_evecs
    options_dict['dmaps']             = dmaps
    options_dict['dmap_first_evec']   = dmaps_first_evec
    options_dict['dmaps_m_override']  = dmaps_m_override
    options_dict['dmaps_dist_method'] = dmaps_dist_method
    options_dict['projection']        = projection
    options_dict['sampling']          = sampling
    options_dict['projection_source'] = projection_source
    options_dict['projection_target'] = projection_target
    options_dict['ito_pot_method']    = ito_pot_method
    options_dict['ito_kde_bw_factor'] = ito_kde_bw_factor
    options_dict['parallel']          = parallel
    options_dict['n_jobs']            = n_jobs
    options_dict['save_samples']      = save_samples
    options_dict['samples_fname']     = samples_fname
    options_dict['samples_fmt']       = samples_fmt
    options_dict['verbose']           = verbose
    
    scaling_dict = dict()
    scaling_dict['training']         = None
    scaling_dict['centers']          = None
    scaling_dict['scales']           = None
    scaling_dict['reconst_training'] = None
    scaling_dict['augmented']        = None
    
    pca_dict = dict()
    pca_dict['training']         = None
    pca_dict['scaled_evecs_inv'] = None
    pca_dict['scaled_evecs']     = None
    pca_dict['evecs']            = None
    pca_dict['mean']             = None
    pca_dict['eigvals']          = None
    pca_dict['eigvals_trunc']    = None
    pca_dict['reconst_training'] = None
    pca_dict['augmented']        = None
    
    dmaps_dict = dict()
    dmaps_dict['training']      = None
    dmaps_dict['eigenvectors']  = None
    dmaps_dict['eigenvalues']   = None
    dmaps_dict['dimension']     = None
    dmaps_dict['epsilon']       = None
    dmaps_dict['basis']         = None
    dmaps_dict['reduced_basis'] = None
    dmaps_dict['eps_vs_m']      = None
    
    ito_dict = dict()
    ito_dict['Z0']            = None
    ito_dict['a']             = None
    ito_dict['Zs']            = None
    ito_dict['Zs_steps']      = None
    ito_dict['t']             = None
    
    data_dict = dict()
    data_dict['training']         = training
    data_dict['augmented']        = None
    data_dict['reconst_training'] = None
    data_dict['rmse']             = None

    plom_dict = dict()
    plom_dict['job_desc'] = job_desc
    plom_dict['data']     = data_dict
    plom_dict['input']    = inp_params_dict
    plom_dict['options']  = options_dict
    plom_dict['scaling']  = scaling_dict
    plom_dict['pca']      = pca_dict
    plom_dict['dmaps']    = dmaps_dict
    plom_dict['ito']      = ito_dict
    plom_dict['summary']  = None
    
    if scaling == False:
        if projection_source == "scaling":
            raise ValueError("Cannot set <projection_source> to 'scaling' when <scaling> is set to False")
        if projection_target == "scaling":
            raise ValueError("Cannot set <projection_target> to 'scaling' when <scaling> is set to False")
    
    if pca == False:
        if projection_source == "pca":
            raise ValueError("Cannot set <projection_source> to 'pca' when <pca> is set to False")
        if projection_target == "pca":
            raise ValueError("Cannot set <projection_target> to 'pca' when <pca> is set to False")
    
    if dmaps == False and projection_target == "dmaps":
        raise ValueError("Cannot set <projection_target> to 'dmaps' when <dmaps> is set to False")
        

    return plom_dict
###############################################################################
def parse_input(input_file="input.txt"):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    lines = [line.lstrip() for line in lines if 
             not line.lstrip().startswith(('*', '#')) and len(line.lstrip())]
    
    lines = [line.split('#')[0].rstrip() for line in lines]
    
    lines = [line.replace("'", "").replace('"', '') for line in lines]
    
    args = dict()
    for line in lines:
        split = line.split()
        key = split[0]
        val = line.replace(key, '', 1).lstrip()
        try:
            val = int(val)
        except:
            try:
                val = float(val)
            except:
                pass
    
        if (val == "True" or val == "true"):
            val = True
        if (val == "False" or val == "false"):
            val = False
        if (val == "None" or val == "none"):
            val = None
        
        if key == "training":
            try:
                val = np.loadtxt(val)
            except:
                try:
                    val = np.load(val)
                except:
                    raise OSError("Training data file not found. File should" +
                                  " be raw text or Numpy array (.npy)")
        
        args[key] = val
    
    return args
    
###############################################################################
def _scaleMinMax(X, verbose=True):
    """
    Scale the features of dataset X.
    
    Data is scaled to be in interval [0, 1]. If any dimensions have zero range
    (constant vector), then set that scale to 1.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data that will be scaled.
    
    verbose : bool, (default is True)
        If True, print relevant information. The default is True.
    
    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled data.
    
    means : ndarray of shape (n_features, )
        Means of individual features.
    
    scales : ndarray of shape (n_features, )
        Range of individual features.

    """
    if verbose:
        print("\n\nScaling data.")
        print("-------------")
        print("Input data dimensions:", X.shape)
        print("Using 'MinMax' scaler.")

    Xmin = np.min(X, axis=0)
    Xmax = np.max(X, axis=0)
    scale = Xmax - Xmin
    scale[scale==0] = 1
    X_scaled = (X - Xmin) / scale
    if verbose:
        # print("Scaling complete.")
        print("Output data dimensions:", X_scaled.shape)
    return X_scaled, Xmin, scale

###############################################################################
def _scaleNormalize(X, verbose=True):
    """
    Scale the features of dataset X.
    
    Data is scaled to have zero mean and unit variance. If any dimensions have
    zero variance, then set that scale to 1.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data that will be scaled.
    
    verbose : bool, optional
        If True, print relevant information. The default is True.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled data.
    
    means : ndarray of shape (n_features, )
        Means of individual features.
    
    scales : ndarray of shape (n_features, )
        Standard deviations of individual features.

    """
    if verbose:
        print("\n\nScaling data.")
        print("-------------")
        print("Input data dimensions:", X.shape)
        print("Using 'Normalization' scaler.")
    
    means, scales = np.mean(X, axis=0), np.std(X, axis=0)
    scales[scales == 0.0] = 1.0
    X_scaled = (X - means) / scales
    if verbose:
        # print("Scaling complete.")
        print("Output data dimensions:", X_scaled.shape)
    return X_scaled, means, scales

###############################################################################
def scale(plom_dict):
    """
    PLoM wrapper for scaling functions.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    training       = plom_dict['data']['training']
    scaling_method = plom_dict['options']['scaling_method']
    verbose        = plom_dict['options']['verbose']
    
    if scaling_method == "MinMax":
        scaled_train, centers, scales = _scaleMinMax(training, verbose)
    elif scaling_method == "Normalization":
        scaled_train, centers, scales = _scaleNormalize(training, verbose)
    
    plom_dict['scaling']['training'] = scaled_train
    plom_dict['scaling']['centers']  = centers
    plom_dict['scaling']['scales']   = scales

###############################################################################
def _inverse_scale(X, centers, scales, verbose=True):
    """
    Scale back the data to the original representation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data set to be scaled back to original representation.
    
    means : ndarray of shape (n_features, )
        Means of individual features.
    
    scales : ndarray of shape (n_features, )
        Range of individual features.

    Returns
    -------
    X_unscaled : ndarray of shape (n_samples, n_features)
        Unsaled data set.

    """
    return X * scales + centers

###############################################################################
def inverse_scale(plom_dict):
    """
    PLoM wrapper for inverse scaling function.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    centers = plom_dict['scaling']['centers']
    scales  = plom_dict['scaling']['scales']
    
    reconst_training = plom_dict['scaling']['reconst_training']
    augmented        = plom_dict['scaling']['augmented']
    
    if reconst_training is not None:
        X = _inverse_scale(reconst_training, centers, scales)
        plom_dict['data']['reconst_training'] = X
    
    if augmented is not None:
        X = _inverse_scale(augmented, centers, scales)
        plom_dict['data']['augmented'] = X

###############################################################################
def _pca(X, method='cum_energy', cumulative_energy=(1-1e-7), 
         eigenvalues_cutoff=0, pca_dim=1, scale_evecs=True, verbose=True):    
    """
    Normalize data set (n_samples x n_features) via PCA.
    
    Parameters
    ----------
    X: ndarray of shape (n_samples, n_features)
        Input data that will be normalized via PCA.
    
    method : string, optional (default is 'cum_energy')
        Method to select dimension (nu) of truncated basis.
        If 'cum_energy', use 'cumulative_energy' (see below).
        If 'eigv_cutoff', use 'eigenvalues_cutoff' (see below).
        If 'pca_dim', use 'pca_dim' (see below).
    
    cumulative_energy : float, (default is (1-1e-7))
        Used if method = 'cum_energy'.
        Specifies the total cumulative energy needed to truncate the basis. 
        The dimension 'nu' is selected as the smallest integer such that the 
        sum of the largest 'nu' eigenvalues divided by the sum of all 
        eigenvalues is greater than or equal to 'cumulative_energy'.
    
    eigenvalues_cutoff : float, (default is 0)
        Used if method = 'eigv_cutoff'.
        Specifies the smallest eigenvalue for which an eigenvector (basis 
        vector) is retained. Eigenvectors associated with eigenvalues smaller 
        than this cutoff value are dropped.
    
    pca_dim : int, optional (default is 1)
        Used if method = 'pca_dim'.
        Specifies dimension (nu = pca_dim) to be used for the truncated basis.
    
    scale_evecs: bool, optional (default is True)
        If True, the principal components (eigenvectors of the covariance matrix
        of X, onto which X is project) are scaled by the inverse of the square 
        root of the eigenvalues:
        scaled_eigvecs = eigvecs / sqrt_eigvals
    
    verbose : bool, optional (default is True)
        If True, print relevant information.
    
    Returns
    -------
    X_pca : ndarray of shape (n_samples, nu)
        Normalized data.
    
    scaled_eigvecs_inv : ndarray of shape (n_features, nu)
        Eigenvectors scaled by square root of eigenvalues (to be used for 
        denormalization if 'scale_evecs' is True).
    
    scaled_eigvecs: ndarray of shape (n_features, nu)
        Eigenvectors scaled by the inverse of the square root of the 
        eigenvalues. Used for projection of X if 'scale_evecs' is True.
    
    eigvecs: ndarray of shape (n_features, nu)
        Unscaled eigenvectors. Used for projection of X if 'scale_evecs' is 
        False.
    
    means : ndarray of shape (n_features, )
        Means of the individual features.
    
    eigvals : ndarray of shape (nu, )
        Eigenvalues of the covariance matrix of the data set.
    
    eigvals_trunc : ndarray of shape (n_features, )
        Top nu eigenvalues of the covariance matrix of the data set.
    
    """
    if verbose:
        print("\n\nPerforming PCA.")
        print("---------------")
        print("Input data dimensions:", X.shape)
    N, n = X.shape
    means = np.mean(X, axis=0)
    X = X - means
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if verbose:
        print("PCA eigenvalues:", eigvals)
    
    if method=='eigv_cutoff':
        if verbose:
            print(f"Using specified cutoff value = {eigenvalues_cutoff} "
                  "for truncation.")
            print(f"Dropping eigenvalues less than {eigenvalues_cutoff}")
        eigvals_trunc = eigvals[eigvals > eigenvalues_cutoff]
        if verbose:
            print("PCA retained eigenvalues:", eigvals_trunc)
    elif method=='cum_energy':
        if verbose:
            print("Criteria for truncation: cumulative energy content = "
                  f"{cumulative_energy}")
        tot_eigvals = np.sum(eigvals)
        for i in range(len(eigvals)+1):
            if np.sum(eigvals[0:i])/tot_eigvals > (1-cumulative_energy):
                eigvals_trunc = eigvals[i-1:]
                break
        if verbose:
            print("PCA retained eigenvalues:", eigvals_trunc)
            print("PCA cumulative energy content =",
                  np.sum(eigvals_trunc)/tot_eigvals)
    elif method=='pca_dim':
        if verbose:
            print(f"Using specified PCA dimension = {pca_dim}")
        eigvals_trunc = eigvals[-pca_dim:]
        if verbose:
            print("PCA retained eigenvalues:", eigvals_trunc)

    if verbose:
        print("Number of features:", n, "->", len(eigvals_trunc))
    num_dropped_features = n - len(eigvals_trunc)
    eigvecs = eigvecs[:, num_dropped_features:]
    sqrt_eigvals = np.sqrt(eigvals_trunc)
    scaled_eigvecs = eigvecs / sqrt_eigvals
    scaled_eigvecs_inv = eigvecs * sqrt_eigvals
    
    # compute X_pca as wide matrix (then transpose); faster in later comps
    if scale_evecs:
        X_pca = np.dot(scaled_eigvecs.T, X.T).T
    else:
        X_pca = np.dot(eigvecs.T, X.T).T
    
    # compute X_pca as tall matrix; slower in later comps
    # X_pca = np.dot(X, scaled_eigvecs)
    
    if verbose:
        print("Output data dimensions:", X_pca.shape)
    return (X_pca, scaled_eigvecs_inv, scaled_eigvecs, eigvecs, means, eigvals, 
            eigvals_trunc)

###############################################################################
def pca(plom_dict):
    """
    PLoM wrapper for normalization (PCA) function.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    scaling = plom_dict['options']['scaling']
    if scaling:
        X = plom_dict['scaling']['training']
    else:
        X = plom_dict['data']['training']
    
    method      = plom_dict['options']['pca_method']
    scale_evecs = plom_dict['options']['pca_scale_evecs']
    verbose     = plom_dict['options']['verbose']
    
    cumulative_energy  = plom_dict['input']['pca_cum_energy']
    eigenvalues_cutoff = plom_dict['input']['pca_eigv_cutoff']
    pca_dim            = plom_dict['input']['pca_dim']
    
    (X_pca, scaled_evecs_inv, scaled_evecs, evecs, 
    means, evals, evals_trunc) = _pca(X, method, cumulative_energy, 
                                      eigenvalues_cutoff, pca_dim, scale_evecs,
                                      verbose=verbose)
    
    plom_dict['pca']['training']         = X_pca
    plom_dict['pca']['scaled_evecs_inv'] = scaled_evecs_inv
    plom_dict['pca']['scaled_evecs']     = scaled_evecs
    plom_dict['pca']['evecs']            = evecs
    plom_dict['pca']['mean']             = means
    plom_dict['pca']['eigvals']          = evals
    plom_dict['pca']['eigvals_trunc']    = evals_trunc

###############################################################################
def _inverse_pca(X, eigvecs, means):
    """
    Project data set X from PCA space back to original data space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, nu)
        Data set in PCA space (nu-dimensional) to be projected back to original
        data space (n_features-dimensional).

    eigvecs : ndarray of shape (n_features, nu)
        Eigenvectors (principal components) onto which original data X was 
        projected.
    
    means : ndarray of shape (n_features, )
        Means of the individual features.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Data set projected back to original n_features-dimensional space.

    """
    X = np.dot(X, eigvecs.T) + means    
    return X

###############################################################################
def inverse_pca(plom_dict):
    """
    PLoM wrapper for inverse normalization (inverse PCA) function.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    scale_evecs = plom_dict['options']['pca_scale_evecs']
    if scale_evecs:
        eigvecs = plom_dict['pca']['scaled_evecs_inv']
    else:
        eigvecs = plom_dict['pca']['evecs']
        
    mean    = plom_dict['pca']['mean']
    
    scaling = plom_dict['options']['scaling']
    
    if plom_dict['pca']['reconst_training'] is not None:
        reconst_training = plom_dict['pca']['reconst_training']
    else:
        reconst_training = plom_dict['pca']['training']
    
    augmented = plom_dict['pca']['augmented']
    
    if reconst_training is not None:
        X = _inverse_pca(reconst_training, eigvecs, mean)
        if scaling:
            plom_dict['scaling']['reconst_training'] = X
        else:
            plom_dict['data']['reconst_training'] = X
    
    if augmented is not None:
        X = _inverse_pca(augmented, eigvecs, mean)
        if scaling:
            plom_dict['scaling']['augmented'] = X
        else:
            plom_dict['data']['augmented'] = X

###############################################################################
def _sample_projection(H, g):
    """
    Reduce normalized data (n_samples x nu) to random matrix [Z] (nu x m) 
    using reduced DMAPS basis (n_samples x m).
    Find Z such that: [H] = [Z] [g]^T
    where: [H]: normalized data (nu x n_samples)
           [Z]: projected sample (nu x m)
           [g]: DMAPS basis (n_samples x m)
    => [Z] = [H] [a] where [a] = [g] ([g]^T [g])^-1

    Parameters
    ----------
    H : ndarray of shape (nu, n_samples)
        Data set in PCA or original space. This matrix is fed into the DMAPS 
        machinery to find a reduced basis [g].
    
    g : ndarray of shape (n_samples, m)
        Reduced DMAPS basis (m eigenvectors).

    Returns
    -------
    Z : ndarray of shape (nu, m)
        Reduced normalized data, random matrix [Z].
    
    a : ndarray of shape (n_samples, m)
        Reduction matrix [a].

    """
    a = np.dot(g, np.linalg.inv(np.dot(np.transpose(g), g)))
    if H.shape[1] != a.shape[0]:
        H = H.T
    Z = np.dot(H, a)
    return Z, a

###############################################################################
def sample_projection(plom_dict):
    """
    PLoM wrapper for sample projection ([H] = [Z] [g]^T) function.
    Find [Z] for a given [H].

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """    
    projection_source = plom_dict['options']['projection_source']
    projection_target = plom_dict['options']['projection_target']
    verbose = plom_dict['options']['verbose']
    
    if projection_source == "pca":
        H = plom_dict['pca']['training']
    elif projection_source == "scaling":
        H = plom_dict['scaling']['training']
    elif projection_target == "data":
        H = plom_dict['data']['training']
    
    if projection_target == "dmaps":
        g = plom_dict['dmaps']['reduced_basis']
    elif projection_target == "pca":
        g = plom_dict['pca']['training']
    elif projection_target == "scaling":
        g = plom_dict['scaling']['training']
    elif projection_target == "data":
        H = plom_dict['data']['training']
    
    if H is None:
        if verbose:
            print("Projection source [H] not found. Skipping projection.")
    elif g is None:
        if verbose:
            print("Projection target [g] not found. Skipping projection.")
    else:
        Z0, a = _sample_projection(H, g)
        
        plom_dict['ito']['Z0'] = Z0
        plom_dict['ito']['a']  = a
    
###############################################################################
def _inverse_sample_projection(Z, g):
    """
    Reverse the sample projection procedure.
    Map a random matrix [Z] back to the original space using the DMAPS reduced 
    basis.
    Return new matrix [H] = [Z] [g]^T

    Parameters
    ----------
    Z : ndarray of shape (nu, m)
        Reduced normalized data, random matrix [Z].
    
    g : ndarray of shape (n_samples, m)
        Reduced DMAPS basis (m eigenvectors).

    Returns
    -------
    H : ndarray of shape (n_samples, nu)
        Data set in PCA or original space. This matrix is fed into the DMAPS 
        machinery to find a reduced basis [g]. In case of new projected sample 
        [Z], [H] is a new sample in PCA or original space.
        
    """
    # The following line leads to slower computations later on.
    # H = np.dot(Z, g.T).T
    
    # This is faster.
    H = np.dot(g, Z.T)
    return H

###############################################################################
def inverse_sample_projection(plom_dict):
    """
    PLoM wrapper for inverse sample projection ([H] = [Z] [g]^T) function.
    Find [H] for a given [Z].

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """ 
    projection_source = plom_dict['options']['projection_source']
    projection_target = plom_dict['options']['projection_target']
    
    if projection_target == "dmaps":
        g = plom_dict['dmaps']['reduced_basis']
    elif projection_target == "pca":
        g = plom_dict['pca']['training']
    
    if g is None:
        return
    
    Zs = plom_dict['ito']['Zs']
    X_augmented = None
    if Zs is not None:
        X_augmented = []
        for Z_final in Zs:
            X_augmented.append(_inverse_sample_projection(Z_final, g))
        X_augmented = np.vstack(X_augmented)
    
    Z0 = plom_dict['ito']['Z0']
    X_reconst = _inverse_sample_projection(Z0, g)
    
    if projection_source == "pca":
        plom_dict['pca']['augmented'] = X_augmented
        plom_dict['pca']['reconst_training'] = X_reconst
    elif projection_source == "scaling":
        plom_dict['scaling']['augmented'] = X_augmented
        plom_dict['scaling']['reconst_training'] = X_reconst
    else:
        plom_dict['data']['augmented'] = X_augmented
        plom_dict['data']['reconst_training'] = X_reconst
        
###############################################################################
def _get_dmaps_basis(H, epsilon, kappa=1, diffusion_dist_method='standard'):
    """
    Return DMAPS basis.
    Construct diffusion-maps basis, [g], using specified kernel width, epsilon.

    Parameters
    ----------
    H : ndarray of shape (n_samples, nu)
        Normalized data set for which DMAPS basis is constructed.
    
    epsilon : float
        Diffusion-maps kernel width (smoothing parameter, > 0).
    
    kappa : int, optional (default is 1)
        Related to the analysis scale of the local geometric structure of the 
        dataset.
    
    diffusion_dist_method : string, optional (default is 'standard')
        Experimental. Always use 'standard'.
        If 'standard', compute pair-wise distances using standard L2 norm.
        If 'periodic', compute pair-wise distances using periodic norm.

    Returns
    -------
    basis : ndarray of shape (n_samples, n_samples)
        Diffusion-maps basis, [g].
    
    values : ndarray of shape (n_samples, )
        Diffusion-maps eigenvalues.
    
    vectors : ndarray of shape (n_samples, n_samples)
        Diffusion-maps eigenvectors.

    """    
    if diffusion_dist_method == 'standard':
        distances = np.array([np.sum(np.abs(H - a)**2, axis=1) for a in H])

    elif diffusion_dist_method == 'periodic':
        sh = np.shape(H)
        max_th_val = np.max(H[:,0])
        if sh[1] == 2:
            z_dist = distance_matrix(H[:,1].reshape(sh[0],1), 
                                     H[:,1].reshape(sh[0],1))
        else:
            z_dist = 0
        th_dist = distance_matrix(H[:,0].reshape(sh[0],1), 
                                  H[:,0].reshape(sh[0],1))
        th_dist = np.mod(th_dist+max_th_val,2.0*
                         max_th_val+(2.0*max_th_val/99.0))-max_th_val
        # th_dist[th_dist>(max_th_val)] = (2*max_th_val - 
                                         # th_dist[th_dist>(max_th_val)])
        distances = z_dist**2 + th_dist**2

    diffusions = np.exp(-distances / (epsilon))
    scales = np.sum(diffusions, axis=0)**.5
    P = np.linalg.inv(np.diag(scales**2)).dot(diffusions)
    
    # Note: eigenvectors of transition matrix are the same for any power kappa
    normalized = diffusions / (scales[:, None] * scales[None, :])
    values, vectors = np.linalg.eigh(normalized)
    # values, vectors = np.linalg.eigh(
        # np.linalg.matrix_power(normalized, kappa))
    basis_vectors = vectors / scales[:, None]
    basis = basis_vectors * values[None, :]**kappa
    return np.flip(basis,axis=1), np.flip(values), np.flip(vectors,axis=1)

###############################################################################
def _get_dmaps_optimal_dimension(eigvalues, L):
    """
    Estimate reduced manifold dimension, m, using a scale separation cutoff for
    determining where to truncate the spectral decomposition based on the 
    (sorted-decreasing eigenvalues), i.e., if eigenvalue j+1 is less than 
    eigenvalue 2 by more than this factor, then truncate and use eigenvalues 2 
    through j.
    
    Parameters
    ----------
    eigvals :ndarray of shape (n_samples, )
        DMAPS eigenvalues.
    
    L : float
        DMAPS eigenvalues scale separation cutoff value.
    
    Returns
    -------
    m : int
        Manifold dimension, m.
    
    """
    m = len(eigvalues) - 1
    for a in range(2, len(eigvalues)):
        r = eigvalues[a] / eigvalues[1]
        if r < L:
            m = a - 1
            break
    return m

###############################################################################
def _get_dmaps_dim_from_epsilon(H, epsilon, kappa, L, dist_method='standard'):
    """
    Return manifold dimension, m, given epsilon.
    For the given epsilon, compute the DMAPS basis and eigenvalues, and choose 
    the manifold dimension based on these eigenvalues (with cutoff criteria L).

    Parameters
    ----------
    H : ndarray of shape (n_samples, nu)
        Normalized data set.
    
    epsilon : float
        Diffusion-maps kernel width (smoothing parameter, > 0).
    
    kappa : int, optional (default is 1)
        Related to the analysis scale of the local geometric structure of the 
        dataset.
    
    L : float
        DMAPS eigenvalues scale separation cutoff value.
    
    dist_method : string, optional (default is 'standard')
        Experimental. Always use 'standard'.
        If 'standard', compute pair-wise distances using standard L2 norm.
        If 'periodic', compute pair-wise distances using periodic norm.

    Returns
    -------
    m : int
        Manifold dimension.

    """
    basis, eigvals, eigvecs = _get_dmaps_basis(H, epsilon, kappa, dist_method)
    m = _get_dmaps_optimal_dimension(eigvals, L)
    return m

###############################################################################
def _get_dmaps_optimal_epsilon(H, kappa, L, dist_method='standard'):
    """
    Used when epsilon is not specified by user (epsilon='auto').
    Estimate optimal DMAPS kernel width, epsilon.
    Criteria for estimation: choose smallest epsilon that results in smallest 
    manifold dimension, m.
    After estimating epsilon, construct DMAPS basis [g] for  estimated epsilon
    and return epsilon, DMAPS basis, DMAPS eigenvalues, and manifold dimension.

    Parameters
    ----------
    H : ndarray of shape (n_samples, nu)
        Normalized data.
    
    kappa : int
        Related to the analysis scale of the local geometric structure of the 
        dataset.
    
    L : float
        DMAPS eigenvalues scale separation cutoff value.
    
    dist_method : string, optional (default is 'standard')
        Experimental. Always use 'standard'.
        If 'standard', compute pair-wise distances using standard L2 norm.
        If 'periodic', compute pair-wise distances using periodic norm.

    Returns
    -------
    epsilon : float
        Optimal epsilon. This is the smallest epsilon that results in the 
        smallest manifold dimension satisfying the DMAPS eigenvalue cutoff 
        criteria (L).
    
    m_target : int
        Target manifold dimension. This is usally the smallest possible 
        dimension satisfying the DMAPS eigenvalue cutoff criteria (L).
    
    eps_vs_m : ndarray of shape (?, 2)
        Matrix of Epsilon (1st column) vs manifold dimension (2nd column) used 
        when finding optimal epsilon.

    """
    epsilon_list = [0.1, 1, 2, 8, 16, 32, 64, 100, 10000]
    eps_for_m_target = [1, 10, 100, 1000, 10000]
    eps_vs_m = []
    m_target_list = [_get_dmaps_dim_from_epsilon(H, eps, kappa, L, 
                                                 dist_method) 
                     for eps in eps_for_m_target]
    m_target = min(m_target_list)
    upper_bound = eps_for_m_target[np.argmin(m_target_list)]
    lower_bound = epsilon_list[0]
    for eps in epsilon_list[1:]:
        m = _get_dmaps_dim_from_epsilon(H, eps, kappa, L)
        eps_vs_m.append([eps, m])
        if m > m_target:
            lower_bound = eps
        else:
            upper_bound = eps
            break
    while upper_bound - lower_bound > 0.5:
        middle_bound = (lower_bound+upper_bound)/2
        m = _get_dmaps_dim_from_epsilon(H, middle_bound, kappa, L)
        eps_vs_m.append([middle_bound, m])
        if m > m_target:
            lower_bound = middle_bound
        else:
            upper_bound = middle_bound
    m = _get_dmaps_dim_from_epsilon(H, lower_bound, kappa, L)
    while m > m_target:
        lower_bound += 0.1
        m = _get_dmaps_dim_from_epsilon(H, lower_bound, kappa, L)
        eps_vs_m.append([lower_bound, m])
    epsilon = lower_bound
    eps_vs_m = np.unique(eps_vs_m, axis=0)
    return epsilon, m_target, eps_vs_m

###############################################################################
def _dmaps(X, epsilon, kappa=1, L=0.1, first_evec=False, m_override=0,
           dist_method='standard', verbose=True):
    """
    Perform Diffusion-maps analysis on input data set.
    Given data set X, this function performs DMAPS on X using either an 
    optimal value of the DMAPS kernel bandwidth or a specified value.
    This function returns the DMAPS full basis, truncated basis, eigenvalues, 
    manifold dimension, and the epsilon used for the analysis.

    Parameters
    ----------
    X : ndarray of shape (n_samples, nu)
        Data set on which DMAPS analysis is performed. This is usually the 
        normalized data set (PCA).
    
    epsilon : float
        Diffusion-maps kernel width (smoothing parameter, > 0).
    
    kappa : int, optional (default is 1)
        Related to the analysis scale of the local geometric structure of the 
        dataset.
    
    L : float, optional (default is 0.1)
        DMAPS eigenvalues scale separation cutoff value.
    
    first_evec : bool, optional (default is False)
        If True, the first DMAPS eigenvector (constant, usally dropped) is 
        included in the DMAPS reduced basis.
    
    m_override : int, optional (default is 0)
        If greater than 0, this overrides the calculated dimension of the 
        manifold, and sets the dimension equal to 'm_override'.
    
    dist_method : string, optional (default is 'standard')
        Experimental. Always use 'standard'.
        If 'standard', compute pair-wise distances using standard L2 norm.
        If 'periodic', compute pair-wise distances using periodic norm.
    
    verbose : bool, optional (default is True)
        If True, print relevant information.

    Returns
    -------
    red_basis : ndarray of shape (n_samples, m)
        Reduced DMAPS basis (m eigenvectors).
        
    basis : ndarray of shape (n_samples, n_samples)
        Full DMAPS basis (n_samples eigenvectors).
        
    epsilon : float
        Diffusion-maps kernel width (specified or computed as optimal value) 
        used when computing the DMAPS basis.
        
    m : int
        Reduced DMAPS basis dimension
        
    eigvals : ndarray of shape (n_samples, )
        DMAPS eigenvalues.
        
    eigvecs : ndarray of shape (n_samples, n_samples)
        DMAPS eigenvectors.
        
        
    eps_vs_m : ndarray of shape (?, 2)
        Matrix of Epsilon (1st column) vs manifold dimension (2nd column) used 
        when finding optimal epsilon.
    
    """
    start_time = datetime.now()
    if verbose:
        print("\n\nPerforming DMAPS analysis.")
        print("--------------------------")
        print("Input data dimensions:", X.shape)

    if epsilon == 'auto':
        if verbose:
            print("Finding best epsilon for analysis.")
        epsilon, m_opt, eps_vs_m = _get_dmaps_optimal_epsilon(X, kappa, L, 
                                                              dist_method)
        basis, eigvals, eigvecs = _get_dmaps_basis(X, epsilon, kappa, 
                                                   dist_method)
        if m_override > 0:
            m = m_override
        else:
            m = m_opt
        if verbose:
            print("Epsilon = %.2f" %epsilon)
            print(f"Manifold eigenvalues: {str(eigvals[1:m+2])[1:-1]} [...]")
            print(f"Manifold dimension: m optimal = {m_opt}")
            if m_override>0:
                print("Overriding manifold dimension.")
            print(f"m used = {m}")
    
    else:
        eps_vs_m = []
        epsilon = np.atleast_1d(epsilon)
        if len(epsilon)==1 and verbose:
            print("Using specified epsilon for analysis.")
        elif verbose:
            print("Using specified epsilon list for analysis.")
        for eps in epsilon:
            basis, eigvals, eigvecs = _get_dmaps_basis(X, eps, kappa, 
                                                       dist_method)
            m_opt = _get_dmaps_optimal_dimension(eigvals, L)
            if m_override > 0:
                m = m_override
            else:
                m = m_opt
            eps_vs_m.append([eps, m_opt])
            if verbose:
                if len(epsilon)>1:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Epsilon = %.2f" %eps)
                print(f"Manifold eigenvalues: {str(eigvals[1:m+2])[1:-1]} "
                      "[...]")
                print(f"Manifold dimension: m optimal = {m_opt}")
                if m_override>0:
                    print("Overriding manifold dimension.")
                print(f"m used = {m}")
        if verbose:
            if len(epsilon)>1:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Using last epsilon in specified list.")
                print("Epsilon = %.2f" %eps)
                print(f"Manifold eigenvalues: {str(eigvals[1:m+2])[1:-1]} " 
                      "[...]")
                print(f"Manifold dimension: m optimal = {m_opt}")
                if m_override>0:
                    print("Overriding manifold dimension.")
                print(f"m used = {m}")
        epsilon =  epsilon[-1]
        eps_vs_m = np.unique(eps_vs_m, axis=0)

        
    if first_evec: # indices of first and last eigenvectors to be used
        if verbose:
            print("Including first (trivial) eigenvector in projection basis.")
        s = 0
        if m_override==0:
            m = m+1
        e = m
    else:
        s = 1
        e = m+1
    red_basis = basis[:, s:e]
    
    # eps_vs_m.sort() # not needed since np.unique is used a few lines earlier
    
    end_time = datetime.now()
    if verbose:
        print(f'Using {e-s} DMAPS eigenvectors ({s} to {e-1}).')
        print(f"DMAPS data dimensions: {red_basis.shape}")
        print(f"*** DMAPS time = {str(end_time-start_time)[:-3]} ***")
    return red_basis, basis, epsilon, m, eigvals, eigvecs, eps_vs_m

###############################################################################
def dmaps(plom_dict):
    """
    PLoM wrapper for dmaps function.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    
    epsilon = plom_dict['input']['dmaps_epsilon']
    kappa   = plom_dict['input']['dmaps_kappa']
    L       = plom_dict['input']['dmaps_L']
    first_evec  = plom_dict['options']['dmap_first_evec']
    m_override  = plom_dict['options']['dmaps_m_override']
    dist_method = plom_dict['options']['dmaps_dist_method']
    verbose     = plom_dict['options']['verbose']
    
    if plom_dict['pca']['training'] is not None:
        X = plom_dict['pca']['training']
    elif plom_dict['scaling']['training'] is not None:
        X = plom_dict['scaling']['training']
    else:
        X = plom_dict['data']['training']
    
    red_basis, basis, epsilon, m, eigvals, eigvecs, eps_vs_m = \
        _dmaps(X, epsilon, kappa, L, first_evec, m_override, dist_method, 
               verbose)
    
    plom_dict['dmaps']['eigenvectors']  = eigvecs
    plom_dict['dmaps']['eigenvalues']   = eigvals
    plom_dict['dmaps']['dimension']     = m
    plom_dict['dmaps']['epsilon']       = epsilon
    plom_dict['dmaps']['basis']         = basis
    plom_dict['dmaps']['reduced_basis'] = red_basis
    plom_dict['dmaps']['training']      = red_basis
    plom_dict['dmaps']['eps_vs_m']      = eps_vs_m

###############################################################################
def _get_L(H, u, kde_bw_factor=1, method=2):
    """
    Compute the gradient of the potential to be used in each ito step.
    
    Parameters
    ----------
    H : ndarray of shape (nu, n_samples)
        Typically, this is the normalized (PCA) data set. If the 
        non-normalized data set is used, this would be of shape 
        (n_features, n_samples).
        
    u :  ndarray of shape (nu, n_samples) 
        Product of intermediate matrix [zHalf] and transpose of reduced DMAPS 
        basis [g].
    
    kde_bw_factor : float, optional (default is 1.0)
        Multiplier that modifies the computed KDE bandwidth (Silverman 
        rule-of-thumb).
    
    method : int, optional (default is 2)
        Experimental.
        This is the most expensive part of the computation. 
        Methods 1-8 compute the same joint KDE using different algorithms.
        Method 2 is the most efficient.
        Method 9 computes a conditional (Nadaraya-Watson KDE) * marginal tanh 
        approximation. This is used when the distribution of one of the 
        variables is to be specified.
        Method 10 a computes conditional (Nadaraya-Watson KDE) * marginal KDE.
            
    Returns
    -------
    pot : ndarray of shape (nu, n_samples)
        Gradient of the potential for the given u. 
    
    """
    if method in [1, 2]: # c++ library
        nu, N = H.shape
        H_flat = H.flatten().astype(np.float64)
        u_flat = u.flatten().astype(np.float64)
        pot = np.zeros((nu, N), dtype=np.float64)

        # Call the C++ function
        get_L_cpp(
            H_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nu,
            N
        )
    
    
    elif method == 3: # default Python implementation
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s
        
        dist_mat_list = [(scaled_H.T - x).T for x in u.T]
        
        norms_list = np.exp((-1/(2*shat**2)) * np.array(list(map(\
            lambda x: np.linalg.norm(x, axis=0)**2, dist_mat_list))))
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
    
    
    elif method == 4:
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s
        
        dist_mat_list = np.asfortranarray(np.tile(scaled_H.T, (N, 1)) - 
                        np.repeat(u.T, N, axis=0))
        norms_list = np.exp((-1/(2*shat**2)) * 
                     np.linalg.norm(dist_mat_list, axis=1)**2)
        q_list = np.sum(np.reshape(norms_list, (N, N)), axis=1) / N
        product = dist_mat_list * norms_list[:, None] / shat**2 / N
        dq_list = np.sum(np.reshape(product, (N, N, -1)), axis=1)
        pot = (dq_list / q_list[:,None]).T
    
    
    elif method==5:
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s
        
        dist_mat_list = list(map(lambda x: (scaled_H.T - x).T, u.T))
        
        norms_list = np.exp((-1/(2*shat**2)) * np.array(list(map(\
            lambda x: np.linalg.norm(x, axis=0)**2, dist_mat_list))))
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
    
    
    elif method == 6:    
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s

        dist_mat_list = list(map(lambda x: (scaled_H.T - x).T, u.T))
        
        norms_list = np.exp((-1/(2*shat**2)) * \
            (distance_matrix(scaled_H.T, u.T)**2).T) # N x N
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
    
    
    elif method == 7:    
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s

        dist_mat_list = [(scaled_H.T - x).T for x in u.T]
        
        norms_list = np.exp((-1/(2*shat**2)) * \
            (distance_matrix(scaled_H.T, u.T)**2).T) # N x N
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
    
    
    elif method == 8:    
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s

        dist_mat_list = list(map(lambda x: (scaled_H.T - x).T, u.T))
        
        norms_list = np.exp(
            (-1/(2*shat**2)) * 
            np.array([np.linalg.norm(x, axis=0)**2 for x in dist_mat_list]))
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
    
    
    elif method == 9:    
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s

        dist_mat_list = [(scaled_H.T - x).T for x in u.T]
        
        norms_list = np.exp(
            (-1/(2*shat**2)) * 
            np.array([np.linalg.norm(x, axis=0)**2 for x in dist_mat_list]))
        
        q_list = np.array(list(map(np.sum, norms_list))) / N
        
        product = np.array(list(map(np.dot, dist_mat_list, norms_list)))
        
        dq_list = product/shat**2/N
        pot = (dq_list/q_list[:,None]).transpose()
            
    
    elif method == 10:
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s
        
        raw_dist = np.array([scaled_H.T - x for x in u.T])

        exp_dist = np.exp((-1/(2*shat**2))*np.sum(raw_dist**2, axis=2))

        q = 1/N * np.sum(exp_dist, axis=1)

        dq = np.array([np.sum(raw_dist[:,:,i]*exp_dist, axis=1) 
                        for i in range(nu)]) / shat**2 / N
        pot = dq/q

    
    elif method == 11:
        nu, N = H.shape
        s = (4 / (N*(2+nu))) ** (1/(nu+4))*kde_bw_factor
        shat = s / np.sqrt(s**2 + (N-1)/N)
        scaled_H = H * shat / s
        
        raw_dist = [scaled_H.T - x for x in u.T]

        exp_dist = np.exp((-1/(2*shat**2))*np.linalg.norm(raw_dist, axis=2)**2)

        q = 1/N * np.sum(exp_dist, axis=1)

        dq = [np.sum(np.array(raw_dist)[:,:,i]*exp_dist, axis=1) 
                        for i in range(nu)]/shat**2/N
        pot = dq/q
    
    
    elif method == 12: # conditional (Nadaraya-Watson KDE) * marginal tanh approx.
        H = H.T
        u = u.T
        nu, N = 1, H.shape[0]
        eta_th = H[:,0]
        eta_z = H[:,1]
        u_th = u[:,0]
        u_z = u[:,1]
        hz = (4 / (N*(2+nu))) ** (1/(nu+4)) * kde_bw_factor
        ht = (4 / (N*(2+nu))) ** (1/(nu+4)) * kde_bw_factor
        a = min(eta_th)
        b = max(eta_th)
        dd = 100
        
        th_raw_dist = np.subtract.outer(u_th, eta_th)
        # numerator of w_i (each row in the matrix corresponds to one theta_l)
        th_dist = np.exp((-1/(2*ht**2)) * (th_raw_dist**2))
        # denom. of w_i (each number in the list corresponds to one theta_l)
        th_dist_tot = np.array(list(map(np.sum,th_dist)))
        q_th = (1/2/(b-a)) * (np.tanh(dd*(u_th-a)) - 
                              np.tanh(dd*(u_th-b))).reshape(N) # q(theta) (N,)
        
        z_raw_dist = np.subtract.outer(u_z, eta_z)
        z_dist = np.exp((-1/(2*hz**2)) * (z_raw_dist**2))
        q_z = np.sum((1/np.sqrt(2*np.pi)/hz * z_dist * th_dist),
                      axis=1) / th_dist_tot # q(z|theta) (N,)
    
        q = q_z * q_th
        
        dq_dz = np.sum((-1/np.sqrt(2*np.pi)/hz**3) * z_raw_dist * z_dist * 
                        th_dist, axis=1) / th_dist_tot * q_th
        
        dq_dt_1 = q_th
        dq_dt_2 = q_z
        dq_dt_3 = (dd/2/(b-a) / (np.cosh(dd*(u_th-a))**2) - 
                    dd/2/(b-a) / (np.cosh(dd*(u_th-b))**2))
        dq_dt_4 = (-1/np.sqrt(2*np.pi)/hz/ht**2 * 
                    np.sum(z_dist * th_raw_dist * th_dist, axis=1) * 
                    th_dist_tot - 
                    (1/np.sqrt(2*np.pi)/hz) * np.sum(z_dist*th_dist, axis=1) * 
                    (-1/ht**2)*np.sum(th_raw_dist*th_dist, axis=1)
                    ) / th_dist_tot**2
        dq_dt = dq_dt_4 * dq_dt_1 + dq_dt_2 * dq_dt_3    
       
        dq = np.array((dq_dt, dq_dz))
        pot = dq/(q)

    
    elif method == 13: # conditional (Nadaraya-Watson KDE) * marginal KDE
        H = H.T
        u = u.T
        nu, N = 1, H.shape[0]
        eta_th = H[:,0]
        eta_z = H[:,1]
        u_th = u[:,0]
        u_z = u[:,1]
        hz = (4 / (N*(2+nu))) ** (1/(nu+4)) * kde_bw_factor
        ht = (4 / (N*(2+nu))) ** (1/(nu+4)) * kde_bw_factor
        
        th_raw_dist = np.subtract.outer(u_th, eta_th)
        # numerator of w_i (each row in the matrix corresponds to one theta_l)
        th_dist = np.exp((-1/(2*ht**2)) * (th_raw_dist**2))
        # denom. of w_i (each number in the list corresponds to one theta_l)
        th_dist_tot = np.array(list(map(np.sum,th_dist)))
        # q(theta) (N,)
        q_th = (1/N/ht) * np.sum((1/np.sqrt(2*np.pi) * th_dist), axis=1)
        
        z_raw_dist = np.subtract.outer(u_z, eta_z)
        z_dist = np.exp((-1/(2*hz**2)) * (z_raw_dist**2))
        # q(z|theta) (N,)
        q_z  = np.sum((1/np.sqrt(2*np.pi)/hz * z_dist * th_dist), 
                      axis=1) / th_dist_tot
    
        q = q_z * q_th
        
        dq_dz = np.sum((-1/np.sqrt(2*np.pi)/hz**3) * z_raw_dist * z_dist * 
                        th_dist, axis=1) / th_dist_tot * q_th
        
        dq_dt_1 = q_th
        dq_dt_2 = q_z
        dq_dt_3 = np.sum((-1/np.sqrt(2*np.pi)/ht**3/N) * th_raw_dist * th_dist,
                          axis=1)
        dq_dt_4 = (-1/np.sqrt(2*np.pi)/hz/ht**2 * 
                    np.sum(z_dist * th_raw_dist * th_dist, axis=1) * 
                    th_dist_tot - (1/np.sqrt(2*np.pi)/hz) * 
                    np.sum(z_dist*th_dist, axis=1) * (-1/ht**2) * 
                    np.sum(th_raw_dist*th_dist, axis=1)
                    ) / th_dist_tot**2
        dq_dt = dq_dt_4 * dq_dt_1 + dq_dt_2 * dq_dt_3    
        
        dq = np.array((dq_dt, dq_dz))
        pot = dq/(q)
        
    return pot

###############################################################################
def _initialize_potential_cpp(pot_method=1, verbose=True):

    if pot_method in ["cpp_eigen", "cpp"]:
        pot_method = 1
    elif pot_method == "cpp_native":
        pot_method = 2
    
    if pot_method == 1:
        lib_type1 = "eigen"
        lib_type2 = "native"
    elif pot_method == 2:
        lib_type1 = "native"
        lib_type2 = "eigen"
    else:
        return pot_method
    
    lib_ext = ".dll" if os.name == "nt" else ".so"
    
    for lib_type in [lib_type1, lib_type2]:
        lib_name = "potential_" + lib_type + lib_ext
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", lib_name)
        
        if verbose:
            print(f"\nAttempting to load C++ potential library: {lib_name}")
        
        try:
            # Load the compiled C++ library
            global potential_lib
            potential_lib = ctypes.CDLL(lib_path)

            # Define the function interface
            global get_L_cpp
            get_L_cpp = potential_lib.get_L
            get_L_cpp.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # H
                ctypes.POINTER(ctypes.c_double),  # u
                ctypes.POINTER(ctypes.c_double),  # pot (output)
                ctypes.c_int,  # nu
                ctypes.c_int   # N
            ]
            get_L_cpp.restype = None

            H_test = np.array([[1, 2, 3], [4, 5, 6]])
            u_test = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
            res_test = _get_L(H_test, u_test, method=1)
            res_true = np.array([ [0.41949073, -0.07669393, -1.82789796],
                                  [-0.41914386, -0.91532852, -2.66653255] ])
            np.testing.assert_allclose(res_true, res_test)
            if verbose:
                print(f"C++ potential library '{lib_name}' loaded successfully\n")
            return pot_method

        except Exception as e:
            if verbose:
                print(f"Failed to load '{lib_name}'")
                print(e)
    
    if verbose:
        print("\nPython implementation will be used instead\n")

    pot_method = 3

    return pot_method

###############################################################################
def _simulate_entire_ito(Z, H, basis, a, f0=1, dr=0.1, t='auto', n=1, 
                         parallel=False, n_jobs=-1, kde_bw_factor=1, 
                         pot_method=1, M0=0, verbose=True):
    """
    Evolve the ISDE for 't' steps 'n' times. Obtain 'n' new samples at the end.
    If t is 'auto', compute required number of steps.
    
    Parameters
    ----------
    Z : ndarray of shape (nu, m)
        Reduced random matrix [Z] before evolution of ISDE.

    H : ndarray of shape (nu, n_samples)
        Typically, this is the normalized (PCA) data set. If the 
        non-normalized data set is used, this would be of shape 
        (n_features, n_samples).

    basis : ndarray of shape (n_samples, m)
        Reduced DMAPS basis.

    a : ndarray of shape (n_samples, m)
        Reduction matrix [a].

    f0 : float, optional (default is 1.0)
        Parameter that allows the dissipation term of the nonlinear 
        second-order dynamical system (dissipative Hamiltonian system) to be 
        controlled (damping that kills the transient response).

    dr : float, optional (default is 0.1)
        Sampling step of the continuous index parameter used in the 
        integration scheme.

    t : int/str, optional (default is 'auto')
        Number of steps that the Ito stochastic differential equation is 
        evolved for.
        If 'auto', number of steps is calculated internally.

    n : int, optional (default is 1)
        Number of times that we sample the manifold. The actual number of 
        data points generated is equal to this 'n' times the original number 
        of points in the data set (n_samples).
    
    parallel : bool, optional (default is False)
        If True, run sampling in parallel.
    
    n_jobs : int, optional (default is -1)
        Number of jobs started by joblib.Parallel().
        Used if 'parallel' is True.
    
    kde_bw_factor : float, optional (default is 1.0)
        Multiplier that modifies the computed KDE bandwidth (Silverman 
        rule-of-thumb).
    
    pot_method : int, optional (default is 2)
        Experimental.
        This is the most expensive part of the computation. 
        Methods 1-8 compute the same joint KDE using different algorithms.
        Method 2 is the most efficient.
        Method 9 computes a conditional (Nadaraya-Watson KDE) * marginal tanh 
        approximation. This is used when the distribution of one of the 
        variables is to be specified.
        Method 10 a computes conditional (Nadaraya-Watson KDE) * marginal KDE.
    
    verbose : bool, optional (default is True)
        If True, print relevant information.
                
    Returns
    -------
    Zs : list of n ndarrays of shape (nu, m) each
         New reduced manifold samples matrix [Zs].
    
    Zs_steps : list n lists of t ndarrays of shape (nu, m)
        Ito steps for generated samples. FOR DEBUGGING.
    
    t : int
        Number of steps used in the Ito stochastic differential equation 
        evolution.
    
    """
    st = datetime.now()
    nu, N = H.shape
    s = (4 / (N*(2+nu))) ** (1/(nu+4)) * kde_bw_factor
    shat = s / np.sqrt(s**2 + (N-1)/N)
    fac = 2.0*np.pi*shat / dr
    if verbose: 
        print(f"From Ito sampler: fac = {fac:.3f}")
    steps = 4*np.log(100)/f0/dr
    if t == 'auto':
        t = int(steps+1)
    if verbose: 
        print(f"From Ito sampler: {steps:.1f} steps needed; {t:.0f} steps ", 
              "provided")
    Zs = []
    Zs_steps = []
    
    if pot_method in ["cpp_eigen", "cpp"]:
        pot_method = 1
    elif pot_method == "cpp_native":
        pot_method = 2
    elif pot_method == "python":
        pot_method = 3
    
    if pot_method in [1, 2]:
        pot_method = _initialize_potential_cpp(pot_method, verbose)

    # first sample at end of transient period if M0 > 0
    n_done = 0
    if M0 > 0:
        Zw, Z_steps = _simulate_ito_walk(Z,  t, H, basis, a, f0, dr, 
                                         kde_bw_factor, pot_method)
        Zs.append(Zw)
        Zs_steps.append(Z_steps)
        Z = Zw
        t = M0
        n_done = 1
        if verbose:
            print(f"Sample 1/{n} generated.")
    
    if parallel:
        if verbose:
            print(f'Generating {n - n_done} samples in parallel...')
            
        res = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_ito_walk)(
                Z, t, H, basis, a, f0, dr, kde_bw_factor, pot_method, parallel) 
            for i in range(n - n_done))
        [Zs_par, Zs_steps_par] = np.array(res, dtype=object).T
        Zs += Zs_par.tolist()
        Zs_steps += Zs_steps_par.tolist()
    else:
        for i in range(n_done, n):
            Zw, Z_steps = _simulate_ito_walk(Z,  t, H, basis, a, f0, dr, 
                                             kde_bw_factor, pot_method, parallel)
            if verbose:
                print("Sample %i/%i generated." %((i+1),n))
            Zs.append(Zw)
            Zs_steps.append(Z_steps)
    et = datetime.now()
    if verbose:
        print(f"*** Sampling time = {str(et-st)[:-3]} ***")
    return Zs, Zs_steps, t

###############################################################################
def _simulate_ito_walk(Z, t, H, basis, a, f0=1, dr=0.1, kde_bw_factor=1, 
                       pot_method=1, parallel=False):
    """
    Evolve one ISDE for 't' steps. Obtain one new sample at the end.

    Parameters
    ----------
    Z : ndarray of shape (nu, m)
        Reduced random matrix [Z] before evolution of ISDE.

    t : int
        Number of steps used in the Ito stochastic differential equation 
        evolution.

    H : ndarray of shape (nu, n_samples)
        Typically, this is the normalized (PCA) data set. If the 
        non-normalized data set is used, this would be of shape 
        (n_features, n_samples).

    basis : ndarray of shape (n_samples, m)
        Reduced DMAPS basis.

    a : ndarray of shape (n_samples, m)
        Reduction matrix [a].

    f0 : float, optional (default is 1.0)
        Parameter that allows the dissipation term of the nonlinear 
        second-order dynamical system (dissipative Hamiltonian system) to be 
        controlled (damping that kills the transient response).

    dr : float, optional (default is 0.1)
        Sampling step of the continuous index parameter used in the 
        integration scheme.

    kde_bw_factor : float, optional (default is 1.0)
        Multiplier that modifies the computed KDE bandwidth (Silverman 
        rule-of-thumb).

    pot_method : int, optional (default is 2)
        Experimental.
        This is the most expensive part of the computation. 
        Methods 1-8 compute the same joint KDE using different algorithms.
        Method 2 is the most efficient.
        Method 9 computes a conditional (Nadaraya-Watson KDE) * marginal tanh 
        approximation. This is used when the distribution of one of the 
        variables is to be specified.
        Method 10 a computes conditional (Nadaraya-Watson KDE) * marginal KDE.

    Returns
    -------
    Z : ndarray of shape (nu, m)
        New reduced manifold sample matrix [Z].

    steps : list of t ndarrays of shape (nu, m)
        Ito steps for generated sample [Z].

    """
    nu, N = H.shape
    Y = np.random.randn(nu, N).dot(a)
    steps = []
    
    if parallel and pot_method in [1, 2]:
        _ = _initialize_potential_cpp(pot_method, verbose=False)
    
    for j in range(0, t):
        Z, Y = _simulate_ito_step(Z, Y, H, basis, a, f0, dr, kde_bw_factor, 
                                  pot_method)
        # save ito steps
        steps.append(Z)
    return Z, steps

###############################################################################
def _simulate_ito_step(Z, Y, H, basis, a, f0, dr, kde_bw_factor, pot_method):
    """
    Evolve the ISDE for one step. Compute matrices [Z] and {Y] at end of step.
    
    Parameters
    ----------
    Z : ndarray of shape (nu, m)   
        Reduced random matrix [Z] evolved at start of current step.
    
    Y : ndarray of shape (nu, m)
        matrix [Y], evolved at start of current step.
    
    H : ndarray of shape (nu, n_samples)
        Typically, this is the normalized (PCA) data set. If the 
        non-normalized data set is used, this would be of shape 
        (n_features, n_samples).
    
    basis : ndarray of shape (n_samples, m)
        Reduced DMAPS basis.
    
    a : ndarray of shape (n_samples, m)
        Reduction matrix [a].
    
    f0 : float
        Parameter that allows the dissipation term of the nonlinear 
        second-order dynamical system (dissipative Hamiltonian system) to be 
        controlled (damping that kills the transient response).
    
    dr : float
        Sampling step of the continuous index parameter used in the 
        integration scheme.
    
    kde_bw_factor : float, optional (default is 1.0)
        Multiplier that modifies the computed KDE bandwidth (Silverman 
        rule-of-thumb).

    pot_method : int, optional (default is 2)
        Experimental.
        This is the most expensive part of the computation. 
        Methods 1-8 compute the same joint KDE using different algorithms.
        Method 2 is the most efficient.
        Method 9 computes a conditional (Nadaraya-Watson KDE) * marginal tanh 
        approximation. This is used when the distribution of one of the 
        variables is to be specified.
        Method 10 a computes conditional (Nadaraya-Watson KDE) * marginal KDE.
        
    Returns
    -------
    Znext : ndarray of shape (nu, m)
        Matrix [Z] evolved at end of current step.
    
    Ynext : ndarray of shape (nu, m)
        Matrix [Y] evolved at end of current step.
    
    """
    nu, N = H.shape
    b = f0*dr/4
    Weiner = dr**0.5 * np.random.randn(nu, N)
    dW = Weiner.dot(a)
    Zhalf = Z + (dr/2) * Y
    L = _get_L(H, np.dot(Zhalf, np.transpose(basis)), kde_bw_factor, 
               pot_method).dot(a)
    Ynext = (1-b)/(1+b) * Y + dr/(1+b) * L + np.sqrt(f0)/(1+b) * dW
    Znext = Zhalf + (dr/2) * Ynext    
    return Znext, Ynext

###############################################################################
def _sampling(Z0, H, basis, a, f0=1, dr=0.1, t='auto', num_samples=1, 
              parallel=False, n_jobs=-1, kde_bw_factor=1, pot_method=1, 
              verbose=True):
    """
    Calls the sampling function '_simulate_entire_ito' which evolves the ISDE 
    for 't' steps 'n' times. Obtain 'n' new samples at the end.
    If t is 'auto', compute required number of steps.
    
    Parameters
    ----------
    Z0 : ndarray of shape (nu, m)
        Reduced random matrix [Z] before evolution of ISDE.

    H : ndarray of shape (nu, n_samples)
        Typically, this is the normalized (PCA) data set. If the 
        non-normalized data set is used, this would be of shape 
        (n_features, n_samples).

    basis : ndarray of shape (n_samples, m)
        Reduced DMAPS basis.

    a : ndarray of shape (n_samples, m)
        Reduction matrix [a].

    f0 : float, optional (default is 1.0)
        Parameter that allows the dissipation term of the nonlinear 
        second-order dynamical system (dissipative Hamiltonian system) to be 
        controlled (damping that kills the transient response).

    dr : float, optional (default is 0.1)
        Sampling step of the continuous index parameter used in the 
        integration scheme.

    t : int/str, optional (default is 'auto')
        Number of steps that the Ito stochastic differential equation is 
        evolved for.
        If 'auto', number of steps is calculated internally.

    num_samples : int, optional (default is 1)
        Number of times that we sample the manifold. The actual number of 
        data points generated is equal to this 'n' times the original number 
        of points in the data set (n_samples).
    
    parallel : bool, optional (default is False)
        If True, run sampling in parallel.
    
    n_jobs : int, optional (default is -1)
        Number of jobs started by joblib.Parallel().
        Used if 'parallel' is True.
    
    kde_bw_factor : float, optional (default is 1.0)
        Multiplier that modifies the computed KDE bandwidth (Silverman 
        rule-of-thumb).
    
    pot_method : int, optional (default is 2)
        Experimental.
        This is the most expensive part of the computation. 
        Methods 1-8 compute the same joint KDE using different algorithms.
        Method 2 is the most efficient.
        Method 9 computes a conditional (Nadaraya-Watson KDE) * marginal tanh 
        approximation. This is used when the distribution of one of the 
        variables is to be specified.
        Method 10 a computes conditional (Nadaraya-Watson KDE) * marginal KDE.
    
    verbose : bool, optional (default is True)
        If True, print relevant information.
                
    Returns
    -------
    Zs : list of n ndarrays of shape (nu, m) each
         New reduced manifold samples matrix [Zs].
    
    Zs_steps : list n lists of t ndarrays of shape (nu, m)
        Ito steps for generated samples. FOR DEBUGGING.
    
    t : int
        Number of steps used in the Ito stochastic differential equation 
        evolution.
    
    """
    if verbose:
        print("\n\nPerforming Ito sampling.")
        print("------------------------")
        print(f"Projected data (Z) dimensions: {Z0.shape}")
    Zs, Zs_steps, t = _simulate_entire_ito(Z0, H, basis, a, f0, dr, t, 
                                           num_samples, parallel, n_jobs, 
                                           kde_bw_factor, pot_method, verbose)
    
    return Zs, Zs_steps, t

###############################################################################
def sampling(plom_dict):
    """
    PLoM wrapper for sampling function.

    Parameters
    ----------
    plom_dict : dictionary
        PLoM dictionary containing all elements computed by the PLoM framework.
        The relevant dictionary key gets updated by this function.

    Returns
    -------
    None.

    """
    projection_source = plom_dict['options']['projection_source']
    projection_target = plom_dict['options']['projection_target']
    
    if projection_source == "pca":
        X = plom_dict['pca']['training']
    elif projection_source == "scaling":
        X = plom_dict['scaling']['training']
    else:
        X = plom_dict['data']['training']
    
    if projection_target == "dmaps":
        basis = plom_dict['dmaps']['reduced_basis']
    elif projection_target == "pca":
        basis = plom_dict['pca']['training']
    
    f0            = plom_dict['input']['ito_f0']
    dr            = plom_dict['input']['ito_dr']
    t             = plom_dict['input']['ito_steps']
    num_samples   = plom_dict['input']['ito_num_samples']
    parallel      = plom_dict['options']['parallel']
    n_jobs        = plom_dict['options']['n_jobs']
    kde_bw_factor = plom_dict['options']['ito_kde_bw_factor']
    pot_method    = plom_dict['options']['ito_pot_method']
    verbose       = plom_dict['options']['verbose']
    Z             = plom_dict['ito']['Z0']
    a             = plom_dict['ito']['a']
    
    # X = np.copy(np.transpose(X))
    Zs, Zs_steps, t = _sampling(Z, X.T, basis, a, f0, dr, t, num_samples,
                                parallel, n_jobs, kde_bw_factor, pot_method, 
                                verbose)
    
    plom_dict['ito']['Zs']          = Zs
    plom_dict['ito']['Zs_steps']    = Zs_steps
    plom_dict['ito']['t']           = t

###############################################################################
def save_samples(plom_dict):
    
    samples_fname  = plom_dict['options']['samples_fname']
    samples_fmt    = plom_dict['options']['samples_fmt']
    job_desc       = plom_dict['job_desc']
    verbose        = plom_dict['options']['verbose']
    
    if verbose:
        print("\nSaving generated samples to file...")
    
    if samples_fname is None or samples_fname == "" or samples_fname.startswith("."):
        samples_fname = (job_desc.replace(' ', '_') + '_samples_'
                        + time.strftime('%X').replace(':', '_') + "." + samples_fmt)
    elif not samples_fname.endswith(samples_fmt):
        samples_fname = f"{samples_fname}.{samples_fmt}"
        
    if samples_fname.lower().endswith('.npy'):
        np.save(samples_fname, plom_dict['data']['augmented'])
    else:
        np.savetxt(samples_fname, plom_dict['data']['augmented'])
    
    if verbose:
        print(f"Samples saved to {samples_fname}\n")
    
###############################################################################
def make_summary(plom_dict):
    """
    This function takes a populated PLoM dictionary and creates a summary of 
    the PLoM run.
    The summary is saved to the 'summary' key in the PLoM dictionary.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary.

    Returns
    -------
    None.

    """
    job      = plom_dict['job_desc']
    inputs   = plom_dict['input']
    options  = plom_dict['options']
    data     = plom_dict['data']
    pca      = plom_dict['pca']
    dmaps    = plom_dict['dmaps']
    
    training_shape = data['training'].shape
    sc_method      = options['scaling_method']
    
    summary = ["Job Summary\n-----------"]
    
    summary.append(f"Job: {job}\n")

    summary.append(f"Training data dimensions: {training_shape}\n")

    if options['scaling']:
        summary.append("Scaling")
        summary.append(f"Used '{sc_method}' method for scaling.\n")

    if options['pca']:
        summary.append("PCA")

        if options['pca_method']=='cum_energy':
            summary.append("Used Cumulative Energy Content criteria for PCA.")
        elif options['pca_method']=='eigv_cutoff':
            summary.append("Used specified cutoff value criteria for PCA.")
        elif options['pca_method']=='pca_dim':
            summary.append("Used specified PCA dimension.")
        pca_shape = pca['training'].shape
        summary.append(f"PCA features reduction: {training_shape[1]} -> " +
                       f"{pca_shape[1]}\n")

    if options['dmaps']:
        summary.append("DMAPS")
        summary.append(f"Input epsilon: {inputs['dmaps_epsilon']}")
        summary.append(f"Used epsilon: {dmaps['epsilon']:.2f}")
        summary.append("DMAPS eigenvalues: " +
                       f"{dmaps['eigenvalues'][1:dmaps['dimension']+1]}" + 
                       f" [{dmaps['eigenvalues'][dmaps['dimension']+1]:.4f}"+
                       " ...]")
        summary.append(f"Manifold dimension = {dmaps['dimension']}")
        summary.append(f"Used {dmaps['reduced_basis'].shape[1]} eigenvectors" +
                       " for projection.")
        summary.append("Projected data (Z) dimensions: " + 
                       f"{dmaps['training'].shape}\n")

    if data['augmented'] is not None:
        summary.append("Sampling")

        summary.append(f"Generated {inputs['ito_num_samples']} samples.")

        summary.append("Augmented data dimensions: " + 
                       f"{data['augmented'].shape}\n")
    
    plom_dict['summary'] = summary
    

###############################################################################
def save_summary(plom_dict, fname=None):
    if fname is None:
        fname = plom_dict['job_desc'] + "_plom_summary.txt"
    summary = '\n'.join(plom_dict['summary'])
    with open(fname, "w") as summary_file:
        summary_file.write(summary)
        
###############################################################################
def mse(X, Y, *, squared=True):
    """
    Mean squared error reconstruction loss.

    Parameters
    ----------
    X : ndarray of shape (n_samples,) or (n_samples, n_features)
        Ground truth (correct) target values.
    
    Y : array-like of shape (n_samples,) or (n_samples, n_features)
        Estimated target values.

    squared : bool, optional (default is True)
        If True, returns MSE value; if False, returns RMSE value.

    Returns
    -------
    error : float
        A non-negative floating point value (the best value is 0.0).

    """
    error = np.mean((X - Y) ** 2)
    return error if squared else np.sqrt(error)
    
###############################################################################
def _short_date():
    return datetime.now().replace(microsecond=0)

###############################################################################
def _normalize_values(values):
    exponents = [np.floor(np.log10(abs(num))) for num in values]
    min_exponent = int(min(exponents))
    normalized_values = [num / 10**min_exponent for num in values]
    return normalized_values, min_exponent

###############################################################################
def run(plom_dict):

## Start    
    start_time = datetime.now()
    job_desc       = plom_dict['job_desc']
    inputs         = plom_dict['input']
    options        = plom_dict['options']
    
    verbose        = options['verbose']
    scaling_opt    = options['scaling']
    pca_opt        = options['pca']
    dmaps_opt      = options['dmaps']
    projection_opt = options['projection']
    sampling_opt   = options['sampling']
    saving_opt     = options['save_samples']
    num_samples    = inputs['ito_num_samples']
    
    
    if verbose:
        print(f"\nPLoM run starting at {str(datetime.now()).split('.')[0]}")

## Scaling
    if scaling_opt:
        scale(plom_dict)

## PCA
    if pca_opt:
        pca(plom_dict)

## DMAPS
    if dmaps_opt:
        dmaps(plom_dict)

## Projection (Z)
    if projection_opt:
        sample_projection(plom_dict)

## Sampling
    if sampling_opt and num_samples != 0:
        sampling(plom_dict)

## Inverse Projection (Z)
    if projection_opt:
        inverse_sample_projection(plom_dict)

## Inverse PCA
    if pca_opt:
        inverse_pca(plom_dict)

## Inverse scaling
    if scaling_opt:
        inverse_scale(plom_dict)

## MSE
    rmse = mse(plom_dict['data']['training'], 
               plom_dict['data']['reconst_training'], 
               squared=False)
    plom_dict['data']['rmse'] = rmse
    if verbose:
        print(f'\nTraining data reconstruction RMSE = {rmse:.6E}')

## Summary
    make_summary(plom_dict)

## Saving samples
    if saving_opt:
        save_samples(plom_dict)
        save_summary(plom_dict)

## End
    end_time = datetime.now()
    if verbose:
        if plom_dict['data']['augmented'] is not None:
            print("\nAugmented data dimensions: ", 
                  f"{plom_dict['data']['augmented'].shape}")
        print(f"\n*** Total run time = {str(end_time-start_time)[:-3]} ***")
        print(f"\nPLoM run complete at {str(datetime.now()).split('.')[0]}")

###############################################################################
def run_dmaps(plom_dict):

## Start
    start_time = datetime.now()
    
    options        = plom_dict['options']
    
    verbose        = options['verbose']
    scaling_opt    = options['scaling']
    pca_opt        = options['pca']
    dmaps_opt      = options['dmaps']
    projection_opt = options['projection']


    if verbose:
        print("\nPLoM run (DMAPS ONLY) starting at ", 
              f"{str(datetime.now()).split('.')[0]}")

## Scaling
    if scaling_opt:
        scale(plom_dict)

## PCA
    if pca_opt:
        pca(plom_dict)

## DMAPS
    if dmaps_opt:
        dmaps(plom_dict)

## Projection (Z)
    if projection_opt:
        sample_projection(plom_dict)

## Inverse Projection (Z)
    if projection_opt:
        inverse_sample_projection(plom_dict)

## Inverse PCA
    if pca_opt:
        inverse_pca(plom_dict)

## Inverse scaling
    if scaling_opt:
        inverse_scale(plom_dict)

## MSE
    rmse = mse(plom_dict['data']['training'], 
               plom_dict['data']['reconst_training'], 
               squared=False)
    plom_dict['data']['rmse'] = rmse
    if verbose:
        print(f'\nTraining data reconstruction RMSE = {rmse:.6E}')

## Summary
    make_summary(plom_dict)

## End
    end_time = datetime.now()
    if verbose:
        print(f"\n*** Total run time = {str(end_time-start_time)[:-3]} ***")
        print("\nPLoM run (DMAPS ONLY) complete at ", 
              f"{str(datetime.now()).split('.')[0]}")

###############################################################################
def run_sampling(plom_dict):

## Start
    start_time = datetime.now()
    
    inputs         = plom_dict['input']
    options        = plom_dict['options']
    
    verbose        = options['verbose']
    scaling_opt    = options['scaling']
    pca_opt        = options['pca']
    projection_opt = options['projection']
    sampling_opt   = options['sampling']
    saving_opt     = options['save_samples']
    num_samples    = inputs['ito_num_samples']
    
    if verbose:
        print(f"\nPLoM run (SAMPLING ONLY) starting at \
{str(datetime.now()).split('.')[0]}")

## Check if DMAPS already run
    if plom_dict['ito']['Z0'] is None:
        raise Exception("Sampling aborted. DMAPS results not found. "+
                        "Run DMAPS before sampling.")

## Sampling
    if sampling_opt and num_samples != 0:
        sampling(plom_dict)

## Inverse Projection (Z)
    if projection_opt:
        inverse_sample_projection(plom_dict)

## Inverse PCA
    if pca_opt:
        inverse_pca(plom_dict)

## Inverse scaling
    if scaling_opt:
        inverse_scale(plom_dict)

## Summary
    make_summary(plom_dict)

## Saving samples
    if saving_opt:
        save_samples(plom_dict)
        save_summary(plom_dict)

## End
    end_time = datetime.now()
    if verbose:
        if plom_dict['data']['augmented'] is not None:
            print("\nAugmented data dimensions: ", 
                  f"{plom_dict['data']['augmented'].shape}")
        print(f"\n*** Total run time = {str(end_time-start_time)[:-3]} ***")
        print("\nPLoM run (SAMPLING ONLY) complete at ", 
              f"{str(datetime.now()).split('.')[0]}")

###############################################################################
####################                                       ####################
#################### Some tools to be used after PLoM runs ####################
####################                                       ####################
###############################################################################

###############################################################################
############################ Plotting functions ###############################
###############################################################################
def plot2D_reconstructed_training(plom_dict, i=0, j=1, size=9, pt_size=10, 
                                  color=['cmap','cmap']):
    training = plom_dict['data']['training'].T
    reconst_training = plom_dict['data']['reconst_training'].T
    [c1, c2] = color
    if c1 == 'cmap': c1 = range(training.shape[1])
    if c2 == 'cmap': c2 = range(training.shape[1])
    plt.figure(figsize=(size, size))
    t_plot = plt.scatter(training[i], training[j], s=pt_size, c=c1)
    t_r_plot = plt.scatter(reconst_training[i], reconst_training[j], 
                           s=pt_size, c=c2)
    plt.legend((t_plot, t_r_plot), ('Training', 'Reconstructed training'), 
               loc='best')
    plt.gca().set_aspect('equal')
    plt.title(f'Training vs reconstructed training, n={training.shape[1]}')
    plt.show()

###############################################################################
def plot2d_samples(plom_dict, i=0, j=1, size=9, pt_size=10):
    training = plom_dict['data']['training'].T
    samples = plom_dict['data']['augmented'].T
    N = training.shape[1]
    num_sample = plom_dict['input']['ito_num_samples']
    plt.figure(figsize=(size, size))
    plt.scatter(training[i], training[j], color='b', s=pt_size, 
                label='Training', marker="+")
    for k in range(num_sample):
        plt.scatter(samples[i, k*N:(k+1)*N], samples[j, k*N:(k+1)*N], 
                    color=np.random.rand(3), s=pt_size)
    plt.legend(loc='best')
    plt.gca().set_aspect('equal')
    plt.title('Training + New samples')
    plt.show()

###############################################################################
def plot3d_samples(plom_dict, i=0, j=1, size=9, pt_size=10):
    training = plom_dict['data']['training'].T
    samples = plom_dict['data']['augmented'].T
    
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(training[0], training[1], training[2], marker='o')
    ax.scatter(samples[0], samples[1], samples[2], marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('Training + New samples')
    plt.show()

###############################################################################
def plot_dmaps_eigenvalues(plom_dict, n=0, size=8, pt_size=10, save=False):
    evals = plom_dict['dmaps']['eigenvalues'][1:]
    m = plom_dict['dmaps']['dimension']
    if n == 0:
        n = max(min(2*m, evals.size), 10)
    elif n=='all':
        n = evals.size
    plt.figure(figsize=(size, size/2))
    plt.plot(range(m), evals[:m], c='r')
    plt.scatter(range(m), evals[:m], c='r', s=pt_size)
    plt.plot(range(m-1, n), evals[m-1:n], c='b', alpha=0.25)
    plt.scatter(range(m, n), evals[m:n], c='b', s=pt_size)
    plt.yscale("log")
    plt.title(f"DMAPS Eigenvalues (m={m})")
    if save:
        plt.savefig('DMAPS_eigenvalues.png')
    plt.show()

###############################################################################
def plot2D_dmaps_basis(plom_dict, vecs=[1,2], size=9, pt_size=10):
    evecs = plom_dict['dmaps']['basis'][:, vecs].T
    c = range(evecs.shape[1])
    plt.figure(figsize=(size, size))
    plt.scatter(evecs[0], evecs[1], s=pt_size, c=c)
    plt.title(f'DMAPS basi vectors {vecs[0]} (x) vs {vecs[1]} (y)')
    plt.show()

###############################################################################
def plot_pca_eigenvalues(plom_dict, log=True, save=False):
    evals = np.flip(plom_dict['pca']['eigvals'])
    evals = evals[evals > 1e-15]
    plt.figure(figsize=(12, 6))
    if log:
        plt.yscale('log')
        plt.ylim(evals.min()*0.9, evals.max()*1.1)
    plt.scatter(range(len(evals)), evals, s=7)
    plt.title("PCA Eigenvalues")
    plt.xticks(np.arange(0, len(evals), 5))
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.grid()
    if save:
        plt.savefig('PCA_eigenvalues.png')
    plt.show()

###############################################################################
########################## Loading and saving data  ###########################
###############################################################################
def load_dict(fname):
    file = open(fname, "rb")
    plom_dict = pickle.load(file)
    return plom_dict  

###############################################################################
def save_dict(plom_dict, fname=None):
    if fname is None:
        fname = plom_dict['job_desc'] + "_dict.plom"
    file = open(fname, "wb")
    pickle.dump(plom_dict, file)
    file.close()

###############################################################################
def save_epsvsm(plom_dict, fname=None):
    if fname is None:
        fname = plom_dict['job_desc'] + "_epsvsm.txt"
    eps = plom_dict['dmaps']['eps_vs_m']
    n = len(eps)
    with open(fname, "a") as eps_file:
        eps_file.write("\nEps         m\n-------------\n")
        for i in range(n):
            e = f"{eps[i][0]:.2f}"
            m = str(eps[i][1])
            s = e + (12-len(e))*" " + m + "\n"
            eps_file.write(s)
            
###############################################################################
def save_training(plom_dict, fname=None, fmt='txt'):
    """
    Save training data to txt or npy.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary containing the training data.
    
    fname : str, optional (default is None)
        File name. If None, job description will be used instead.
    
    fmt : str, optional (default is 'txt')
        Format of saved file.
        If 'txt', save data to txt file.
        If 'npy', save data to npy file.

    Returns
    -------
    None.

    """
    if fname is None:
        fname = plom_dict['job_desc'] + "_training."
    if fmt == 'txt':
        np.savetxt(fname+fmt, plom_dict['data']['training'])
    elif fmt == 'npy':
        np.save(fname+fmt, plom_dict['data']['training'])
    else:
        raise Exception("'fmt' argument can be either 'txt' or 'npy'.")
    
###############################################################################
def _save_samples(plom_dict, fname=None, fmt='txt'):
    """
    Save generated samples to txt or npy.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary containing the generated samples.
    
    fname : str, optional (default is None)
        File name. If None, job description will be used instead.
    
    fmt : str, optional (default is 'txt')
        Format of saved file.
        If 'txt', save data to txt file.
        If 'npy', save data to npy file.

    Returns
    -------
    None.

    """
    if fname is None:
        fname = plom_dict['job_desc'] + "_samples."
    if fmt == 'txt':
        np.savetxt(fname+fmt, plom_dict['data']['augmented'])
    elif fmt == 'npy':
        np.save(fname+fmt, plom_dict['data']['augmented'])
    else:
        raise Exception("'fmt' argument can be either 'txt' or 'npy'.")
    
###############################################################################
############################## Misc. functions  ###############################
###############################################################################
def print_summary(plom_dict):
    print(*plom_dict['summary'], sep='\n')
    
###############################################################################
def print_epsvsm(plom_dict):
    eps = plom_dict['dmaps']['eps_vs_m']
    n = len(eps)
    print("Eps         m\n-------------")
    for i in range(n):
        e = f"{eps[i][0]:.2f}"
        m = str(eps[i][1])
        s = e + (12-len(e))*" " + m
        print(s)
###############################################################################
def list_input_parameters(plom_dict=None):
    if plom_dict is None:
        input_params = list(initialize()['input'].keys())
    else:
        input_params = list(plom_dict['input'].keys())
    for _ in input_params:
        print(_)
###############################################################################
def list_options(plom_dict=None):
    if plom_dict is None:
        options = initialize()['options']
    else:
        options = list(plom_dict['options'].keys())
    for _ in options:
        print(_)

###############################################################################
def get_diffusion_distances(plom_dict, full_basis=False):
    if full_basis:
        data = plom_dict['dmaps']['basis'][:, 1:]
    else:
        data = plom_dict['dmaps']['reduced_basis']
    
    distances = distance_matrix(data, data)
    return distances

###############################################################################
def get_training(plom_dict):
    """
    Get training data from PLoM dictionary.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary.

    Returns
    -------
    training : ndarray of shape (n_samples, n_features)
        Training data.

    """
    return plom_dict['data']['training']

###############################################################################
def get_reconst_training(plom_dict):
    """
    Get reconstructed training data from PLoM dictionary.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary.

    Returns
    -------
    reconst_training : ndarray of shape (n_samples, n_features)
        Reconstructed training data.

    """
    return plom_dict['data']['reconst_training']

###############################################################################
def get_samples(plom_dict, k=0):
    """
    Get samples generated by PLoM.

    Parameters
    ----------
    plom_dict : dict
        PLoM dictionary.
        
    k : int, optional (default is 0)
        Number of samples requested where each sample is a dataset of size
        equal to the training dataset size.

    Returns
    -------
    samples : ndarray of shape (k*train_size, n_features)
        A matrix of generated samples. If k is 0, all samples are returned.
        If k is a positive integer, k samples are returned.

    """
    
    # check if 'k' is an int
    if not isinstance(k, int):
        raise TypeError("k (number of samples requested) must be an integer.")
    # check if 'k' is not negative
    if k < 0:
        raise ValueError("k (number of samples requested) must be 0 or ", 
                         "positive.")
    
    # k==0: all samples requested
    if k == 0:
        samples = plom_dict['data']['augmented']
        if samples is None:
            raise ValueError("Samples not found. Sampling not run?")
        if not isinstance(samples, np.ndarray):
            raise TypeError("'plom_dict['data']['augmented']' is not a 2D ", 
                            "Numpy array.")
    
    # k !=0: k samples requested
    else:
        try:
            train_size = plom_dict['data']['training'].shape[0]
        except:
            raise AttributeError("plom_dict['data']['training'] is not a 2D ", 
                                 "Numpy array. Unable to deduce training ", 
                                 "dataset size.")
        try:
            samples_size = plom_dict['data']['augmented'].shape[0]
        except:
            raise AttributeError("plom_dict['data']['augmented'] is not a 2D", 
                                 "Numpy array. Unable to deduce samples ", 
                                 "dataset size.")
        if k*train_size < samples_size:
            samples = plom_dict['data']['augmented'][:k*train_size]
        else:
            raise Exception("Too many samples requested (available samples = ",
                            f"{int(samples_size/train_size)}")

    return samples

###############################################################################
########################## Conditioning functions  ############################
###############################################################################

# def gaussian_kde(training, pt, kde_bw_factor=1, options=None):
#     N, n = training.shape
#     h = (4 / (N*(2+n))) ** (1/(n+4)) * kde_bw_factor
#     return h, (1/N * np.sum(np.exp((-1/2/h/h) * 
#                                    np.linalg.norm(training - pt, axis=1)**2)))

###############################################################################
# def plot_training_pdf(plom_dict, size=9, surface=True):
#     training  = plom_dict['training']
#     options   = plom_dict['options']
#     bw_factor = plom_dict['input']['kde_bw_factor']
    
#     xmin = 1.2 * min(training[:, 0])
#     xmax = 1.2 * max(training[:, 0])
#     ymin = 1.2 * min(training[:, 1])
#     ymax = 1.2 * max(training[:, 1])
#     xs, ys = np.meshgrid(np.linspace(xmin,xmax,100), 
#                           np.linspace(ymin,ymax,100))
#     xs_flat = xs.flatten()
#     ys_flat = ys.flatten()
#     grid = np.array((xs_flat, ys_flat)).T
    
#     pdf_flat = np.array([gaussian_kde(training, pt, bw_factor, options)[1] 
#                          for pt in grid])
#     pdf = pdf_flat.reshape(xs.shape)
#     h_used = gaussian_kde(training, [0,0], bw_factor, options)[0]
    
#     fig = plt.figure(figsize=(size, size))
#     ax = fig.add_subplot(111, projection='3d')
#     if surface:
#         ax.plot_surface(xs, ys, pdf, cmap=cm.coolwarm)
#     else:
#         ax.scatter(xs, ys, pdf)
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.title(f'Bandwidth used = {h_used}\n(Bandwidth factor = {bw_factor})')
#     plt.show()
    
###############################################################################
def _conditional_expectation(data, qoi_cols, cond_cols, cond_vals, weights=None,
                             sw=None, bw_opt_kwargs={}, return_bw=False, 
                             verbose=True):
    """
    Computes the conditional expectation of a quantity of interest (QoI), E[Q | W=w0], 
    given a set of conditioning variables and their corresponding values. Optionally, 
    the function can compute optimal bandwidths for kernel smoothing when calculating 
    conditioning weights.

    :param data: (np.ndarray) A 2D array of shape (N, D), where N is the number of samples 
                 and D is the number of variables. The data matrix contains the samples 
                 from which the conditional expectation is computed.
                 
    :param qoi_cols: (int or list of int) Index or indices of the column(s) in `data` that represent 
                     the quantity or quantities of interest (QoI) for which the conditional expectation 
                     is computed. If a single integer is provided, an optimal bandwidth for the QoI can 
                     be found and used. If multiple QoIs are provided (a list of integers), Silverman's 
                     bandwidth will be used, even if the user requests an optimal bandwidth.
                     
    :param cond_cols: (list of int) List of column indices in `data` that represent the conditioning 
                      variables (RVs) W.
                      
    :param cond_vals: (list or np.ndarray) List or array of values for the conditioning variables 
                      W=w0. The length should match the number of conditioning variables specified 
                      in `cond_cols`.
    
    :param weights: (np.ndarray, optional) Precomputed weights for conditioning. If provided, these 
                    weights will be considered final and used directly. If not provided, the function 
                    will compute the weights based on the selected bandwidth method (Silverman's or 
                    optimal).
                    
    :param sw: (str or float, optional) Specifies the bandwidth for kernel smoothing. If a numerical 
               value (either a single float or a vector of floats) is provided, it will be directly 
               used as the bandwidth. If a string is provided:
               - "optimal_joint": Jointly optimizes a bandwidth vector of size equal to the number of 
                 conditioning variables.
               - "optimal_marg": Optimizes the bandwidth vector one dimension at a time (marginally).
               If no valid option is provided or if multiple QoIs are specified, Silverman's bandwidth 
               will be used.
               Default is None.
    
    :param bw_opt_kwargs: (dict, optional) Additional keyword arguments to pass to the bandwidth 
                          optimization functions when `sw` is set to "optimal_joint" or "optimal_marg".
    
    :param return_bw: (bool, optional) If True, the function returns the computed or selected bandwidth 
                      along with the conditional expectation and variance. Default is False.
                      
    :param verbose: (bool, optional) If True, the function prints progress and intermediate results. 
                    Default is True.

    :returns:
        - expn: (float or np.ndarray) The conditional expectation of the QoI, E[Q | W=w0].
        - var:  (float or np.ndarray) The conditional variance of the QoI, Var(Q | W=w0).
        - sw:   (float, optional) The bandwidth used for conditioning weights, returned if `return_bw=True`.
    
    :raises:
        - ValueError: If an invalid option for bandwidth is provided or if optimal bandwidth 
                      computation is attempted for multiple QoIs.
    
    :notes:
        - This function can handle multivariate QoI (multiple columns) and multivariate conditioning 
          (multiple conditioning variables).
        - The kernel bandwidth for conditioning weights can either be pre-specified, computed using 
          Silverman's rule, or optimized using joint/marginal optimization methods.
    """
    
    start = _short_date()
    if verbose:
        print('\n***********************************************************')
        print('Conditional expectation evaluation starting at', start)
    
    Nsim = data.shape[0]
    nw = np.atleast_1d(cond_cols).shape[0]
    nq = 1
    
    if verbose:
        print(f'\nEstimating the conditional expectation of <variable \
{qoi_cols}> conditioned on <variable{"" if nw==1 else "s"} {cond_cols}> = \
<{cond_vals}>.')
        print(f'Using N = {Nsim} samples.')
    
    # Conditioning weights
    # if sw is None:
        # sw = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq))
    # if verbose:
        # print("\nComputing conditioning weights.")
        # print(f'Using bw = {sw:.6f} for conditioning weights.')
    # weights = _get_conditional_weights(X[:, cond_cols], cond_vals, sw,
                                       # verbose=verbose)
    
    ### Conditioning weights
    if weights is None:
        if type(sw) is str and sw.startswith("optimal"):
            if np.atleast_1d(qoi_cols).shape[0] > 1:
                print("Optimal bandwidth can be found for single QoI only. Using Silverman's bandwidth instead.")
                sw = None
            else:
                if verbose:
                    print("\nFinding optimal bandwidth for conditioning.")
                if sw == "optimal_joint":
                    sw = conditioning_jointly_optimal_bw(
                        data, cond_cols, qoi_cols, verbose=verbose,
                        **bw_opt_kwargs)
                elif sw == "optimal_marg":
                    sw = conditioning_marginally_optimal_bw(
                        data, cond_cols, qoi_cols, verbose=verbose,
                        **bw_opt_kwargs)
                else:
                    if verbose:
                        print("Invalid option for bandwidth. Using Silverman's bandwidth.")
                    sw = None
        if sw is None:
            if verbose:
                print("Using Silverman's bandwidth.")
            sw = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq))
        if verbose:
            print("\nComputing conditioning weights.")
            print(f'Using bw = {sw} for conditioning weights.')
        weights = _get_conditional_weights(data[:, cond_cols], cond_vals, sw,
                                           verbose=verbose)
    else:
        print("\nUsing user-specified weights.")
    Nsim = np.sum(weights)**2 / np.sum(weights**2)
    
    ## Expectation evaluation
    if verbose:
        print("\nComputing expectation value.")
    q = data[:, qoi_cols]
    expn = np.atleast_1d(np.dot(weights, q))
    var = np.atleast_1d(np.dot(weights, q*q) - expn**2)
    if expn.shape[0] == 1:
        expn = expn[0]
        var = var[0]
    if verbose:
        print(f"\nConditional expected value of variable(s) {qoi_cols}: E = \
{expn}")
        print(f"\nConditional variance of variable(s) {qoi_cols}: Var = {var}")

    end = _short_date()
    if verbose:
        print('\nConditioning complete at', end)
        print('Time =', end-start)
    
    if return_bw:
        return expn, var, sw
    else:
        return expn, var

###############################################################################
def conditional_expectation(obj, qoi_cols, cond_cols, cond_vals, weights=None, 
                            sw=None, bw_opt_kwargs={}, return_bw=False, 
                            verbose=True):
    """
    Compute the conditional expectation of a quantity of interest (QoI) given certain conditioning variables.

    This function serves as a wrapper around the `_conditional_expectation` function, which calculates the 
    expected value of a specified QoI conditioned on provided values of other random variables (RVs).

    :param obj: (dict or np.ndarray) 
        - If a dictionary, it should contain a key 'data' with a subkey 'augmented' holding the data matrix.
        - If a NumPy array, it should represent the data matrix directly.
        
    :param qoi_cols: (int or list) 
        Column index or indices of the RV for which the conditional expectation is computed.

    :param cond_cols: (list) 
        Column indices of conditioning RVs.

    :param cond_vals: (list) 
        Values of the conditioning RVs corresponding to `cond_cols`.

    :param weights: (np.ndarray, optional) 
        User-specified weights for the conditioning. If `None`, the function will compute the weights based on bandwidth.

    :param sw: (float or str, optional) 
        Specifies the bandwidth for the conditioning weights. 
        If a numerical value is provided, it will be directly used. 
        If 'optimal_joint' or 'optimal_marg', the function will compute an optimal bandwidth vector.

    :param bw_opt_kwargs: (dict, optional) 
        Additional keyword arguments for bandwidth optimization.

    :param return_bw: (bool, optional) 
        If `True`, the function returns the computed bandwidth along with the expectation and variance.

    :param verbose: (bool, optional) 
        If `True`, the function will print verbose output during execution.

    :raises ValueError: 
        If `obj` is not of type `dict` or `np.ndarray`.

    :return: 
        - (float, float) if `return_bw` is `False`: The conditional expectation and variance of the QoI.
        - (float, float, float) if `return_bw` is `True`: The conditional expectation, variance, and bandwidth.
    """
                            
    args = {'qoi_cols': qoi_cols, 'cond_cols': cond_cols, 'cond_vals': cond_vals,
            'weights': weights, 'sw': sw, 'bw_opt_kwargs': bw_opt_kwargs, 
            'return_bw': return_bw, 'verbose': verbose}
    
    if isinstance(obj, dict):
        data = obj['data']['augmented']
    elif isinstance(obj, np.ndarray):
        data = obj
    else:
        raise ValueError("Invalid type for 'obj'. Expected dict or np.ndarray.")
    
    if return_bw:
        expn, var, sw = _conditional_expectation(data, **args)
        return expn, var, sw
    else:
        expn, var = _conditional_expectation(data, **args)
        return expn, var
    
###############################################################################
def _get_conditional_weights(W, w0, sw=None, nq=1, parallel=False, batches=2,
                              verbose=True):
    """
    Calculate the conditional weights based on the provided conditioning variables.

    This function computes weights for the conditioning variables, which are used in estimating
    the conditional expectation. The weights are determined based on the distance of each sample
    from the specified conditioning value `w0`, adjusted by a specified bandwidth.

    :param W: (np.ndarray) 
        A 2D NumPy array where each row represents a sample and each column represents a conditioning variable. 
        If a 1D array is provided, it will be reshaped into a 2D array with one column.

    :param w0: (np.ndarray) 
        A 1D array representing the conditioning values against which the weights are computed.

    :param sw: (float, optional) 
        The bandwidth for the weights. If `None`, it will be computed based on the number of samples and variables.

    :param nq: (int, optional) 
        The number of quantities of interest (QoIs). Default is 1.

    :param parallel: (bool, optional) 
        If `True`, the computation of norms will be executed in parallel across batches for efficiency.

    :param batches: (int, optional) 
        The number of batches to split the data into when parallel computation is enabled. Default is 2.

    :param verbose: (bool, optional) 
        If `True`, the function will print verbose output during execution.

    :return: (np.ndarray) 
        A normalized array of conditional weights corresponding to the samples in `W`.
    """
    
    if W.ndim == 1:
        W = W[:, np.newaxis]
    Nsim, nw = W.shape
    w_std = np.std(W, axis=0)
    if sw is None:
        sw = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq))
    
    if parallel:
        batch_size = int(Nsim / batches)
        r = Nsim % batch_size
        w_norms = Parallel(n_jobs=-1)(
            delayed(np.linalg.norm)(
                (W[i*batch_size: (i+1)*batch_size]-w0)/w_std, axis=1) 
            for i in range(batches))
        w_norms = (np.array(w_norms)).flatten()
        if r>1:
            w_norms = np.append(w_norms, 
                                np.linalg.norm((W[-r:]-w0)/w_std, axis=1))
        elif r==1:
            w_norms = np.append(w_norms, 
                                np.linalg.norm((W[-r:]-w0)/w_std, axis=0))
        w_norms = -1/(2*sw**2) * w_norms**2
    else:
        w_norms = -np.linalg.norm((W-w0)/(w_std*np.sqrt(2*sw**2)), axis=1)**2 # verified; allows for vector-valued bandwidth sw

    w_dist = np.exp(w_norms - np.max(w_norms))
    w_dist_tot = np.sum(w_dist)
    return w_dist/w_dist_tot

###############################################################################
def _evaluate_kernels_sum(X, x, H, kernel_weights=None):
    X = np.asarray(X)
    x = np.asarray(x)
    H = np.asarray(H)
    
    if X.ndim == 1:
        X = X[:, np.newaxis]
    
    if H.ndim == 0: ## H is specified as a scalar, i.e. 1-D pdf
        H = np.atleast_2d(H**2)
    elif H.ndim == 1: ## H is specified as a list of n BWs, i.e. n-D pdf
        H = np.diag(H**2)
    
    N, n = X.shape
    
    if kernel_weights is None:
        kernel_weights = 1./N
    
    diff = X - x
    Hinv = np.linalg.inv(H)
    factor = 1/(2*np.pi)**(n/2)/np.sqrt(np.linalg.det(H)) 
    pdf = (factor * kernel_weights * 
           np.exp((-1/2) * np.sum(np.matmul(diff, Hinv) * diff, axis=1)))
    pdf = np.sum(pdf)
    return np.append(x, pdf)

###############################################################################
def forward_model(x, data, qoi_col=None, cond_cols=None, h=None):
    """
    Predict the quantity of interest (QoI) based on input variables using a conditional expectation approach.

    This function estimates the expected value of the QoI for given input values by computing
    the conditional expectation of the QoI given the specified conditioning features from the data.

    Parameters
    ----------
    x: array-like, shape (n_samples, n_features)
        Values of the input variables for which the QoI is predicted.

    data: ndarray, shape (n_samples, n_features)
        Dataset used to estimate the 'n_features'-dimensional joint probability density function (PDF).

    qoi_col: int, optional
        Index of the quantity of interest (model output) within the features. If not specified, it defaults to the last column.

    cond_cols: list of int, optional
        Predefined list of indices of conditional features to rank. If None (default), assumes all features except the one specified by `qoi_col`:
        cond_cols = [i for i in range(n_features) if i != qoi_col].

    h: float or array-like, optional
        Bandwidth(s) to be used in the conditional expectation computation. Must be non-negative.

    Returns
    -------
    y_pred: array-like or float
        The predicted values of the QoI based on the input values. If multiple predictions are made, returns an array; 
        if a single prediction is made, returns a float.

    Raises
    ------
    ValueError
        If any bandwidth(s) are negative or if there is a dimension mismatch between `cond_cols` and `x`.
    """
    
    n_x = data.shape[1] - 1
    x = np.array(x).reshape(-1, n_x)
    N_x = x.shape[0]
    if N_x == 1 and n_x > 1:
        x = x.T
    
    if qoi_col is None:
        qoi_col = np.atleast_1d(n_x)
    if cond_cols is None:
        qoi_col = np.atleast_1d(qoi_col)
        cond_cols = np.array([i for i in range(n_x) if i not in qoi_col])
    
    assert n_x == len(cond_cols), "Dimension mismatch between cond_cols and x"

    y_pred = []
    for i in range(N_x):
        exp, var = _conditional_expectation(data, qoi_col, cond_cols, x[i], sw=h,
                                            verbose=False)
        y_pred.append(exp)
    
    return np.array(y_pred) if len(y_pred) > 1 else y_pred[0]

###############################################################################
def _fitness(h, X_train, y_train, model_data, logscale=False, h_full=None, h_idx=None, verbose=False):
    """
    Evaluates the fitness of the model based on the mean squared error (MSE) between 
    the predicted outputs and the actual outputs for the training data.

    This function computes the predicted values using the forward model and calculates 
    the MSE as a measure of the model's performance. The bandwidth parameter can be 
    adjusted, and results can be printed for debugging or evaluation purposes.

    Parameters
    ----------
    h: float or array-like
        The bandwidth parameter(s) used in the forward model. If `logscale` is True, 
        this value is treated as being in log-space.

    X_train: ndarray of shape (n_samples, n_features)
        The training input data used for making predictions, where n_samples is the 
        number of samples and n_features is the number of features.

    y_train: ndarray of shape (n_samples,)
        The actual output values corresponding to `X_train`, used to compute the MSE.

    model_data: ndarray of shape (n_samples, n_features)
        The dataset used by the forward model to generate predictions. It should 
        include both the conditional features and the quantity of interest.

    logscale: bool, optional (default=False)
        If True, the bandwidth parameter `h` is treated as being in log-space and 
        is exponentiated before being used.

    h_full: ndarray or None, optional (default=None)
        An optional array to store the current bandwidth value at the specified index. 
        This allows tracking multiple bandwidth values.

    h_idx: int or None, optional (default=None)
        The index in `h_full` where the current bandwidth value should be stored. 
        This is relevant only if `h_full` is provided.

    verbose: bool, optional (default=False)
        If True, detailed output regarding the evaluation process is printed to the console.

    Returns
    -------
    mse_: float
        The computed mean squared error (MSE) between the predicted values and the 
        actual output values for the training data.

    Notes
    -----
    The function will print the current bandwidth and MSE values if `verbose` is set to True.
    """
    
    if verbose:
        print("Evaluating fitness")
    
    if h_full is not None and h_idx is not None:
        h_full[h_idx] = h
        h = h_full
    
    if logscale:
        h = np.exp(h)
    
    y_pred = forward_model(X_train, model_data, h=np.array(h))
    mse_ = mse(y_train, y_pred)
    
    if verbose:
        print("H =", h)
        print("MSE =", mse_, "\n")
    return (mse_)

###############################################################################
def conditioning_jointly_optimal_bw(
    data, cond_cols=None, qoi_col=None, split_frac=0.1, split_seed=None, optimizer='ga', 
    opt_seed=None, ga_bounds=(1e-09, 1e3), ga_workers=-1, polish=True, 
    return_mse=False, kfolds=5, logscale=False, shuffle=True,
    print_precision=4, verbose=False):
    """
    Jointly optimizes the bandwidths for conditional features in kernel density 
    estimation using a genetic algorithm or differential evolution.

    This function computes the optimal bandwidths for conditioning variables in 
    a conditional probability density estimation framework. The bandwidths are 
    optimized using cross-validation and the differential evolution algorithm, 
    with options for specifying search bounds, logging, and adjusting for log-scale.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The input dataset containing both the conditioning variables and the 
        quantity of interest (QoI).
    
    cond_cols : list of int
        Indices of the conditioning features (input variables) to optimize the 
        bandwidth for.
    
    qoi_col : int or list of int
        Index of the quantity of interest (model output). Only a single QoI is allowed.

    split_frac : float, optional (default=0.1)
        Fraction of the data used for the test split during cross-validation. 
        Must be between 0 and 1.

    split_seed : int or None, optional (default=None)
        Seed for reproducibility of the training/testing split. If provided, 
        the data will be shuffled accordingly.

    optimizer : str, optional (default='ga')
        The optimization method to use. Currently supports 'ga' (genetic algorithm).

    opt_seed : int or None, optional (default=None)
        Random seed for the optimizer to ensure reproducibility.

    ga_bounds : tuple, optional (default=(1e-09, 1e3))
        Bounds for the genetic algorithm when searching for optimal bandwidths.

    ga_workers : int, optional (default=-1)
        Number of CPU cores to use for parallel optimization. Use -1 to utilize all available cores.

    polish : bool, optional (default=True)
        Whether to perform a local search after the genetic algorithm has converged.

    return_mse : bool, optional (default=False)
        If True, the function will return both the optimal bandwidth and the mean squared error (MSE).

    kfolds : int, optional (default=5)
        Number of cross-validation folds to use for evaluating the fitness of bandwidths.

    logscale : bool, optional (default=False)
        If True, the bandwidths are optimized in log-scale and converted back via exponentiation.

    shuffle : bool, optional (default=True)
        Whether to shuffle the dataset before performing the train-test split for cross-validation.

    print_precision : int, optional (default=4)
        Number of decimal places to use when printing bandwidth values during optimization.

    verbose : bool, optional (default=False)
        If True, prints detailed progress, including bandwidth values and mean squared error, during optimization.

    Returns
    -------
    h_opt : ndarray
        The optimized bandwidth values for the conditioning variables.

    mse_opt : float, optional
        The mean squared error of the model using the optimized bandwidths, 
        returned if `return_mse` is True.

    Raises
    ------
    ValueError
        If more than one QoI is provided in `qoi_col`, as the function supports 
        optimization for a single QoI only.

    Notes
    -----
    - The function performs k-fold cross-validation, where in each fold the dataset 
      is split into training and testing sets based on `split_frac`. The optimal 
      bandwidths are found by minimizing the MSE between predicted and actual outputs.
    
    - The `differential_evolution` optimizer is used for finding optimal bandwidths. 
      The search bounds and the log-scale option can significantly affect the results.

    - If `verbose` is set to True, detailed output about the optimization process 
      is printed for each fold.

    Example
    -------
    >>> data = np.random.randn(100, 5)
    >>> cond_cols = [0, 1, 2]
    >>> qoi_col = 4
    >>> h_opt, mse_opt = conditioning_jointly_optimal_bw(data, cond_cols, qoi_col, return_mse=True)
    """
    
    data = np.atleast_2d(np.array(np.squeeze([data])))
    N, n = data.shape
    
    if qoi_col is None:
        qoi_col = np.atleast_1d(n-1)
    if cond_cols is None:
        qoi_col = np.atleast_1d(qoi_col)
        cond_cols = np.array([i for i in range(n) if i not in qoi_col])
    
    qoi_col = np.atleast_1d(qoi_col)
    if qoi_col.shape[0] > 1:
        raise ValueError("Optimal bandwidth can be found for single QoI only.")
    
    xy_idx = np.hstack((np.atleast_1d(cond_cols), np.atleast_1d(qoi_col)))
    
    data = data[:, xy_idx]
    
    if split_seed is not None:
        shuffle = True
    
    h_opt_k = []
    mse_opt_k = []
    
    if verbose:
        print("OPTIMIZING BANDWIDTHS JOINTLY\n\n")
    
    start = _short_date()
    
    for k in range(kfolds):
        if verbose:
            print(f"Fold {k+1}/{kfolds}\n")
        seed = split_seed
        if split_seed is not None:
            seed = k + split_seed
        model_data, train_data = train_test_split(
            data, test_size=split_frac, random_state=split_seed, shuffle=shuffle)
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
    
        fitness_args = (X_train, y_train, model_data, logscale)
        if logscale:
                bounds = [(np.log(ga_bounds[0]), np.log(ga_bounds[1])) for i in range(len(cond_cols))]
        else:
            bounds = [ga_bounds for i in range(len(cond_cols))]
        
        if ga_workers != 1:
            updating='deferred'
        else:
            updating='immediate'
        result = differential_evolution(_fitness, bounds, args=fitness_args, 
                                        seed=opt_seed, workers=ga_workers, 
                                        polish=polish, updating=updating)
        
        values = np.exp(result.x).tolist() if logscale else result.x
    
        if verbose:
            # formatted_values = ", ".join([f'{x:.{print_precision}e}' for x in values])
            # formatted_values = ", ".join([f'{x:.{print_precision}f}' for x in values])
            formatted_values = ", ".join([f'{x:.{print_precision}f}e{_normalize_values(values)[1]}' for x in _normalize_values(values)[0]])
            print(f"Optimal bandwidths H = [{formatted_values}]")

            if result.fun > 1e-2:
                print(f"MSE = {result.fun:.{print_precision}f}")
            else:
                print(f"MSE = {result.fun:.{print_precision}e}")
            print("\n*************************************************\n")
        
        h_opt_k.append(values)
        mse_opt_k.append(result.fun)
        
    h_opt = np.array(h_opt_k).mean(axis=0)
    mse_opt = np.array(mse_opt_k).mean(axis=0)
    
    if verbose:
        formatted_h_opt = ", ".join([f'{x:.{print_precision}e}' for x in h_opt])
        formatted_h_opt = ", ".join([f'{x:.{print_precision}f}' for x in h_opt])
        formatted_h_opt = ", ".join([f'{x:.{print_precision}f}e{_normalize_values(h_opt)[1]}' for x in _normalize_values(h_opt)[0]])
        print(f"\nFinal bandwidth H = [{formatted_h_opt}]")

        if mse_opt > 1e-2:
            print(f"Final MSE = {mse_opt:.{print_precision}f}")
        else:
            print(f"Final MSE = {mse_opt:.{print_precision}e}")
        print(f"\nTime elapsed = {_short_date() - start}")
        
    return h_opt, mse_opt if return_mse else h_opt

###############################################################################
def conditioning_marginally_optimal_bw(
    data, cond_cols=None, qoi_col=None, split_frac=0.1, ranking_kfolds=1, 
    opt_kfolds=1, opt_cycles=2, split_seed=None, shuffle=True, optimizer='ga', 
    opt_seed=None, ga_bounds=(1e-09, 1e3), logscale=False, ga_workers=1, polish=True, 
    return_mse=False, verbose=False):
    """
    Optimizes marginal bandwidths for conditional probability density estimation
    using cross-validation and differential evolution.

    This function performs a two-stage optimization process to first rank conditional
    bandwidths and then optimize them sequentially. It supports k-fold cross-validation
    for both ranking and optimizing the bandwidths and can shuffle the data for
    randomness in the training-test splitting.

    Parameters
    ----------
    data : np.array
        2D array of input data. Rows represent observations, and columns represent
        features, including conditional variables and the QoI.
    cond_cols : list, optional
        List of column indices in `data` corresponding to the conditional variables.
        If `None`, all columns except the QoI are used.
    qoi_col : int, optional
        Index of the column in `data` representing the QoI. If `None`, the last
        column is used as the QoI.
    split_frac : float, optional
        Fraction of the data to use as the test set during each fold of cross-validation.
        Default is 0.1.
    ranking_kfolds : int, optional
        Number of k-fold cross-validation splits for ranking bandwidths. Default is 1.
    opt_kfolds : int, optional
        Number of k-fold cross-validation splits for optimizing bandwidths. Default is 1.
    opt_cycles : int, optional
        Number of optimization cycles to refine the bandwidths. Default is 2.
    split_seed : int, optional
        Random seed for shuffling the data during train-test splitting. Default is `None`.
    shuffle : bool, optional
        Whether to shuffle the data before splitting into training and test sets.
        Default is True.
    optimizer : str, optional
        Optimization algorithm to use. Default is `'ga'` (genetic algorithm via
        `differential_evolution`).
    opt_seed : int, optional
        Random seed for the optimizer. Default is `None`.
    ga_bounds : tuple, optional
        Bounds for the bandwidth search space during optimization. Default is (1e-09, 1e3).
    logscale : bool, optional
        If True, perform optimization in log-space for the bandwidths. Default is False.
    ga_workers : int, optional
        Number of parallel workers for the optimizer. Default is 1.
    polish : bool, optional
        If True, polish the final result of `differential_evolution` for more refined solutions.
        Default is True.
    return_mse : bool, optional
        If True, return the mean squared error (MSE) along with the optimized bandwidths.
        Default is False.
    verbose : bool, optional
        If True, prints detailed logs during the optimization process. Default is False.

    Returns
    -------
    h_opt_final : np.array
        The final optimized bandwidths for the conditional variables.
    mse_opt_final : float, optional
        The final mean squared error (MSE) after bandwidth optimization. This is returned 
        only if `return_mse` is set to True.

    Raises
    ------
    ValueError
        If more than one QoI is provided.

    Notes
    -----
    - The function assumes that only one QoI (Quantity of Interest) is provided. If
      multiple QoIs are passed, it raises a ValueError.
    - The optimization process is performed sequentially, which helps in optimizing
      bandwidths more efficiently for a large number of conditional variables.
    - Cross-validation and data shuffling are used to ensure that the results are
      generalizable and robust.
    
    Workflow
    --------
    1. Data Preprocessing:
       - The input data is converted to a 2D array, and if no QoI is specified, the last column
         is used as the QoI. The remaining columns are used as conditional variables.
    
    2. Bandwidth Ranking (K-Fold Cross-Validation):
       - For each ranking fold, the data is split into training and model datasets.
       - For each conditional variable, bandwidth is optimized using the training set, and the
         corresponding MSE is calculated. This step ranks the variables based on their influence.
    
    3. Sequential Bandwidth Optimization:
       - After ranking, the bandwidths are optimized sequentially in multiple cycles and k-folds.
       - During each cycle, each conditional variable’s bandwidth is optimized while keeping others
         fixed. The process repeats for a specified number of optimization cycles.
    
    4. Results:
       - The final optimized bandwidths are returned, optionally with the final MSE, after averaging
         across k-folds and optimization cycles.

    Example
    -------
    >>> # Example data: 100 samples with 3 conditional variables and 1 QoI
    >>> data = np.random.randn(100, 4)
    >>> # Run the marginal bandwidth optimization with default settings
    >>> h_opt, mse_opt = conditioning_marginally_optimal_bw(data, return_mse=True, verbose=True)
    >>> print("Optimized Bandwidths:", h_opt)
    >>> print("Final MSE:", mse_opt)
    """
    
    data = np.atleast_2d(np.array(np.squeeze([data])))
    N, n = data.shape
    
    if qoi_col is None:
        qoi_col = np.atleast_1d(n-1)
    if cond_cols is None:
        qoi_col = np.atleast_1d(qoi_col)
        cond_cols = np.array([i for i in range(n) if i not in qoi_col])
    
    qoi_col = np.atleast_1d(qoi_col)
    if qoi_col.shape[0] > 1:
        raise ValueError("Optimal bandwidth can be found for single QoI only.")
    
    xy_idx = np.hstack((np.atleast_1d(cond_cols), np.atleast_1d(qoi_col)))
    
    data = data[:, xy_idx]
    
    if split_seed is not None:
        shuffle = True
    
    h_opt_full_k = []
    mse_opt_full_k = []

    if verbose:
        print("OPTMIZING SINGLE BANDWIDTHS FOR RANKING PURPOSES\n")
        print("******************\n")
    for k in range(ranking_kfolds):
        if verbose:
            print(f"Ranking fold {k+1}/{ranking_kfolds}\n")
        seed = split_seed
        if split_seed is not None:
            seed = k + split_seed
        model_data, train_data = train_test_split(data, test_size=split_frac, random_state=seed, shuffle=True)
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        
        h_opt_full = []
        mse_opt_full = []
        
        for cond_col_idx in range(len(cond_cols)):
            cond_col = cond_cols[cond_col_idx]
            if verbose:
                print(f"Optimizing bandwidth for variable {cond_col}")
            fitness_args = (X_train[:, cond_col_idx].reshape(-1, 1), y_train, model_data[:, [cond_col_idx, -1]], logscale)
            
            if logscale:
                bounds = [(np.log(ga_bounds[0]), np.log(ga_bounds[1]))]
            else:
                bounds = [ga_bounds]
            result = differential_evolution(_fitness, bounds, args=fitness_args, seed=opt_seed, workers=ga_workers, polish=polish, strategy='randtobest1bin')
            
            if logscale:
                h_opt = np.exp(result.x)
            else:
                h_opt = result.x
            h_opt_full.append(h_opt)
            mse_opt = [result.fun]
            mse_opt_full.append(mse_opt)
            if verbose:
                print(f"H_{cond_col} =", h_opt)
                print("MSE (from optimizer) =", round(mse_opt[0], 3), "\n")

        h_opt_full_k.append(h_opt_full)
        mse_opt_full_k.append(mse_opt_full)
        
        if verbose:
            print("*************************************************\n")

    h_opt_full_k = np.array(h_opt_full_k).mean(axis=0)
    mse_opt_full_k = np.array(mse_opt_full_k).mean(axis=0)
    
    if verbose:
        print("K-fold averaged single marginally optimal bandwidths:", h_opt_full_k[:, 0].tolist())
        print("K-fold averaged single marginally optimal MSE:", mse_opt_full_k[:, 0].tolist())
    
    h_opt_marg = np.array(h_opt_full_k).reshape(-1,)
    mse_opt = np.array(mse_opt_full_k).reshape(-1,)
    h_mse = np.array([range(len(h_opt_marg)), h_opt_marg, mse_opt]).T
    ind = np.argsort(h_mse[:, 2])
    sorted_h_mse = h_mse[ind]
    cond_cols_sorted_idx = sorted_h_mse[:, 0].astype(int).tolist()
    if verbose:
            print("\n******************************************************\n")


    #-# optimizing ranked bandwidths
    
    if verbose:
        print('\n\n\nOPTIMIZING RANKED BANDWIDTHS SEQUENTIALLY\n\n')
    # h_opt = sorted_h_mse[:, 1]
    if logscale:
        h_opt = np.ones_like(cond_cols_sorted_idx) * np.log(ga_bounds[1])
    else:
        h_opt = np.ones_like(cond_cols_sorted_idx) * ga_bounds[1]

    total_runs = opt_cycles * opt_kfolds * len(cond_cols)
    counter = 0

    for i_ in range(len(cond_cols) * opt_cycles):
        i = i_ % len(cond_cols)
        cond_col_idx = cond_cols_sorted_idx[i]
        # fixed_values = [h for h in h_opt]
        # fixed_values[i] = None
        # _fitness(h, X_train, y_train, model_data, h_full=None, h_idx=None, verbose=False)

        h_opt_k = []
        mse_opt_k = []
        for k in range(opt_kfolds):
            counter += 1
            seed = split_seed
            if split_seed is not None:
                seed = 2*k + split_seed
            model_data, train_data = train_test_split(data, test_size=split_frac, random_state=seed, shuffle=True)
            X_train = train_data[:, :-1]
            y_train = train_data[:, -1]

            if verbose:
                print(f"{counter}/{total_runs}\nOptimizing bandwidth for variable {cond_cols_sorted_idx[i]} ({i+1}/{len(cond_cols)}), Cycle {i_ // len(cond_cols)+1}/{opt_cycles}, fold = {k+1}/{opt_kfolds}:")
            
            fitness_args = (X_train, y_train, model_data, logscale, np.copy(h_opt), cond_col_idx)
            
            if logscale:
                bounds = [(np.log(ga_bounds[0]), np.log(ga_bounds[1]))]
            else:
                bounds = [ga_bounds]
            
            result = differential_evolution(_fitness, bounds, args=fitness_args, seed=opt_seed, workers=ga_workers, polish=polish, strategy='randtobest1bin')
            
            h_opt_updated = np.copy(h_opt)
            h_opt_updated[cond_col_idx] = result.x
            h_opt_k.append(h_opt_updated)
            mse_opt = [result.fun]
            mse_opt_k.append(mse_opt)
            if verbose:
                print(f"H =", np.exp(h_opt_updated) if logscale else np.array(h_opt_updated))
                print("MSE =", round(mse_opt[0], 6), "\n")
        h_opt_k = np.array(h_opt_k).mean(axis=0)
        h_opt = h_opt_k
        mse_opt_k = np.array(mse_opt_k).mean()
        mse_opt = float(mse_opt_k)
        if verbose:
            print("Average fold H =", np.exp(h_opt) if logscale else np.array(h_opt))
            print("Average fold MSE =", round(mse_opt, 6))
            print("\n*************************************************\n")

    h_opt_final = h_opt
    mse_opt_final = mse_opt
    
    h_opt_final = np.exp(h_opt_final) if logscale else np.array(h_opt_final)
    h_opt_final = h_opt_final.reshape(-1,)
    
    if verbose:
        print("\nFinal bandwidth H =", h_opt_final)
        print("\nFinal MSE =", mse_opt_final)
    
    return h_opt_final, mse_opt_final
    
###############################################################################
def rank_features(
    data, cond_cols=None, qoi_col=None, split_frac=0.1, ranking_kfolds=1, 
    split_seed=None, shuffle=True, optimizer='ga', ga_bounds=(1e-09, 1e3), 
    logscale=False, ga_workers=1, polish=True, method='expectation_mse', 
    opt_seed=None, verbose=False):
    
    data = np.atleast_2d(np.array(np.squeeze([data])))
    N, n = data.shape
    
    if qoi_col is None:
        qoi_col = np.atleast_1d(n-1)
    if cond_cols is None:
        qoi_col = np.atleast_1d(qoi_col)
        cond_cols = np.array([i for i in range(n) if i not in qoi_col])
    
    qoi_col = np.atleast_1d(qoi_col)
    cond_cols = np.atleast_1d(cond_cols)
    if qoi_col.shape[0] > 1:
        raise ValueError("Optimal bandwidth can be found for single QoI only.")
    
    xy_idx = np.hstack((cond_cols, qoi_col))
    
    data = data[:, xy_idx]
    
    if split_seed is not None:
        shuffle = True
    
    if verbose:
        print("RANKING FEATURES\n")
        print("******************\n")
    
    if method == 'expectation_mse':
        h_opt_full_k = []
        mse_opt_full_k = []
        
        for k in range(ranking_kfolds):
            if verbose:
                print(f"Ranking fold {k+1}/{ranking_kfolds}\n")
            seed = split_seed
            if split_seed is not None:
                seed = k + split_seed
            model_data, train_data = train_test_split(data, test_size=split_frac, random_state=seed, shuffle=shuffle)
            X_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            
            # h_opt_full = []
            # mse_opt_full = []
            mse_opt_full = np.ones(len(cond_cols)) * np.inf
            h_opt_full = np.zeros(len(cond_cols))
            
            for cond_col_idx in range(len(cond_cols)):
                cond_col = cond_cols[cond_col_idx]
                if verbose:
                    print(f"Optimizing bandwidth for variable {cond_col}")
                fitness_args = (X_train[:, cond_col_idx].reshape(-1, 1), y_train, model_data[:, [cond_col_idx, -1]], logscale)
                
                if logscale:
                    bounds = [(np.log(ga_bounds[0]), np.log(ga_bounds[1]))]
                else:
                    bounds = [ga_bounds]
                result = differential_evolution(_fitness, bounds, args=fitness_args, seed=opt_seed, workers=ga_workers, polish=polish, strategy='randtobest1bin')
                
                if logscale:
                    h_opt = np.exp(result.x)
                else:
                    h_opt = result.x
                mse_opt = result.fun
                
                if mse_opt < mse_opt_full[cond_col_idx]:
                    mse_opt_full[cond_col_idx] = mse_opt
                    h_opt_full[cond_col_idx] = h_opt
                if verbose:
                    print(f"H_{cond_col} =", h_opt)
                    print("MSE (from optimizer) =", round(mse_opt[0], 3), "\n")

            # h_opt_full_k.append(h_opt_full)
            # mse_opt_full_k.append(mse_opt_full)

            
            if verbose:
                print("*************************************************\n")

        # h_opt_full_k = np.array(h_opt_full_k).mean(axis=0)
        # mse_opt_full_k = np.array(mse_opt_full_k).mean(axis=0)
        h_opt_full_k = np.array(h_opt_full)
        mse_opt_full_k = np.array(mse_opt_full)
        
        if verbose:
            print("K-fold best single marginally optimal bandwidths:", h_opt_full_k[:, 0].tolist())
            print("K-fold best single marginally optimal MSE:", mse_opt_full_k[:, 0].tolist())
        
        h_opt_marg = np.array(h_opt_full_k).reshape(-1,)
        mse_opt = np.array(mse_opt_full_k).reshape(-1,)
        h_mse = np.array([cond_cols, h_opt_marg, mse_opt]).T
        ind = np.argsort(h_mse[:, 2])
        sorted_h_mse = h_mse[ind]
        features_idx_sorted = sorted_h_mse[:, 0].astype(int).tolist()
        criteria_sorted = sorted_h_mse[:, 2]
        model_params_sorted = sorted_h_mse[:, 1]
        if verbose:
                print("\n******************************************************\n")
    
    return features_idx_sorted, criteria_sorted, model_params_sorted

###############################################################################
def conditioning_silverman_bw(data, cond_cols, qoi_col, return_mse=False, 
                              split_frac=0.1, split_seed=None):
    """
    Computes the Silverman's bandwidth for the given data and conditional variables,
    optionally returning the mean squared error (MSE) based on a train-test split.

    This function estimates the bandwidth for a kernel density estimation 
    using Silverman's rule of thumb. If requested, it will also compute the MSE 
    by training a model on a subset of the data.

    Parameters
    ----------
    data: ndarray of shape (n_samples, n_features)
        The dataset used for bandwidth estimation and MSE calculation, where 
        n_samples is the number of data points and n_features is the number of features.

    cond_cols: int or array-like
        Indices of the conditional columns to consider for bandwidth estimation.

    qoi_col: int or array-like
        Index of the quantity of interest (model output) column.

    return_mse: bool, optional (default=False)
        If True, the function will return the mean squared error along with the bandwidth.

    split_frac: float, optional (default=0.1)
        Fraction of the data to use for testing when calculating MSE, must be between 0.0 and 1.0.

    split_seed: int, optional
        Seed for the random number generator used for splitting the data into training and testing sets.
        This ensures reproducibility of the split when specified.

    Returns
    -------
    h: float
        The computed Silverman's bandwidth.

    mse: float, optional
        The mean squared error of the model trained on the training data, 
        returned only if `return_mse` is True.
    """
    
    Nsim = data.shape[0]
    nw = len(np.atleast_1d(cond_cols))
    nq = len(np.atleast_1d(qoi_col))
    h = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq))
    
    if return_mse:
        xy_idx = np.hstack((np.atleast_1d(cond_cols), np.atleast_1d(qoi_col)))
        data = data[:, xy_idx]
        if split_seed is not None:
            model_data, train_data = train_test_split(data, test_size=split_frac,
                                                      random_state=split_seed,
                                                      shuffle=True)
        else:
            model_data, train_data = train_test_split(data, test_size=split_frac,
                                                      random_state=None, 
                                                      shuffle=False)
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        mse = _fitness(h, X_train, y_train, model_data)
    
    return h, mse if return_mse else h
    
###############################################################################
def train_test_split(data, test_size=0.1, random_state=None, shuffle=True):
    """
    Splits the dataset into training and testing subsets based on the specified test size.

    This function randomly divides the input dataset into two parts: a training set and a testing set.
    The training set is used to train machine learning models, while the testing set is used for evaluating their performance.

    Parameters
    ----------
    data: array-like or pandas DataFrame
        The dataset to split, which can be either a NumPy array or a pandas DataFrame.

    test_size: float, optional (default=0.1)
        Proportion of the dataset to include in the test split, should be between 0.0 and 1.0.
        For example, if 0.2 is passed, it will allocate 20% of the data to the test set.

    random_state: int, optional
        Controls the shuffling applied to the data before splitting. 
        Pass an integer to ensure reproducibility of the split across multiple runs.

    shuffle: bool, optional (default=True)
        Whether or not to shuffle the data before splitting. If set to False, the data will be split in its original order.

    Returns
    -------
    train_data: array-like or pandas DataFrame
        The training subset of the data.

    test_data: array-like or pandas DataFrame
        The testing subset of the data.
    """
    
    # Set the random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate the number of test samples
    n_samples = len(data)
    n_test = int(np.floor(test_size * n_samples))
    n_train = n_samples - n_test
    
    # Generate a shuffled array of indices
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    # Split the indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
        
    # Use the indices to split the data
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    return train_data, test_data

###############################################################################
def _conditional_pdf(X, qoi_cols, cond_cols, cond_vals, weights=None, grid=None,
                     sw=None, bw_opt_kwargs={}, sq=None, pdf_Npts=200, 
                     parallel=True, verbose=True):
    
    start = _short_date()
    if verbose:
        print('\n***********************************************************')
        print('Conditioning starting at', start)
    
    q = X[:, qoi_cols]
    Nsim = X.shape[0]
    nw = np.atleast_1d(cond_cols).shape[0]
    nq = np.atleast_1d(qoi_cols).shape[0]
    
    if verbose:
        print(f'\nEstimating the {"marginal" if nq==1 else "joint"} \
distribution of <variable{"" if nq==1 else "s"} {qoi_cols}> conditioned on \
<variable{"" if nw==1 else "s"} {cond_cols}> = <{cond_vals}>.')
        print(f'Using N = {Nsim} samples.')
    
    ## Conditioning weights
    if weights is None:
        if type(sw) is str and sw.startswith("optimal"):
            if nq > 1:
                print("Optimal bandwidth can be found for single QoI only. Using Silverman's bandwidth instead.")
                sw = None
            else:
                if verbose:
                    print("\nFinding optimal bandwidth for conditioning.")
                if sw == "optimal_joint":
                    sw = conditioning_jointly_optimal_bw(
                        X, cond_cols, qoi_cols, verbose=verbose,
                        **bw_opt_kwargs)
                elif sw == "optimal_marg":
                    sw = conditioning_marginally_optimal_bw(
                        X, cond_cols, qoi_cols, verbose=verbose,
                        **bw_opt_kwargs)
                else:
                    if verbose:
                        print("Invalid option for bandwidth. Using Silverman's bandwidth.")
                    sw = None
        if sw is None:
            if verbose:
                print("Using Silverman's bandwidth.")
            sw = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq))
        if verbose:
            print("\nComputing conditioning weights.")
            print(f'Using bw = {sw} for conditioning weights.')
        weights = _get_conditional_weights(X[:, cond_cols], cond_vals, sw,
                                           verbose=verbose)
    else:
        print("\nUsing user-specified weights.")
    Nsim = np.sum(weights)**2 / np.sum(weights**2)
    
    ## PDF evaluation grid
    if grid is None:
        refine_grid = True
        if nq == 1:
            x_range = np.max(q) - np.min(q)
            grid = np.linspace(np.min(q) - 0.2*x_range, np.max(q)+0.2*x_range, pdf_Npts)
        else:
            axes_pts = np.linspace(np.min(q, axis=0), np.max(q, axis=0), 
                                   pdf_Npts, axis=0)
            grid = np.asarray(np.meshgrid(*axes_pts.T))
            grid = grid.reshape(nq, -1).T
        if verbose:
            print(f'\nGenerating PDF evaluation grid from data ({pdf_Npts} \
pts per dimension, {grid.shape[0]} points in total).')
    else:
        grid = np.asarray(grid)
        if verbose:
            print(f'\nUsing specified grid for PDF evaluation \
({grid.shape[0]} points).')

    ## KDE bandwidth
    if sq is None:
        q_std = np.atleast_2d(np.cov(q, bias=False, aweights=weights)**(0.5))
        nw = 0
        # sq = np.asarray((4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq)) * q_std)
        sq = (4 / (Nsim*(2+nw+nq))) ** (1/(4+nw+nq)) * q_std
        if sq.ndim == 0:
            H = np.atleast_2d(sq**2)
        else:
            # H = np.diag(sq**2)
            H = sq**2
        if verbose:
            print(f'\nComputing kernel bandwidth{"" if nq==1 else "s"} using \
Silverman\'s rule of thumb.')
            with np.printoptions(precision=6):
                print(f'Bandwidth used = {np.diag(np.sqrt(H)) / q_std}')
    else:
        q_std = np.atleast_2d(np.cov(q, bias=False, aweights=weights)**(0.5))
        # sq = np.asarray(sq) * q_std
        sq = sq * q_std
        H = sq ** 2
#         ## if H is specified as a scalar, i.e. 1-D pdf or isotropic n-D pdf
#         if sq.ndim == 0:
#             H = np.eye(nq)*(sq**2)
#         ## if H is specified as a list of n BWs, i.e. non-isotropic n-D pdf
#         elif sq.ndim == 1:
#             if sq.shape[0] == 1: ## assume same bw for all variables
#                 H = np.eye(nq)*(sq**2)
#             else:
#                 if sq.shape[0] != nq:
#                     raise ValueError("Number of specified anisotropic \
# bandwidths must be equal to the number of conditioned variables (QoIs).")
#                 H = np.diag(sq**2)
        if verbose:
            print(f'\nUsing specified kernel bandwidth{"s" if nq==1 else ""}.')
            print(f'Bandwidth used = {np.diag(np.sqrt(H)) / q_std}')
    
    ## KDE evaluation
    if parallel:
        if verbose:
            print(f'\nEvaluating KDE on {grid.shape[0]} points in parallel.')
        result = Parallel(n_jobs=-1)(
            delayed(_evaluate_kernels_sum)(q, grid[i], H, weights) 
            for i in range(len(grid)))
    else:
        if verbose:
            print(f'\nEvaluating KDE on {grid.shape[0]} points.')
        result = []
        for x in grid:
            result.append(_evaluate_kernels_sum(q, x, H, weights))
    
    if refine_grid:
        try:
            result = np.array(result)
            pdf_max = np.max(result[:, 1])
            i_ = 0
            for i in range(result.shape[0]):
                if result[i, 1] > 0.000001*pdf_max:
                    xmin = result[i, 0]
                    break
            for j in range(result.shape[0]-1, i, -1):
                if result[j, 1] > 0.000001*pdf_max:
                    xmax = result[j, 0]
                    break
            grid = np.linspace(xmin, xmax, pdf_Npts)

            if parallel:
                if verbose:
                    print(f'\nEvaluating KDE on {grid.shape[0]} points in parallel.')
                result = Parallel(n_jobs=-1)(
                    delayed(_evaluate_kernels_sum)(q, grid[i], H, weights) 
                    for i in range(len(grid)))
            else:
                if verbose:
                    print(f'\nEvaluating KDE on {grid.shape[0]} points.')
                result = []
                for x in grid:
                    result.append(_evaluate_kernels_sum(q, x, H, weights))       
        except:
            pass

    end = _short_date()
    if verbose:
        print('\nConditioning complete at', end)
        print('Time =', end-start)
    
    return np.array(result)

###############################################################################
def conditional_pdf(obj, qoi_cols, cond_cols, cond_vals, weights=None, grid=None, 
                    sw=None, bw_opt_kwargs={}, sq=None, pdf_Npts=200, 
                    parallel=True, verbose=True):
    
    args = {'qoi_cols': qoi_cols, 'cond_cols': cond_cols, 'cond_vals': cond_vals,
            'weights': weights, 'grid': grid, 'sw': sw, 'bw_opt_kwargs': bw_opt_kwargs, 
            'sq': sq, 'pdf_Npts': pdf_Npts, 'parallel': parallel, 'verbose': verbose}
    
    if isinstance(obj, dict):
        data = obj['data']['augmented']
    elif isinstance(obj, np.ndarray):
        data = obj
    else:
        raise ValueError("Invalid type for 'obj'. Expected dict or np.ndarray.")
    
    if isinstance(obj, dict):
        pdf = _conditional_pdf(data, **args)
    else:
        pdf = _conditional_pdf(data, **args)
    return pdf

###############################################################################

### NOTES

### - run marginal optimization with minimize (might be convex)+
### - during cycles, track change in mse. if change is small, drop variable from optimization list
### - parallelize ranking optimization (also use minimize?)
### - projection pursuit
### - optimize in groups
### - use mse cutoff to reduce space
### - compare with silverman as baseline