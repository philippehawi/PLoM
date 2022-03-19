# PLoM
Probabilistic Learning on Manifolds

Open source Python package accompanying [Soize and Ghanem, 2016](https://doi.org/10.1016/j.jcp.2016.05.044) and [Soize and Ghanem, 2020](https://doi.org/10.48550/arXiv.2002.12653).

"The PLoM considers a given initial dataset constituted of a small number of points given in an Euclidean space, which are interpreted as independent realizations of a vector-valued random variable for which its non-Gaussian probability measure is unknown but is, a priori, concentrated in an unknown subset of the Euclidean space. The objective is to construct a learned dataset constituted of additional realizations that allow the evaluation of converged statistics."

## Installation
```bash
pip3 install .
```

## Dependencies
Python 3.7+ and the following packages are needed.

  Package     | Version  
--------------|-------------
  numpy       | >= 1.18.5
  matplotlib  | >= 3.4.2
  scipy       | >= 1.6.2
  joblib      | >= 1.0.1

To install the dependencies found in [requirements.txt](requirements.txt) before installing this package, please run the following command from the package root:

```shell
pip3 install -r requirements.txt
```

## How to use:
This package relies on a Python dictionary, referred to as plom solution dictionary, to set the job parameters and save all calculated quantities. This dictionary should first be created and populated with the appropriate keys and job parameters (using `sol_dict = plom.initialize(**args)`). Then, the algorithm is run using `plom.run(sol_dict)`.

### To run interactively or using a Python script:
1. Create a dictionary of job parameters. This dictionary should contain the training data. There are 2 ways (1.1 OR 1.2) to do this.

   1.1 Run the installed script, [`plom_make_input_template.py`](scripts/plom_make_input_template.py) to generate a [template PLoM input file](scripts/input_template.txt).
   ```bash
   plom_make_input_template.py
   ```
   In the input file, the `value` for `training` should be the path of the training data file.
   Then, parse the input file to generate the dictionary of job parameters using:
   ```python
   from plom import parse_input, initialize, run
   args = parse_input('input_template.txt')
   ```
   OR
   
   1.2 Load the training data and define a dictionary as follows:
   ```python
   training = np.loadtxt('training.txt')
   
   args = {
		  "training"          : training,

		  "scaling"           : True,
		  "scaling_method"    : 'Normalization',
		  # "scaling_method"    : MinMax,
		   
		  # # choose one pca_method and with option that directly follows it
		  "pca"               : True,
		  "pca_method"        : 'cum_energy',
		  "pca_cum_energy"    : 1,
		  # "pca_method"        : 'eigv_cutoff',
		  # "pca_eigv_cutoff"   : 0,
		  # "pca_method"        : 'pca_dim',
		  # "pca_dim"           : 1,
		  
		  "dmaps"             : True,
		  "dmaps_epsilon"     : 'auto', # 'auto', list, or float
		  "dmaps_kappa"       : 1,
		  "dmaps_L"           : 0.1,
		  "dmaps_first_evec"  : False,
		  "dmaps_m_override"  : 0,
		  "dmaps_dist_method" : 'standard',

		  "sampling"          : True,
		  "num_samples"       : 10,
		  "parallel"          : False,
		  "n_jobs"            : -1,
		  "save_samples"      : True,
		  "samples_fname"     : None, # if None, file will be named using job_desc and save time
		  "samples_fmt"       : 'npy', # 'npy' or 'txt'
		  
		  "projection"        : True,
		  "projection_source" : 'pca', # 'pca', 'scaling', or 'data'
		  "projection_target" : 'dmaps', # 'dmaps' or 'pca'
		  "ito_f0"            : 1,
		  "ito_dr"            : 0.1,
		  "ito_steps"         : 'auto', # int or 'auto'
		  "ito_pot_method"    : 2.1,
		  "ito_kde_bw_factor" : 1,
		  
		  "job_desc"          : "Job description",
		  "verbose"           : True
		  }
   ```

2. Create a PLoM solution dictionary containing the training data and job parameters. All calculated quantities will be saved in this dictionary.
    ```python
	solution_dict = initialize(**args)
	```

3. Run the full PLoM algorithm.
    ```python
	run(solution_dict)
	```

A `.txt` or `.npy` file containing the generated samples will saved if the flag `save_samples` is set to `True` in the parameters dictionary (or input file).

### To run from CLI:

1. Run the installed script, [`plom_make_input_template.py`](scripts/plom_make_input_template.py) to generate a [template PLoM input file](scripts/input_template.txt). 
   ```bash
   plom_make_input_template.py
   ```
   The `value` for `training` should be the path of the training data file.
   
2. Run the installed script, [`plom_run.py`](scripts/plom_run.py) from the command line.

   If run without an argument, the script will look for the input file `input.txt` under the current directory.
   ```bash
   plom_run.py
   ```
   Alternatively, the name of the input file can be specified:
   ```bash
   plom_run.py custom_input_file.txt
   ```

## Examples
* [Example 1: 2 circles in 2 dimensions](examples/circle2d/2circles_script.ipynb)
* [Example 2: spiral in 3 dimensions](examples/spiral3d/spiral_script.ipynb)
