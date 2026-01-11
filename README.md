# Topological Optimization of barcodes - Deforming Point Clouds to Control Persistent Homology

## Code environment setup

For setting up the code environment we suggest using conda, as it is the most convenient.

You can use the following commands for setting up the conda env with the needed dependencies:

```sh
conda create -n topological-optimization python=3.13 numpy scipy matplotlib.pyplot persim -y
conda activate topological-optimization
conda install -c conda-forge gudhi
pip install ripser
```

> [!Note]
> `ripser` might not be available in the default conda channels, so it's installed via pip.
> Although the code uses gudhi for computing the Rips complex, ripser contains a valuable function for plotting the persistence diagram.

## Code structure

The file `topological_optimizer.py` is the main part of the code and contains the algorithm for gradient descent feature manipulation, as well as the optimization towards a targed signature.

The rest of the files contain either helper methods and structures or visualizations.

## Animations

The animations and plots we obtained in our research are available under the `resources` folder.
