# Topological Optimization of barcodes - Deforming Point Clouds to Control Persistent Homology

## Code environment setup

For setting up the code environment we suggest using conda, as it is the most convenient.

You can use the following commands for setting up the conda env with the needed dependencies:

```sh
conda create -n topological-optimization python=3.13 numpy scipy -y
conda activate topological-optimization
conda install -c conda-forge ripser
conda install -c conda-forge gudhi
```

> [!NOTE]
> Note: `ripser` might not be available in the default conda channels, so it's installed via pip.
>

## Code structure
