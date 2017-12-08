MicroArray Analysis Vincent Dufour-Decieux - Shan X. Wang's lab
==================================

## Setup
1- Download [Anaconda](https://www.anaconda.com/download/), a free installer that includes Python and all the common scientific packages.

2- Create a conda environment with an Ipython kernel:

```
 conda create --name name_env python=3 ipykernel
```

3- Activate your conda environment:

```
source activate name_env
```
4- Install the depencies :

```
conda install -c astropy photutils
conda install -c ioam holoviews bokeh
conda install scikit-image==0.13.0
conda install pandas==0.20.3
conda install tifffile -c conda-forge
```
The following step is necessary in some computers:
```
pip uninstall Pillow
pip install Pillow
```

## Usage

Either use the Jupyter notebook:

cd to the notebook directory and lunch jupyter notebook:

```
jupyter notebook
```

## Contact
Cedric Espenel  
E-mail: espenel@stanford.edu
