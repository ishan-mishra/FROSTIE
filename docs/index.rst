.. FROSTIE documentation master file, created by
   sphinx-quickstart on Mon Jun 27 15:04:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FROSTIE's documentation!
===================================

FROSTIE is spectroscopic retrieval package, written in Python, for analyzing reflectance spectra of planetary surfaces. 'Spectroscopic retrieval' involves analyzing spectroscopic data from planetary surfaces and atmospheres using appropriate physical models and statistical tools. The goal is to infer properties like composition, temperature, etc. of the planetary body being observed. These methods can also be applied to simulated data from future missions/observatories, to help assess their science output and inform their design.

While FROSTIE has been only applied to data of Europa so far, with the right input files (i.e., optical constants of species of interest) it can be readily used to study any planetary surface in the solar system. FROSTIE's features currently include:

* Forward modelling of reflectance spectra using ``PyHapke``, a Python implementation of the popular Hapke bi-directional reflectance model `(Hapke (1981) <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JB086iB04p03039>`_).
* Support for one component or multi-component model using parameters like abundance, grain-size, and porosity.
* A Bayesian inference framework that uses the forward model to find the best model (i.e., the best set of species among the candidates) and derive probability distributions of parameters. 
* Plotting routines to instantly produce publication quality plots
* An interactive widget that allows the user to play with the forward model using sliders and buttons to change parameters.

FROSTIE is available under the BSD 3-Clause License. If you use FROSTIE, 
please cite `Mishra et al. (2022) <https://ui.adsabs.harvard.edu/abs/2021Icar..35714215M/abstract>`_. 

.. toctree::
   :maxdepth: 1
   :hidden:

   content/installation

.. toctree::
   :maxdepth: 2
   :caption: Guide:
   
   content/notebooks/one_component_model
   content/notebooks/multi_component_model
   content/notebooks/interactive_model
   
.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:
   
   autoapi/index


