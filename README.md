# ODEnlls

*ODEnlls* is a Python3 library for fitting chemical reaction kinetics data. It
accomplishes this task in two steps: 

1. Numerical simulations of the ordinary differential equations (ODE) for an
   arbitrary set of chemical reactions provide model data.

2. These simulated data are fit to fitting to the experimental kinetic data
   using non-linear least squares (nlls) methods.

In addition, you can use this library to simulate kintetic profiles for
arbitrary sets of chemical reactions, to get a sense of how they might behave
if observed.

## Dependencies 

This package consists of a single Python module file that was developed using
Python 3.6; however, it should work on most other Python 3 versions as long as
the external dependencies are filled. 

This module requires a number of external modules as well. The versions given
below are the ones that were tested during development. 

* Numpy >= 1.13.3
* scipy>=1.0.0
* pandas>=0.21.1
* matplotlib>=2.1.1

Older versions of these modules may work as well, but you may want to run the
[py.test] unit tests (*coming soon*) to ensure they work properly.

## Installation

At this point, you can only install this code via the GitHub repo. However,
some packaging options will be available shortly.

## Usage

Detailed documentation will be available soon. For the time being, use the
following [Jupyter] notebooks as a demonstration of the module capabilities.

[Notebook 1] demonstrates fitting of simple first-order irreversible reaction
kinetic data.
[Notebook 2] demonstrates a more complicated first-order reversible reaction
as well as simultaneous fitting capabilities for all kinetic data.


[py.test]: https://docs.pytest.org/en/latest/
[Jupyter]: http://jupyter.org/
