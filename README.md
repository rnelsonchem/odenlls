# ODEnlls

*ODEnlls* is a Python3 library for simulating and fitting chemical kinetics
data. These two pieces are accomplished as follows: 

1. Kinetic models are simulated using numerical simulations of the ordinary
   differential equations (ODE) for an arbitrary set of chemical reactions.
Rate constants and starting concentrations can be varied arbitrarily to
observe the predicted changes in concentration with time.

2. These ODE simulations are fit to experimental kinetic data using non-linear
   least squares (nlls) methods. These fits yield the best-fit rate constant
and concentration parameters for a given set of kinetic data.

## Dependencies 

This package consists of a single Python module file that was developed using
Python 3.6; however, it should work on most other Python 3 versions with the
appropriate external dependencies listed below. 

* Numpy >= 1.13.3
* scipy>=1.0.0
* pandas>=0.21.1
* matplotlib>=2.1.1

The package versions above were used during development. Older/newer versions
should work as well. Older versions of these modules may work as well, but you
may want to run the [py.test] unit tests (*coming soon*) to ensure they work
properly.

## Installation

*ODEnlls* is installable using either Python's `pip` package manager or
[`conda`], the package manager for the [Anaconda Python distribution].

To get the latest release using `pip`, use the following command:

    $ pip install ODEnlls

Or to install from the latest GitHub commit:

    $ pip install git+https://github.com/rnelsonchem/ODEnlls.git

Using `conda`, the following command will install the latest release of this
package.

    $ conda install -c rnelsonchem odenlls

## Usage

The *ODEnlls* module capabilities are demonstrated in several [Jupyter]
notebooks, which are located in the "examples" directory on the [GitHub
project page]. A summary of these notebooks is as follows:

* The [TLDR Notebook] is a very brief overview of *ODEnlls* functionality
  with very little explanatory text.

* [Notebook 1] demonstrates simulation of a simple first-order irreversible
  reaction.

* In [Notebook 2], reaction data fitting is shown for a user-generated set of
  first-order irreversible reaction data.

* [Notebook 3] highlights fitting of a real-world data set using a series of
  reversible first-order reactions.



[py.test]: https://docs.pytest.org/en/latest/
[Jupyter]: http://jupyter.org/
[`conda`]: https://conda.io/docs/
[Anaconda Python Distribution]: https://www.anaconda.com/download/
[GitHub project page]: https://github.com/rnelsonchem/ODEnlls
[Notebook 1]: https://github.com/rnelsonchem/ODEnlls/blob/master/examples/1.%20First%20order%20irreversible%20kinetics%20simulation.ipynb 
[Notebook 2]: https://github.com/rnelsonchem/ODEnlls/blob/master/examples/2.%20First%20order%20irreversible%20kinetics%20fitting.ipynb
[Notebook 3]: https://github.com/rnelsonchem/ODEnlls/blob/master/examples/3.%20First%20order%20reversible%20kinetics%20simulation%20and%20fitting.ipynb
[TLDR Notebook]: https://github.com/rnelsonchem/ODEnlls/blob/master/examples/TLDR.ipynb
