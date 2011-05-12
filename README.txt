===========================================
PyTRMM - Python tools for reading TRMM data
===========================================

PyTRMM is a BSD licensed Python library containing tools for reading
the data files produced by the Tropical Rainfall Measuring Mission
(TRMM - http://trmm.gsfc.nasa.gov/). At this stage the package has
tools to read the TRMM 3B4XRT data files, but the code should be
useful for reading other data provided by TRMM.

Download
--------

Source code can be obtained at http://github.com/sahg/pytrmm

Install
-------

The PyTRMM package currently consists of pure Python code and does not
require any extension modules to be built. However the model depends
on NumPy (http://www.numpy.org).

Once the Numpy has been been successfully installed, PyTRMM can be
installed at the command line for any operating system:

    $ python setup.py install

The installation can be tested in the Python interpreter:

    >>> import pytrmm
