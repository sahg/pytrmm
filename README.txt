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
require any extension modules to be built. However the package depends
on NumPy (http://www.numpy.org).

Once Numpy has been been successfully installed, PyTRMM can be
installed at the command line for any operating system:

    $ python setup.py install

Usage Example
-------------

    >>> from pytrmm import TRMM3B42RTFile
    >>> trmm_file = TRMM3B42RTFile(file_name)
    >>> print(trmm_file.header())
    >>> precip = trmm_file.precip()
    >>> print('Array dimensions:', precip.shape)
    >>> print('Data max:', precip.max())
    >>> print('Data min:', precip.min())
    >>> print('Data mean:', precip.mean())
    >>> print('Data std-dev:', precip.std())

Contributions and bug reports
-----------------------------

If you have improvements to make to PyTRMM, or find a bug, we'd love
to hear about it. Please create an issue at
https://github.com/sahg/pytrmm/issues (currently requires a free
Github account). Alternatively make use of the PyTOPKAPI mailing list
(see http://sahg.github.com/PyTOPKAPI/contact.html).
