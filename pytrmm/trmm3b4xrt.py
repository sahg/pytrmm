__all__ = ['TRMM3B42RTFile']

import sys
from gzip import GzipFile

import numpy as np
from numpy import ma

class TRMM3B4XRTFile:
    """Base Class for read operations on TRMM 3B4XRT files.

    This class should not be used directly, use one of the derived
    classes instead.

    """
    def __init__(self, filename):
        self.filename = filename
        self._header_offset = 2880

        self._read_header()

        self._rows = int(self._hdr['number_of_latitude_bins'])
        self._cols = int(self._hdr['number_of_longitude_bins'])

    def _read_binary(self):
        """Read file as a binary string.

        """
        if self.filename.split('.')[-1] == 'gz':
            fp = GzipFile(self.filename)
        else: # assume decompressed binary file
            fp = open(self.filename, 'rb')
        data_string = fp.read()
        fp.close()

        return data_string

    def _read_header(self):
        """Read the file header.

        """
        data_string = self._read_binary()

        self._hdr = {}
        for item in data_string[:self._header_offset].split():
            key, val = item.split('=')
            self._hdr[key] = val

    def _read_field(self, field_num):
        """Read a data field from the file.

        """
        dtype_list = self._hdr['variable_type'].split(',')
        dtype_list = [int(s[-1]) for s in dtype_list]

        nfields = int(self._hdr['number_of_variables'])

        if field_num in range(nfields):
            strt_offset = self._header_offset
            k = field_num - 1
            while k >= 0:
                strt_offset += dtype_list[k]*self._rows*self._cols
                k = k-1
        else:
            raise IOError("Can't read field number %d. File %s only contains %d fields, and fields are indexed from 0." \
                          % (field_num, self.filename, nfields))

        var_type = self._hdr['variable_type'].split(',')[field_num]

        if var_type == 'signed_integer1':
            dtype = np.int8
            end_offset = strt_offset + self._rows*self._cols
        elif var_type == 'signed_integer2':
            dtype = np.int16
            end_offset = strt_offset + 2*self._rows*self._cols
        else:
            raise IOError, 'Badly formed header in %s' % self.filename

        data_string = self._read_binary()

        field = np.fromstring(data_string[strt_offset:end_offset], dtype)
        if sys.byteorder == 'little':
            field = field.byteswap()

        field = field.reshape(self._rows, self._cols)

        return field

    def header(self):
        """Return a copy of the file header in a dictionary.

        """
        return dict(self._hdr)

class TRMM3B42RTFile(TRMM3B4XRTFile):
    """Class for read operations on TRMM 3B42RT files.

    Example Usage:

    >>> from pytrmm import TRMM3B42RTFile
    >>> trmm_file = TRMM3B42RTFile(file_name)
    >>> print(trmm_file.header())
    >>> precip = trmm_file.precip()
    >>> print 'Array dimensions:', precip.shape
    >>> print 'Data max:', precip.max()
    >>> print 'Data min:', precip.min()
    >>> print 'Data mean:', precip.mean()
    >>> print 'Data std-dev:', precip.std()

    """
    def precip(self, scaled=True, floats=True, masked=True):
        """Return the entire field of rainfall values.

        The data are returned as a 2D Numpy array.

        """
        precip_scale_factor = 100.0

        raw_field = self._read_field(0)

        if masked:
            precip = np.ma.masked_equal(raw_field,
                                        int(self._hdr['flag_value']))
        if floats:
            precip = np.ma.asarray(precip, np.float32)
        if scaled:
            precip /= precip_scale_factor

        return precip
