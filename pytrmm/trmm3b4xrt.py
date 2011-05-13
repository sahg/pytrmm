__all__ = ['TRMM3B40RTFile', 'TRMM3B41RTFile', 'TRMM3B42RTFile']

import sys
import warnings
from gzip import GzipFile

import numpy as np
from numpy import ma

class TRMM3B4XRTFile(object):
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

    def _read_scaled_masked_field(self, field_num, dtype=np.float32):
        """Return a scaled and masked data field.

        """
        scale_factor = float(self._hdr['variable_scale'].split(',')[field_num])

        raw_field = self.read_raw_field(field_num)

        field = np.ma.masked_equal(raw_field, int(self._hdr['flag_value']))
        field = np.ma.asarray(field, dtype)
        field /= scale_factor

        return field

    def read_raw_field(self, field_num):
        """Read a raw data field from the file.

        Reads the requested field from file if possible. The returned
        field is unscaled and unmasked integer data.

        Parameters
        ----------
        field_num : int
            The zero-indexed field number to read.

        Returns
        -------
        field : Numpy ndarray
            The unprocessed integer data contained in the file.

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

class TRMM3B40RTFile(TRMM3B4XRTFile):
    """Class for read operations on TRMM 3B40RT files.

    Example Usage:

    >>> from pytrmm import TRMM3B40RTFile
    >>> trmm_file = TRMM3B40RTFile(file_name)
    >>> print(trmm_file.header())
    >>> precip = trmm_file.precip()
    >>> print('Array dimensions:', precip.shape)
    >>> print('Data max:', precip.max())
    >>> print('Data min:', precip.min())
    >>> print('Data mean:', precip.mean())
    >>> print('Data std-dev:', precip.std())

    """
    def __init__(self, filename):
        TRMM3B4XRTFile.__init__(self, filename)

        if self._hdr['algorithm_ID'] != '3B40RT':
            algo_warning = """\
The file %s is apparently not a 3B40RT file.
Reported algorithm ID is %s. Try using pytrmm.TRMM%sFile instead.
"""
            algo_warning = algo_warning % (self.filename,
                                           self._hdr['algorithm_ID'],
                                           self._hdr['algorithm_ID'])

            warnings.warn(algo_warning)

    def precip(self):
        """Return the field of precipitation values.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(0)

    def precip_error(self):
        """Return the field of precipitation RMS error estimates.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(1)

    def total_pixels(self):
        """Return the field of total pixels.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(2, dtype=np.int8)

    def ambiguous_pixels(self):
        """Return the field of ambiguous pixels.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(3, dtype=np.int8)

    def rain_pixels(self):
        """Return the field of rain pixels.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(4, dtype=np.int8)

    def source(self):
        """Return the field of data source identifiers.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(5, dtype=np.int8)

class TRMM3B41RTFile(TRMM3B4XRTFile):
    """Class for read operations on TRMM 3B41RT files.

    Example Usage:

    >>> from pytrmm import TRMM3B41RTFile
    >>> trmm_file = TRMM3B41RTFile(file_name)
    >>> print(trmm_file.header())
    >>> precip = trmm_file.precip()
    >>> print('Array dimensions:', precip.shape)
    >>> print('Data max:', precip.max())
    >>> print('Data min:', precip.min())
    >>> print('Data mean:', precip.mean())
    >>> print('Data std-dev:', precip.std())

    """
    def __init__(self, filename):
        TRMM3B4XRTFile.__init__(self, filename)

        if self._hdr['algorithm_ID'] != '3B41RT':
            algo_warning = """\
The file %s is apparently not a 3B41RT file.
Reported algorithm ID is %s. Try using pytrmm.TRMM%sFile instead.
"""
            algo_warning = algo_warning % (self.filename,
                                           self._hdr['algorithm_ID'],
                                           self._hdr['algorithm_ID'])

            warnings.warn(algo_warning)

    def precip(self):
        """Return the field of precipitation values.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(0)

    def precip_error(self):
        """Return the field of precipitation RMS error estimates.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(1)

    def total_pixels(self):
        """Return the field of total pixels.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(2, dtype=np.int8)

class TRMM3B42RTFile(TRMM3B4XRTFile):
    """Class for read operations on TRMM 3B42RT files.

    Example Usage:

    >>> from pytrmm import TRMM3B42RTFile
    >>> trmm_file = TRMM3B42RTFile(file_name)
    >>> print(trmm_file.header())
    >>> precip = trmm_file.precip()
    >>> print('Array dimensions:', precip.shape)
    >>> print('Data max:', precip.max())
    >>> print('Data min:', precip.min())
    >>> print('Data mean:', precip.mean())
    >>> print('Data std-dev:', precip.std())

    """
    def __init__(self, filename):
        TRMM3B4XRTFile.__init__(self, filename)

        if self._hdr['algorithm_ID'] != '3B42RT':
            algo_warning = """\
The file %s is apparently not a 3B42RT file.
Reported algorithm ID is %s. Try using pytrmm.TRMM%sFile instead.
"""
            algo_warning = algo_warning % (self.filename,
                                           self._hdr['algorithm_ID'],
                                           self._hdr['algorithm_ID'])

            warnings.warn(algo_warning)

    def precip(self):
        """Return the field of precipitation values.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(0)

    def precip_error(self):
        """Return the field of precipitation RMS error estimates.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(1)

    def source(self):
        """Return the field of data source identifiers.

        The integer data are returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(2, dtype=np.int8)

    def uncalibrated_precip(self):
        """Return the field of uncalibrated precipitation values.

        The scaled data are in mm/hr and returned as a 2D masked Numpy
        array. Invalid data are masked out.

        """
        return self._read_scaled_masked_field(3)
