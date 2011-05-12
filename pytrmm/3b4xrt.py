__all__ = ['TRMM3B42RTFile', 'read3B42RT']

import sys
from gzip import GzipFile

import numpy as np
from numpy import ma

def read3B42RT(fname):
    precip_scale_factor = 100.0
    rows = 480
    cols = 1440

    fp = open(fname, 'rb')
    data_string = fp.read()
    fp.close()

    precip = np.fromstring(data_string[2880:1385280], np.int16)
    precip = precip.byteswap()
    precip = np.asarray(precip, np.float32)
    precip /= precip_scale_factor
    precip = ma.masked_less_equal(precip, 0)
    precip = precip.reshape(rows, cols)

    return precip

class TRMM3B42RTFile:
    """Class for read operations on TRMM 3B42RT files.

    Example Usage:

    >>> from dataflow.trmm import TRMM3B42RTFile
    >>> current_file = TRMM3B42RTFile(file_name)
    >>> precip = current_file.precip()
    >>> print 'Array dimensions:', precip.shape
    >>> print 'Data max:', precip.max()
    >>> print 'Data min:', precip.min()
    >>> print 'Data mean:', precip.mean()
    >>> print 'Data std-dev:', precip.std()

    """
    def __init__(self, file_name):
        self.fname = file_name
        self.info = dict(cols=1440,
                         rows=480,
                         ll_lon=0.125,
                         ll_lat=-59.875,
                         dlon=0.25,
                         dlat=0.25,
                         grid_size=0.25)

    def precip(self, scaled=True, floats=True, masked=True):
        """Return the entire field of rainfall values.

        The data are returned as a 2D Numpy array.

        """
        precip_scale_factor = 100.0
        rows = 480 # The file headers lie about number of rows
        cols = 1440

        if self.fname.split('.')[-1] == 'gz':
            fp = GzipFile(self.fname)
        else: # assume decompressed binary file
            fp = open(self.fname, 'rb')
        data_string = fp.read()
        fp.close()

        precip = np.fromstring(data_string[2880:1385280], np.int16)
        if sys.byteorder == 'little':
            precip = precip.byteswap()
        if floats:
            precip = np.asarray(precip, np.float32)
        if scaled:
            precip /= precip_scale_factor
        if masked:
            precip = ma.masked_less(precip, 0)
        precip = precip.reshape(rows, cols)

        return precip

    def header(self, scaled=True, floats=True, masked=True):
        """Return the file header in a dictionary.

        """
        if self.fname.split('.')[-1] == 'gz':
            fp = GzipFile(self.fname)
        else: # assume decompressed binary file
            fp = open(self.fname, 'rb')
        data_string = fp.read()
        fp.close()

        hdr = {}
        for item in data_string[:2880].split():
            key, val = item.split('=')
            hdr[key] = val

        return hdr

    def point_values(self, lats, lons):
        """Get the rainfall value at one or more locations.

        A simple nearest neighbour algorithm is used with the returned value
        being the rainfall of the grid box containing the location(s).

        """
        ncols = self.info['cols']
        nrows = self.info['rows']
        lon0 = self.info['ll_lon']
        lat0 = self.info['ll_lat']
        dlon = self.info['dlon']
        dlat = self.info['dlat']

        row_indices, col_indices = find_indices(lats, lons, lat0, lon0,
                                                dlat, dlon, nrows, ncols)

        row_indices = np.asarray(row_indices)
        col_indices = np.asarray(col_indices)

        if row_indices.shape != () and col_indices.shape != ():# multiple points
            rindices = row_indices[(row_indices != -999)
                                   & (col_indices != -999)]
            cindices = col_indices[(row_indices != -999)
                                   & (col_indices != -999)]
        else:# scalar
            rindices = row_indices
            cindices = col_indices

        precip = self.precip()

        result = precip[rindices, cindices]

        return result

    def clip_precip(self, min_lat, max_lat, min_lon, max_lon):
        """Obtain a sub-region specified by a bounding box.

        A 2D masked array is returned. Only pixels with centres
        falling inside the bounding box are returned. If pixel centres
        fall exactly on the boundaries, these pixels are also
        included.

        """
        lon, lat = self._define_grid(min_lon, min_lat, max_lon, max_lat)

        rows = np.unique(lat).size
        cols = np.unique(lon).size

        result = self.point_values(lat, lon).reshape((rows, cols))
        
        return result

    def _define_grid(self, ll_lon, ll_lat, ur_lon, ur_lat):
        """Extract a sub-set of the TRMM 3B42RT grid."""
        cols = self.info['cols']
        rows = self.info['rows']
        x0 = self.info['ll_lon']
        y0 = self.info['ll_lat']
        cell_size = self.info['grid_size']

        # define the grid (pixel centre's)
        xt, yt = np.meshgrid(np.linspace(x0, x0 + (cols-1)*cell_size, num=cols),
                             np.linspace(y0 + (rows-1)*cell_size, y0, num=rows))

        xt = xt.flatten()
        yt = yt.flatten()

        # define points inside the specified domain
        lon = xt[(xt >= ll_lon) & (xt <= ur_lon) &
                  (yt >= ll_lat) & (yt <= ur_lat)]

        lat = yt[(xt >= ll_lon) & (xt <= ur_lon) &
                  (yt >= ll_lat) & (yt <= ur_lat)]

        return lon, lat
