"""
SimpleBlob
-------
Contain data of a blob. Building-block of SimpleBlobsCluster.

SimpleBlobsCluster
-------------
Convert lowest level data from an image into abstract data
potentially representing shapes and edges.

See frame_2D_alg/frame_blobs.py for more information.
"""

from itertools import starmap # For SimpleBlobsCluster's _form_blobs() method
from collections import namedtuple # For blueprint of SimpleBlob object

import numpy as np
import comparison

from scipy.ndimage import label, find_objects # For forming SimpleBlobs

# -----------------------------------------------------------------------------
# Module constants

ave = 30 # Initialize filter value for blobs partitioning.

# -----------------------------------------------------------------------------
# SimpleBlob namedtuple

#     Initialize blueprint of SimpleBlob objects:
#
#     Parameters
#     ----------
#     slice : tuple
#         A tuple containing the slice(start, stop, step) of the region
#         containing this blob.
#     mask : ndarray
#         2D Array where non-zero values are counted as part of
#         this blob.
#
#     (More attributes are going to be added in future versions)

# ----------------------------------------------------------------------
# ------------------------------------------------------------------------------
SimpleBlob = namedtuple('SimpleBlob',
             ['slice',
              'mask',
              ])

# -----------------------------------------------------------------------------
# SimpleBlobsCluster Class

class SimpleBlobsCluster(object):
    """
    Partitioning derts space into two regions by the (hyper) plane:

        g - Ave = 0

    With g as computed gradient magnitude of input at a given point.

    A blob is defined as a contiguous region that belongs to either,
    but not both, of those two regions.

    SimpleBlobsCluster distribute every derts into the blob that contains
    them in the image.

    Parameters
    ----------
    inp : ndarray, str or SimpleDerts
            Contain data or metadata of derts for forming blobs.
    Ave : int or float
        Value for separating derts values
    kwargs
        For other keyword-only arguments, see SimpleDerts.
    """

    def __init__(self, inp, Ave=None, **kwargs):
        """
        Create and initialize an instance of SimpleBlobs.

        Parameters
        ----------
        inp : ndarray, str or SimpleDerts
            Contain data or metadata of derts for forming blobs.
        Ave : int or float
            Value for separating derts values
        kwargs
            For other keyword-only arguments, see SimpleDerts.
        """

        # Initialize derts:
        if isinstance(inp, np.ndarray):
            self.derts = comparison.from_array(inp, **kwargs)
        elif isinstance(inp, str):
            self.derts = comparison.from_image(inp, **kwargs)
        elif isinstance(inp, comparison.SimpleDerts):
            self.derts = inp
        else:
            raise ValueError("Cannot read input of type "
                             + str(type(inp)) + "!!!")

        # Specify Ave:
        if Ave is None:
            self.Ave = ave * 8 / ((self.derts.ksize - 1) * 4)

        # Perform partition:
        self._form_blobs(self.derts.g > self.Ave)

    # -------------------------------------------------------------------------
    # Properties

    # -------------------------------------------------------------------------
    # Methods

    def _form_blobs(self, smap):
        """Generate blob objects from inputs."""

        # Create the container for SimpleBlobs objects:
        self.blobs = []
        self.num_blobs = 0

        # Separate positive and negative blobs, save into blobs list:
        for m in (smap, ~smap):
            labeled_smap, n = label(m)
            self.num_blobs += n # Add number of generated blobs to num_blobs.

            # Save information of generated blobs:
            slices = find_objects(labeled_smap)
            self.blobs.extend(starmap(
                lambda label, slice:
                    SimpleBlob(slice=slice,
                               mask=(labeled_smap == label)[slice],
                               ),
                enumerate(slices, start=1),
            ))

    def I(self, blob_index):
        """Return sum of p from index of a blob."""
        return self.derts.i[self.blobs[blob_index].slice].sum()

    def G(self, blob_index):
        """Return sum of g from index of a blob."""
        return self.derts.g[self.blobs[blob_index].slice].sum()

    def Dy(self, blob_index):
        """Return sum of dy from index of a blob."""
        return self.derts.dy[self.blobs[blob_index].slice].sum()

    def Dx(self, blob_index):
        """Return sum of dx from index of a blob."""
        return self.derts.dx[self.blobs[blob_index].slice].sum()


# ----------------------------------------------------------------------
# ------------------------------------------------------------------------------