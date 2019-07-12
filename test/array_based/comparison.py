"""
SimpleDerts
-------
Compute gradient of 2D input array.

See frame_2D_alg/frame_blobs.py for more information.
"""

import numpy as np

from utils import imread, shrunk, kernel

# -----------------------------------------------------------------------------
# Module constants

# -----------------------------------------------------------------------------
# SimpleDerts Class

class SimpleDerts(object):
    """
    Read an image and compute the derivatives of its data.
    Currently only read image as gray-scale.

    Parameters
    ----------
    input : ndarray
        An input array, from which derivatives are computed.
    k2x2 : bool, optional
        If =True use 2x2 kernel else 3x3.
    """

    def __init__(self, inputs, k2x2=False):
        """
        Create an instance of SimpleDerts, load input from file.

        Parameters
        ----------
        input : ndarray
            An input array, from which derivatives are computed.
        k2x2 : bool, optional
            If =True use 2x2 kernel else 3x3.
        """

        assert (isinstance(inputs, np.ndarray)
                and len(inputs.shape) == 2), "Input must be a 2D array!"

        # Attribute initializations:
        if k2x2:
            shape = shrunk(inputs.shape, 1)
            kernel_size = 2 # Assign kernel size.
        else:
            shape = shrunk(inputs.shape, 2)
            kernel_size = 3  # Assign kernel size.

        self.kernel = kernel(kernel_size) # Generate kernel.
        self.data = np.empty((4,) + shape)

        # Assign p:
        if k2x2:
            self.data[0] = (inputs[:-1, :-1]
                            + inputs[:-1, 1:]
                            + inputs[1:, 1:]
                            + inputs[1:, :-1])
        else:
            self.data[0] = (inputs[:-2, 1:-1]
                            + inputs[1:-1, 2:]
                            + inputs[2:, 1:-1]
                            + inputs[1:-1, :-2])

        self._generate_derivatives(inputs)

    # -------------------------------------------------------------------------
    # Properties

    @property
    def shape(self):
        """Return shape of data."""
        return self.data.shape

    @property
    def ksize(self):
        """Return kernel size."""
        return self.kernel.shape[1]

    @property
    def i(self):
        return self.data[0]

    @property
    def g(self):
        return self.data[1]

    @property
    def dy(self):
        return self.data[2]

    @property
    def dx(self):
        return self.data[3]

    # -------------------------------------------------------------------------
    # Methods

    def _generate_derivatives(self, inputs):
        """Generate derivatives from inputs."""

        # Convolve inputs with kernels:
        for y in range(self.shape[1]):
            for x in range(self.shape[2]):
                # Convolve then sum across the two last dimensions:
                self.data[2:, y, x] = (inputs[y : y+self.ksize,
                                              x : x+self.ksize]
                                       * self.kernel).sum(axis=(1, 2))

        # Compute into gradient magnitudes:
        self.data[1] = np.hypot(self.data[2], self.data[3])

# -----------------------------------------------------------------------------
# Functions

def from_image(path, **kwargs):
    """Initialize and return a SimpleDerts object."""

    assert isinstance(path, str), "Path must be a string!"
    # Read input from file:
    try:
        inputs = imread(path)
    except:
        print('Cannot load specified image!')
        return None

    return SimpleDerts(inputs, **kwargs)

def from_array(a, **kwargs):
    """Create SimpleDerts object from array-like input."""

    return SimpleDerts(np.array(a), **kwargs)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------