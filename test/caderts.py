"""
CADerts
------
Compute derivations of an image's data.
"""

import numpy as np

from frame_2D_alg.utils import imread, shrunk, kernel

# ------------------------------------------------------------------------------
# Module constants

# ------------------------------------------------------------------------------
# CADerts Class

class CADerts:
    """
    Read an image and compute the derivations of its data.
    Currently only read image as grayscale.
    Parameters
    ----------
    path : String contain path to input file.
    k2x2 : If = 1 use 2x2 kernel else 3x3.
    """

    def __init__(self, path, k2x2=0):
        """
        Create an instance of CABlobs, load input from if specified.
        Parameters
        ----------
        path : String contain path to input file.
        k2x2 : If = 1 use 2x2 kernel else 3x3.
        """

        # Read input from file:
        try:
            inputs = imread(path)
        except:
            print('Cannot load specified file!')
            return

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

        self.generate_derivations(inputs)

    # --------------------------------------------------------------------------
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
    def p(self):
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

    # --------------------------------------------------------------------------
    # Method

    def generate_derivations(self, inputs):
        """Generate derivations from inputs."""

        # Convolve inputs with kernels:
        for y in range(self.shape[1]):
            for x in range(self.shape[2]):
                # Convolve then sum across the two last dimensions:
                self.data[2:, y, x] = (inputs[y : y+self.ksize,
                                              x : x+self.ksize]
                                       * self.kernel).sum(axis=(1, 2))

        # Compute into gradient magnitudes:
        self.data[1] = np.hypot(self.data[2], self.data[3])

# ----------------------------------------------------------------------
# ------------------------------------------------------------------------------