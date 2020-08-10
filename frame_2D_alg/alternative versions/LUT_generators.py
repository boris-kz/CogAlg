"""
Provide generators for lookup tables:
- 'slice' objects for CogAlg comparisons.
- Coefficients (weights) for CogAlg comparison
"""
# -----------------------------------------------------------------------------
# Imports

import operator as op
import numpy as np

from functools import reduce
from itertools import (
    repeat, accumulate, chain, starmap, tee
)
from utils import bipolar

# -----------------------------------------------------------------------------
# Constants

SCALER_GA = 255.9 / 2 ** 0.5 / np.pi # not used anymore


# -----------------------------------------------------------------------------
# MTLookupTable class

class MTLookupTable(object):
    """Meta-class for look-up table generators."""

    imports = set()  # For outputting into python script.

    def __init__(self, *args, **kwargs):
        """Meta __init__.py method."""
        self._generate_table(*args, **kwargs)

    def _generate_table(self, *args, **kwargs):
        """Meta-method for generating look-up tables."""
        return self

    def as_code_str(self):
        """
        Meta-method for generating look-up table in Python code.
        """
        return ""

    def to_file(self, path):
        """
        Meta-method for creating a lookup table module.
        """
        with open(path, "w") as file:
            file.write(self.as_code_str())
        return self


# -----------------------------------------------------------------------------
# GenCoeffs class

class GenCoeffs(MTLookupTable):
    """
    Generate coefficients used by comparisons
    of rng in {1, ..., max_rng}.
    """

    def __init__(self, max_rng=3, skip_central=True, ndround=None):
        """
        Instantiate a GenCoeffs object.
        Parameters
        ----------
        max_rng : int
            Maximum value of rng (window-size = rng*2 + 1).
        skip_central : bool
            Use of simplified comparison if True.
        ndround : int
            Number of digits that coefficients are rounded to.
        """
        self.skip_central = skip_central
        self.ndround = ndround
        MTLookupTable.__init__(self, max_rng=max_rng)

    # -------------------------------------------------------------------------
    # Class constants

    imports = {"import numpy as np\n"}
    rim_slices = [
        [  # For flattening outer rim of first two dimensions an ndarray:
            (..., 0, slice(None, -1)), # first row
            (..., slice(None, -1), -1), # last column
            (..., -1, slice(-1, 0, -1)), # last row
            (..., slice(-1, 0, -1), 0), # first column
        ],
        [  # For flattening outer rim of last two dimensions an ndarray:
            (0, slice(None, -1), ...),
            (slice(None, -1), -1, ...),
            (-1, slice(-1, 0, -1), ...),
            (slice(-1, 0, -1), 0, ...),
        ],
    ]

    # -------------------------------------------------------------------------
    # Class methods
    @classmethod
    def flattened_rim(cls, a, arranged_d=0):
        """
        Acquire and flatten the outer-pad of an array's
        last/first two dimensions.
        Parameters
        ----------
        a : array-like
            The input array.
        arranged_d : int
            Operate on last two dimensions if = 0, else
            operate on first two dimensions.
        Examples
        --------
        >>> a = np.arange(9).reshape(3, 3)
        >>> GenCoeffs.flattened_rim(a)
        array([0, 1, 2, 5, 8, 7, 6, 3])
        """
        return np.concatenate(tuple(map(lambda slices: a[slices],
                                        cls.rim_slices[arranged_d])),
                              axis=arranged_d - 1)

    # -------------------------------------------------------------------------
    # Methods

    def _generate_table(self, max_rng):
        """
        Workhorse of GenCoeffs class, compute kernel
        and separate into rng specific coefficients.
        """
        # Calculate combined kernel of rng from 1 to max_rng:
        kers = kernel(max_rng)

        # Separate into kernels of each rng and flatten them:
        self._coeffs = [*reversed([*
                                   map(GenCoeffs.flattened_rim,
                                       map(lambda slices: kers[slices],
                                           zip(
                                               repeat(...),
                                               *tee(chain(
                                                   (slice(None, None),),
                                                   map(lambda i: slice(i, -i),
                                                       range(1, max_rng)))),
                                           ))),
                                   ])]
        self._scalers = [n*(n - 1) / 2 for n in range(1, max_rng * 2 + 2, 2)]
        if self.skip_central:
            self._coeffs = [coeffs[:, : coeffs.shape[1]//2]
                            for coeffs in self._coeffs]
        else:
            self._g_scalers = [*map(
                lambda coeffs: 255.9 / (255 * np.hypot(*coeffs)),
                accumulate(map(
                    lambda coeffs: np.maximum(coeffs, 0).sum(axis=1),
                    self._coeffs,
                ))
            )]
        if self.ndround is not None:
            self._coeffs = [coeffs.round(self.ndround)
                            for coeffs in self._coeffs]

        return self

    def as_code_str(self):
        """
        s = ("\nSCALER_g = {\n"
             + reduce(op.add,
                      ("    %d:%0.15f,\n" % (i, scaler)
                       for i, scaler in
                       enumerate(self._g_scalers, start=1)))
             + "}\n")
        """

        ycoeffslist, xcoeffslist = zip(*self._coeffs)
        s = ("\nY_COEFFS = {\n"
             + reduce(op.add,
                      ("    %d:np." % i + repr(ycoeffs) + ",\n"
                       for i, ycoeffs in
                       enumerate(ycoeffslist, start=1)))
             + "}\n\nX_COEFFS = {\n"
             + reduce(op.add,
                      ("    %d:np." % i + repr(xcoeffs) + ",\n"
                       for i, xcoeffs in
                       enumerate(xcoeffslist, start=1)))
             + "}\n")

        return s

    @property
    def coeffs(self):
        return self._coeffs


# -----------------------------------------------------------------------------
# GenTransSlice class

class GenTransSlice(MTLookupTable):
    """
    Generate slicing for vectorized comparisons.
    """

    def __init__(self, max_rng=3, skip_central=True):
        """
        Instantiate a GenTransSlice object.
        """
        self.skip_central = skip_central
        MTLookupTable.__init__(self, max_rng=max_rng)

    def _generate_table(self, max_rng):
        """Generate target slices for comparison function."""
        self._slices = []
        slice_inds = [*chain((None,), range(1, max_rng * 2 + 1))]

        for r in range(3, max_rng * 2 + 2, 2):
            slices = [*starmap(slice, bipolar(slice_inds[:r]))]
            slices = [*chain(slices,
                             repeat(slices[-1],
                                    r - 2),
                             reversed(slices),
                             repeat(slices[0],
                                    r - 2))]
            slices = [*zip(repeat(...), slices[-r + 1:] + slices[:-r + 1], slices)]
            if self.skip_central:
                slices = [*zip(slices, slices[len(slices)//2 : ])]
            self._slices.append(slices)
        return self

    def as_code_str(self):
        s = "\nTRANSLATING_SLICES = {\n"
        for i, slices in enumerate(self._slices, start=0):
            s += "    %d:[\n" % i
            for sl in slices:
                s += "        " + str(sl) + ",\n"
            s += "    ],\n"
        s += "}\n"

        return s

    @property
    def slices(self):
        return self._slices


# -----------------------------------------------------------------------------
# Functions

def kernel(rng):
    """
    Return coefficients for decomposition of d
    (compared over rng) into dy and dx.
    Here, we assume that kernel width is odd.
    """
    # Start with array of indices:
    indices = np.indices((rng + 1, rng + 1)) * 2

    # Apply computations:
    # quart_kernel = indices / (indices**2).sum(axis=0)
    quart_kernel = indices.astype('float')
    non_zeros = quart_kernel != 0
    quart_kernel[non_zeros] = 1 / quart_kernel[non_zeros]

    # Copy quarter of kernel into full kernel:
    half_ky = np.concatenate(
        (
            np.flip(
                quart_kernel[0, :, 1:],
                axis=1),
            quart_kernel[0],
        ),
        axis=1,
    )

    ky = np.concatenate(
        (
            -np.flip(
                half_ky[1:],
                axis=0),
            half_ky,
        ),
        axis=0,
    )

    kx = ky.T  # Compute kernel for dx (transpose of ky).

    return np.stack((ky, kx), axis=0)


def create_lookup_module(path, *Gen_kwargs_pairs):
    """
    Create a Python module of look-up tables.
    Parameters
    ----------
    path : str
        Path, including the file name to the output file.
    Gen_kwargs_pairs : list of tuples
        Pairs of Generator classes corresponding dicts
        of keyword-arguments to create new objects.
    """
    required_imports = reduce(
        op.add,
        reduce(lambda Gen1, Gen2: Gen1.imports | Gen2.imports,
               next(zip(*Gen_kwargs_pairs))),
    )

    with open(path, "w") as file:
        file.write(required_imports)
        for Generator, kwargs in Gen_kwargs_pairs:
            gen = Generator(**kwargs)
            file.write("\n")
            file.write(gen.as_code_str())

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create look-up tables of translating slices and coefficients:
    create_lookup_module('LUT.py',
                         (GenTransSlice, {'max_rng':3, 'skip_central':False}),
                         (GenCoeffs, {'max_rng':3,
                                      'ndround':3,
                                      'skip_central':False}),
                         )