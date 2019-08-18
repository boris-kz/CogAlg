"""
Provide generators for lookup tables:
- Slicing objects for CogAlg comparisons.
- Coefficients for CogAlg comparison
"""
# -----------------------------------------------------------------------------
# Imports

import operator as op
import numpy as np

from functools import reduce
from itertools import (
    repeat, accumulate, chain, starmap, tee
)
from utils import bipolar, kernel
# -----------------------------------------------------------------------------
# Constants

SCALER_GA = 255.9 / 2**0.5 / np.pi

# -----------------------------------------------------------------------------
# MTLookupTable class

class MTLookupTable(object):
    """Meta-class for look-up table generators."""

    imports = set()  # For outputting into python script.
    def __init__(self, *args, **kwargs):
        """Meta __init__ method."""
        self._generate_table(*args, **kwargs)

    def _generate_table(self, *args, **kwargs):
        """Meta-method for generating look-up table."""
        return self

    def as_code_str(self):
        """
        Meta-method for generating python code string declaring
        look-up table in Python code.
        """
        return ""

    def to_file(self, path):
        """
        Meta-method for outputting loop-up table to Python script.
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

    def __init__(self, max_rng=3):
        """
        Instantiate a GenCoeffs object.
        """
        MTLookupTable.__init__(self, max_rng=max_rng)

    # -------------------------------------------------------------------------
    # Class constants

    imports = {"import numpy as np\n"}
    rim_slices = [
        [# For flattening outer rim of first two dimensions an ndarray:
            (..., 0, slice(None, -1)),
            (..., slice(None, -1), -1),
            (..., -1, slice(-1, 0, -1)),
            (..., slice(-1, 0, -1), 0),
        ],
        [ # For flattening outer rim of last two dimensions an ndarray:
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

        self._g_scalers = [*map(
            lambda coeffs: 255.9 / (255 * np.hypot(*coeffs)),
            accumulate(map(
                lambda coeffs: np.maximum(coeffs, 0).sum(axis=1),
                self._coeffs,
            ))
        )]
        return self

    def as_code_str(self):

        '''
        s = ("\nSCALER_g = {\n"
             + reduce(op.add,
                      ("    %d:%0.15f,\n" % (i, scaler)
                       for i, scaler in
                       enumerate(self._g_scalers, start=1)))
             + "}\n")
        '''

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
    def coeff(self):
        return self._coeffs

# -----------------------------------------------------------------------------
# GenTransSlice class

class GenTransSlice(MTLookupTable):
    """
    Generate slicing for vectorized comparisons.
    """
    def __init__(self, max_rng=3):
        """
        Instantiate a GenTransSlice object.
        """
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
            slices = [*zip(repeat(...), slices[-r+1:] + slices[:-r+1], slices)]
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

def create_lookup_module(path, Generators):
    """Create a Python module of look-up tables."""
    required_imports = reduce(
        op.add,
        reduce(
            lambda gen1, gen2: gen1[0].imports | gen2[0].imports,
            Generators,
        ),
    )

    with open(path, "w") as file:
        file.write(required_imports)
        for Generator, *args in Generators:
            gen = Generator(*args)
            file.write("\n")
            file.write(gen.as_code_str())

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
