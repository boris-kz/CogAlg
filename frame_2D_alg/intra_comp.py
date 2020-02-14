"""
Perform comparison of a and then g over predetermined range.
"""

from itertools import starmap

import operator as op

import numpy as np
import numpy.ma as ma

from utils import pairwise

# -----------------------------------------------------------------------------
# Constants

# Slices for vectorized comparison:
TRANSLATING_SLICES_PAIRS_ = [
    [ # rng = 0
        (
            (Ellipsis, slice(1, None, None), slice(None, -1, None)),
            (Ellipsis, slice(None, -1, None), slice(1, None, None)),
        ),
        (
            (Ellipsis, slice(None, -1, None), slice(None, -1, None)),
            (Ellipsis, slice(1, None, None), slice(1, None, None)),
        ),
    ],
    [ # rng = 1
        (
            (Ellipsis, slice(None, -2, None), slice(None, -2, None)),
            (Ellipsis, slice(2, None, None), slice(2, None, None)),
        ),
        (
            (Ellipsis, slice(None, -2, None), slice(1, -1, None)),
            (Ellipsis, slice(2, None, None), slice(1, -1, None)),
        ),
        (
            (Ellipsis, slice(None, -2, None), slice(2, None, None)),
            (Ellipsis, slice(2, None, None), slice(None, -2, None)),
        ),
        (
            (Ellipsis, slice(1, -1, None), slice(2, None, None)),
            (Ellipsis, slice(1, -1, None), slice(None, -2, None)),
        ),
    ],
]

# coefficients for decomposing d into dy and dx:
Y_COEFFS = [
    np.array([-1, 1]),
    np.array([0.5, 1., 0.5, 0.]),
]
X_COEFFS = [
    np.array([1, 1]),
    np.array([0.5, 0., -0.5, -1.]),
]


# -----------------------------------------------------------------------------
# Functions

def extend_comp(dert__, fig):
    """Select comparison operation."""
    assert isinstance(dert__, ma.MaskedArray)

    # put selection for comparison here


def general_comp(i__, rng, operator):
    """
    General workflow of 2D array comparison.
    Parameters
    ----------
    i__ : array-like
        Array of inputs.
    rng : int
        Chebyshev distance between the a comparand with
        primary comparand (central).
    operator : function
        Binary operator used for comparison.
    Returns
    -------
    out : MaskedArray
        The array of values derived from provided inputs.
    """
    pass


def comp_a(dert__, rng):
    """Compare a over predetermined range."""
    pass


def comp_i(dert__, rng):
    """Compare g over predetermined range."""
    pass


# -----------------------------------------------------------------------------
# Utility functions

def central_slice(k):
    """Return central slice objects (last 2 dimensions)."""
    if k < 1:
        return ..., slice(None), slice(None)
    return ..., slice(k, -k), slice(k, -k)


def rim_mask(shape, i):
    """
    Return 2D array mask where outer pad (pad width=i) is True,
    the rest is False.
    """
    out = np.ones(shape, dtype=bool)
    out[central_slice(i)] = False
    return out


def translated_operation(a, rng, operator):
    """
    Return an array of corresponding results from operations between
    diametrically opposed translated slices.
    Parameters
    ----------
    a : array-like
        Input array.
    rng : int
        Half of the Chebyshev distance between the two inputs
        in each pairs.
    operator : function
        Binary operator used to compute results
    Return
    ------
    out : MaskedArray
        Array of results where additional dimension correspondent
        to each pair of translated slice.
    """
    out = ma.masked_array([*starmap(lambda ts1, ts2: operator(a[ts2], a[ts1]),
                                    TRANSLATING_SLICES_PAIRS_[rng])])

    # Rearrange axes:
    for dim1, dim2 in pairwise(range(out.ndim)):
        out = out.swapaxes(dim1, dim2)

    return out


def angle_diff(a2, a1):
    """
    Return the vector, of which angle is the angle between a2 and a1.
    Note: This only works for angle in 2D space.
    Parameters
    ----------
    a1 , a2 : array-like
        Each contains sin and co-sin of corresponding angle,
        in that order. For vectorized operations, sin/co-sin
        dimension must be the first dimension.
    Return
    ------
    out : MaskedArray
        The first dimension is sin/co-sin of the angle(s) between
        a2 and a1.
    """
    return ma.array([a1[1] * a2[0] - a1[0] * a2[1],
                     a1[0] * a2[0] + a1[1] * a2[1]])

    # OLD VERSION OF angle_diff
    # # Extend a1 vector(s) into basis/bases:
    # y, x = a1
    # bases = [(x, -y), (y, x)]
    # transform_mat = ma.array(bases)
    #
    # # Apply transformation:
    # da = ma.multiply(transform_mat, a2).sum(axis=1)

    # return da

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------