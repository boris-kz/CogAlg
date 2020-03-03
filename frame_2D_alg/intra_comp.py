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
    (  # rng = 0 or 2x2
        (
            (Ellipsis, slice(1, None, None), slice(None, -1, None)),
            (Ellipsis, slice(None, -1, None), slice(1, None, None)),
        ),
        (
            (Ellipsis, slice(None, -1, None), slice(None, -1, None)),
            (Ellipsis, slice(1, None, None), slice(1, None, None)),
        ),
    ),
    (  # rng = 1 or 3x3
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
    ),
    (  # rng = 2 or 5x5
        (
            (Ellipsis, slice(None, -4, None), slice(None, -4, None)),
            (Ellipsis, slice(4, None, None), slice(4, None, None))
        ),
        (
            (Ellipsis, slice(None, -4, None), slice(1, -3, None)),
            (Ellipsis, slice(4, None, None), slice(3, -1, None))
        ),
        (
            (Ellipsis, slice(None, -4, None), slice(2, -2, None)),
            (Ellipsis, slice(4, None, None), slice(2, -2, None))
        ),
        (
            (Ellipsis, slice(None, -4, None), slice(3, -1, None)),
            (Ellipsis, slice(4, None, None), slice(1, -3, None))
        ),
        (
            (Ellipsis, slice(None, -4, None), slice(4, None, None)),
            (Ellipsis, slice(4, None, None), slice(None, -4, None))
        ),
        (
            (Ellipsis, slice(1, -3, None), slice(4, None, None)),
            (Ellipsis, slice(3, -1, None), slice(None, -4, None))
        ),
        (
            (Ellipsis, slice(2, -2, None), slice(4, None, None)),
            (Ellipsis, slice(2, -2, None), slice(None, -4, None))
        ),
        (
            (Ellipsis, slice(3, -1, None), slice(4, None, None)),
            (Ellipsis, slice(1, -3, None), slice(None, -4, None))
        ),
    ),
    (  # rng = 3 or 7x7
        (
            (Ellipsis, slice(None, -6, None), slice(None, -6, None)),
            (Ellipsis, slice(6, None, None), slice(6, None, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(1, -5, None)),
            (Ellipsis, slice(6, None, None), slice(5, -1, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(2, -4, None)),
            (Ellipsis, slice(6, None, None), slice(4, -2, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(3, -3, None)),
            (Ellipsis, slice(6, None, None), slice(3, -3, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(4, -2, None)),
            (Ellipsis, slice(6, None, None), slice(2, -4, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(5, -1, None)),
            (Ellipsis, slice(6, None, None), slice(1, -5, None))
        ),
        (
            (Ellipsis, slice(None, -6, None), slice(6, None, None)),
            (Ellipsis, slice(6, None, None), slice(None, -6, None))
        ),
        (
            (Ellipsis, slice(1, -5, None), slice(6, None, None)),
            (Ellipsis, slice(5, -1, None), slice(None, -6, None))
        ),
        (
            (Ellipsis, slice(2, -4, None), slice(6, None, None)),
            (Ellipsis, slice(4, -2, None), slice(None, -6, None))
        ),
        (
            (Ellipsis, slice(3, -3, None), slice(6, None, None)),
            (Ellipsis, slice(3, -3, None), slice(None, -6, None))
        ),
        (
            (Ellipsis, slice(4, -2, None), slice(6, None, None)),
            (Ellipsis, slice(2, -4, None), slice(None, -6, None))
        ),
        (
            (Ellipsis, slice(5, -1, None), slice(6, None, None)),
            (Ellipsis, slice(1, -5, None), slice(None, -6, None))
        ),
    ),
]

# coefficients for decomposing d into dy and dx:
Y_COEFFS = [
    np.array([-1, 1]),
    np.array([0.5, 1., 0.5, 0.]),
    np.array([0.25, 0.4, 0.5, 0.4, 0.25, 0.2, 0., -0.2]),
    np.array([0.167, 0.231, 0.3, 0.333, 0.3, 0.231,
              0.167, 0.154, 0.1, 0., -0.1, -0.154]),
]
X_COEFFS = [
    np.array([1, 1]),
    np.array([0.5, 0., -0.5, -1.]),
    np.array([0.25, 0.2, 0., -0.2, -0.25, -0.4, -0.5, -0.4]),
    np.array([0.167, 0.154, 0.1, 0., -0.1,
              -0.154, -0.167, -0.23076923, -0.3, -0.33333333,
              -0.3, -0.23076923]),
]


# -----------------------------------------------------------------------------
# Functions

def comp_g(g__):
    pass


def comp_r(dert__):
    pass


def calc_a(dert__):
    """Compute angles of gradient."""
    return dert__[2:] / dert__[1]


def comp_a(a__):
    """Compare angles within 2x2 kernels."""

    # handle mask
    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # comparison
    da__ = translated_operation(a__, rng=0, operator=angle_diff)

    # sum within kernels
    day__ = (da__ * Y_COEFFS[0]).sum(axis=-1)
    dax__ = (da__ * X_COEFFS[0]).sum(axis=-1)

    # compute gradient magnitudes (how fast angles are changing)
    ga__ = ma.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    # pack into dert
    dert__ = ma.stack((*a__[:, :-1, :-1], ga__, day__, dax__), axis=0)

    # handle mask
    dert__.mask = np.isnan(dert__.data)

    return dert__


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
        Each contains sine and cosine of corresponding angle,
        in that order. For vectorized operations, sine/cosine
        dimension must be the first dimension.
    Return
    ------
    out : MaskedArray
        The first dimension is sine/cosine of the angle(s) between
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