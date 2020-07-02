"""
Perform comparison of a and then g over predetermined range.
"""

import operator as op

import numpy as np
import numpy.ma as ma

from utils import pairwise

# -----------------------------------------------------------------------------
# Constants

# Slices for vectorized comparison:
TRANSLATING_SLICES_ = [
    [
        (Ellipsis, slice(None, -1, None), slice(None, -1, None)),
        (Ellipsis, slice(None, -1, None), slice(1, None, None)),
        (Ellipsis, slice(1, None, None), slice(1, None, None)),
        (Ellipsis, slice(1, None, None), slice(None, -1, None)),
    ],
    [
        (Ellipsis, slice(None, -2, None), slice(None, -2, None)),
        (Ellipsis, slice(None, -2, None), slice(1, -1, None)),
        (Ellipsis, slice(None, -2, None), slice(2, None, None)),
        (Ellipsis, slice(1, -1, None), slice(2, None, None)),
        (Ellipsis, slice(2, None, None), slice(2, None, None)),
        (Ellipsis, slice(2, None, None), slice(1, -1, None)),
        (Ellipsis, slice(2, None, None), slice(None, -2, None)),
        (Ellipsis, slice(1, -1, None), slice(None, -2, None)),
    ],
]
TRANSLATING_SLICES_PAIRS_ = [
    (  # rng = 0 or 2x2
        (
            (Ellipsis, slice(None, -1, None), slice(None, -1, None)),
            (Ellipsis, slice(1, None, None), slice(1, None, None)),
        ),
        (
            (Ellipsis, slice(None, -1, None), slice(1, None, None)),
            (Ellipsis, slice(1, None, None), slice(None, -1, None)),
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
]

# coefficients for decomposing d into dy and dx:

Y_COEFFS = [
    np.array([-1, -1, 1, 1]),
    np.array([-0.5, -0.5, -0.5,  0. ,  0.5,  0.5,  0.5,  0. ]),
]

X_COEFFS = [
    np.array([-1, 1, 1, -1]),
    np.array([-0.5,  0. ,  0.5,  0.5,  0.5,  0. , -0.5, -0.5]),
]

# -----------------------------------------------------------------------------
# Functions

def comp_g(dert__, odd):
    g__ = dert__[0]
    a__ = dert__[1:]

    # loop through each pair of comparands in a kernel
    dgy__ = ma.zeros(np.subtract(g.shape, rng))
    dgx__ = ma.zeros(np.subtract(g.shape, rng))
    for x_coeff, y_coeff, (ts, _ts) in zip(X_COEFFS[rng],
                                           Y_COEFFS[rng],
                                           TRANSLATING_SLICES_PAIRS_[rng]):
        # find angle differences
        da__ = angle_diff(a__[ts], a__[_ts])

        # compute dg: dg = g - _g * cos(da) at each position
        dg__ = g__[ts] - g__[_ts] * da__[1]

        # accumulate dgy, dgx
        dgx__ += dg__ * x_coeff
        dgy__ += dg__ * y_coeff

    gg__ = ma.hypot(dgy__, dgx__)

    return ma.stack((g__, gg__, dgy__, dgx__))


def comp_r(dert__, fig):

    i__, g__, dy__, dx__ = dert__[:]
    dy__ = dy__[:, rng:-rng, rng:-rng]
    dx__ = dx__[:, rng:-rng, rng:-rng]

    # comparison
    d__ = translated_operation(i__, rng=rng, operator=op.sub)

    # sum within kernels
    dy__ += (d__ * Y_COEFFS[rng]).sum(axis=-1)
    dx__ += (d__ * X_COEFFS[rng]).sum(axis=-1)

    # compute gradient magnitudes
    g__ = ma.hypot(dy__, dx__)

    return ma.stack((i__, g__, dy__, dx__))


def comp_a(dert__, inp):
    """
    Compare angles within specified rng.
    Parameters
    ----------
    dert__ : masked_array
        Contain input arrays.
    rng : int
        Chebyshev distance between the comparands.
    inp : tuple
        Index/indices of input arrays.
    Returns
    -------
    adert__ : masked_array
        Array of shape (5, <height>, <width>).
        - adert__[0, :, :] : sine inputs' (gradient) angles.
        - adert__[1, :, :] : cosine inputs' (gradient) angles.
        - adert__[2, :, :] : magnitude of angle derivation (ga).
        - adert__[3, :, :] : sines of angle differences by x axis.
        - adert__[4, :, :] : cosines of angle differences by x axis.
        - adert__[5, :, :] : sines of angle differences by y axis.
        - adert__[6, :, :] : cosines of angle differences by y axis.
    """
    if isinstance(inp, (list, tuple, set)):
        inp = list(inp)
    else:
        raise ValueError("'inp' should be a tuple of integers).")
    if len(inp) == 3:
        a__ = calc_a(dert__, inp)
    elif len(inp) == 5:
        a__ = calc_aga(dert__, inp)
    else:
        raise ValueError("'inp' should contain the index/indices "
                         "for g, dy, dx in that order.")
    return comp_angle(a__)


def calc_a(dert__, inp):
    """Compute angles of gradient."""
    return dert__[inp[1:]] / dert__[inp[0]]


def calc_aga(dert__, inp):
    """Compute angles of angle gradients."""
    g__ = dert__[inp[1]]
    day__ = np.arctan2(*dert__[inp[1:3]])
    dax__ = np.arctan2(*dert__[inp[3:]])
    return np.stack((day__, dax__)) / g__


def comp_angle(a__):
    """Compare angles."""

    # handle mask
    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # comparison
    da__,a__ = translated_operation(a__, rng=rng, operator=angle_diff)

    # sum within kernels
    day__ = (da__ * Y_COEFFS[rng]).sum(axis=-1)
    dax__ = (da__ * X_COEFFS[rng]).sum(axis=-1)

    # compute gradient magnitudes (how fast angles are changing)
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

    # pack into dert
    dert__ = ma.stack((*a__, ga__, *day__, *dax__), axis=0)

    # handle mask
    dert__.mask = np.isnan(dert__.data)

    return dert__


# -----------------------------------------------------------------------------
# Utility functions

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


    # get centered pixels for 2x2 and 3x3 based on rng
    a__ = a[:,1:-1,1:-1] if rng != 0 \
    else (a[:,:-1,:-1]+a[:,:-1,1:]+a[:,1:,1:]+a[:,1:,:-1]) /4

    # compute angle difference
    out = ma.masked_array([operator(a__, a[ts]) for ts in
                           TRANSLATING_SLICES_[rng]])

    # Rearrange axes:
    for dim1, dim2 in pairwise(range(out.ndim)):
        out = out.swapaxes(dim1, dim2)

    return out,a__


def translated_array(a, rng):
    """
    Like translated_operation, but without applying operation to slices
    """
    out = ma.stack([a[ts] for ts in TRANSLATING_SLICES_[rng]])

    # Rearrange axes:
    for dim1, dim2 in pairwise(range(out.ndim)):
        out = out.swapaxes(dim1, dim2)

    return out


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