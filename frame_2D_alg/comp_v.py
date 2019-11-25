"""
Perform comparison of g or a over predetermined range.
"""

import operator as op

import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Constants

# Define scalers:
SCALER_g = {
    1:0.354801226089485,
    2:0.168952964804517,
    3:0.110256954721035,
}
SCALER_ga = 57.597736326150859

# Define slicing for vectorized rng comparisons:
TRANSLATING_SLICES = {
    1:[
        (Ellipsis, slice(None, -2, None), slice(None, -2, None)),
        (Ellipsis, slice(None, -2, None), slice(1, -1, None)),
        (Ellipsis, slice(None, -2, None), slice(2, None, None)),
        (Ellipsis, slice(1, -1, None), slice(2, None, None)),
        (Ellipsis, slice(2, None, None), slice(2, None, None)),
        (Ellipsis, slice(2, None, None), slice(1, -1, None)),
        (Ellipsis, slice(2, None, None), slice(None, -2, None)),
        (Ellipsis, slice(1, -1, None), slice(None, -2, None)),
    ],
    2:[
        (Ellipsis, slice(None, -4, None), slice(None, -4, None)),
        (Ellipsis, slice(None, -4, None), slice(1, -3, None)),
        (Ellipsis, slice(None, -4, None), slice(2, -2, None)),
        (Ellipsis, slice(None, -4, None), slice(3, -1, None)),
        (Ellipsis, slice(None, -4, None), slice(4, None, None)),
        (Ellipsis, slice(1, -3, None), slice(4, None, None)),
        (Ellipsis, slice(2, -2, None), slice(4, None, None)),
        (Ellipsis, slice(3, -1, None), slice(4, None, None)),
        (Ellipsis, slice(4, None, None), slice(4, None, None)),
        (Ellipsis, slice(4, None, None), slice(3, -1, None)),
        (Ellipsis, slice(4, None, None), slice(2, -2, None)),
        (Ellipsis, slice(4, None, None), slice(1, -3, None)),
        (Ellipsis, slice(4, None, None), slice(None, -4, None)),
        (Ellipsis, slice(3, -1, None), slice(None, -4, None)),
        (Ellipsis, slice(2, -2, None), slice(None, -4, None)),
        (Ellipsis, slice(1, -3, None), slice(None, -4, None)),
    ],
    3:[
        (Ellipsis, slice(None, -6, None), slice(None, -6, None)),
        (Ellipsis, slice(None, -6, None), slice(1, -5, None)),
        (Ellipsis, slice(None, -6, None), slice(2, -4, None)),
        (Ellipsis, slice(None, -6, None), slice(3, -3, None)),
        (Ellipsis, slice(None, -6, None), slice(4, -2, None)),
        (Ellipsis, slice(None, -6, None), slice(5, -1, None)),
        (Ellipsis, slice(None, -6, None), slice(6, None, None)),
        (Ellipsis, slice(1, -5, None), slice(6, None, None)),
        (Ellipsis, slice(2, -4, None), slice(6, None, None)),
        (Ellipsis, slice(3, -3, None), slice(6, None, None)),
        (Ellipsis, slice(4, -2, None), slice(6, None, None)),
        (Ellipsis, slice(5, -1, None), slice(6, None, None)),
        (Ellipsis, slice(6, None, None), slice(6, None, None)),
        (Ellipsis, slice(6, None, None), slice(5, -1, None)),
        (Ellipsis, slice(6, None, None), slice(4, -2, None)),
        (Ellipsis, slice(6, None, None), slice(3, -3, None)),
        (Ellipsis, slice(6, None, None), slice(2, -4, None)),
        (Ellipsis, slice(6, None, None), slice(1, -5, None)),
        (Ellipsis, slice(6, None, None), slice(None, -6, None)),
        (Ellipsis, slice(5, -1, None), slice(None, -6, None)),
        (Ellipsis, slice(4, -2, None), slice(None, -6, None)),
        (Ellipsis, slice(3, -3, None), slice(None, -6, None)),
        (Ellipsis, slice(2, -4, None), slice(None, -6, None)),
        (Ellipsis, slice(1, -5, None), slice(None, -6, None)),
    ],
}

# coefficients for decomposing d into dy and dx:
Y_COEFFS = {
    1:np.array([-0.5, -1. , -0.5,  0. ,  0.5,  1. ,  0.5,  0. ]),
    2:np.array([-0.25, -0.4 , -0.5 , -0.4 , -0.25, -0.2 ,  0.  ,  0.2 ,  0.25,
                0.4 ,  0.5 ,  0.4 ,  0.25,  0.2 ,  0.  , -0.2 ]),
    3:np.array([-0.16666667, -0.23076923, -0.3       , -0.33333333, -0.3       ,
                -0.23076923, -0.16666667, -0.15384615, -0.1       ,  0.        ,
                0.1       ,  0.15384615,  0.16666667,  0.23076923,  0.3       ,
                0.33333333,  0.3       ,  0.23076923,  0.16666667,  0.15384615,
                0.1       ,  0.        , -0.1       , -0.15384615]),
}
X_COEFFS = {
    1:np.array([-0.5,  0. ,  0.5,  1. ,  0.5,  0. , -0.5, -1. ]),
    2:np.array([-0.25, -0.2 ,  0.  ,  0.2 ,  0.25,  0.4 ,  0.5 ,  0.4 ,  0.25,
                0.2 ,  0.  , -0.2 , -0.25, -0.4 , -0.5 , -0.4 ]),
    3:np.array([-0.16666667, -0.15384615, -0.1       ,  0.        ,  0.1       ,
                 0.15384615,  0.16666667,  0.23076923,  0.3       ,  0.33333333,
                0.3       ,  0.23076923,  0.16666667,  0.15384615,  0.1       ,
                0.        , -0.1       , -0.15384615, -0.16666667, -0.23076923,
                -0.3       , -0.33333333, -0.3       , -0.23076923]),
}

# -----------------------------------------------------------------------------
# Functions

def comp_v(dert__, nI, rng):
    """
    Compare g or a over range = rng
    Parameters
    ----------
    dert__ : MaskedArray
        Contain the arrays: g, m, dy, dx.
    nI : int
        Determine comparands.
    rng : int
        Determine translation between comparands.
    Return
    ------
    out : MaskedArray
        The array that contain result from comparison.
    """
    assert isinstance(dert__, ma.MaskedArray)

    if nI in (2, 3, 4, 5): # Input is dy or ay, indices are wrong here
        return comp_a(dert__, rng)
    else: # Input is g or ga:
        return comp_i(select_i(dert__, nI), rng)


def select_i(dert__, nI):
    """
    Select inputs to compare.
    """
    i__ = dert__[nI]
    if nI == 0: # Accumulated m, dy, dx:
        if len(dert__) == 10:
            m__, dy__, dx__ = dert__[2:5]
        else:
            dy__, dx__ = dert__[2:4]
            m__ = ma.zeros(i__.shape)
    else: # Initialized m, dy, dx:
        m__, dy__, dx__ = [ma.zeros(i__.shape) for _ in range(3)]

    return i__, m__, dy__, dx__


def comp_i(dert__, rng):
    """
    Compare g over predetermined range.
    """
    # Unpack dert__:
    i__, m__, dy__, dx__ = dert__

    # Compare gs:
    d__ = translated_operation(i__, rng, op.sub)
    comp_field = central_slice(rng)

    # Decompose and add to corresponding dy and dx:
    dy__[comp_field] += (d__ * Y_COEFFS[rng]).sum(axis=-1)
    dx__[comp_field] += (d__ * X_COEFFS[rng]).sum(axis=-1)

    # Compute ms:
    m__[comp_field] += translated_operation(i__, rng, ma.minimum).sum(axis=-1)

    # Apply mask:
    msq = np.ones(i__.shape, dtype=int)  # Rim mask.
    msq[comp_field] = i__.mask[comp_field] + d__.mask.sum(axis=-1)  # Summed d mask.
    imsq = msq.nonzero()
    m__[imsq] = dy__[imsq] = dx__[imsq] = ma.masked # Apply mask.

    # Compute gg:
    g__ = ma.hypot(dy__, dx__) * SCALER_g[rng]

    return ma.stack((i__, g__, m__, dy__, dx__), axis=0) # ma.stack() for extra array dimension.


def comp_a(dert__, rng):
    """
    Compute and compare a over predetermined range.
    """
    # Unpack dert__:
    if len(dert__) in (5, 12): # idert or full dert with m.
        i__, g__, m__, dy__, dx__ = dert__[:5]
    else: # idert or full dert without m.
        i__, g__, dy__, dx__ = dert__[:4]

    if len(dert__) > 10: # if ra+:
        a__ = dert__[-7:-5] # Computed angle (use reverse indexing to avoid m check).
        day__ = dert__[-4:-2] # Accumulated day__.
        dax__ = dert__[-2:] # Accumulated day__.
    else: # if fa:
        # Compute angles:
        a__ = ma.stack((dy__, dx__), axis=0) / g__
        a__.mask = g__.mask

        # Initialize dax, day:
        day__, dax__ = [ma.zeros((2,) + i__.shape) for _ in range(2)]

    # Compute angle differences:
    da__ = translated_operation(a__, rng, angle_diff)
    comp_field = central_slice(rng)


    # Decompose and add to corresponding day and dax:
    day__[comp_field] = (da__ * Y_COEFFS[rng]).mean(axis=-1)
    dax__[comp_field] = (da__ * X_COEFFS[rng]).mean(axis=-1)

    # Apply mask:
    msq = np.ones(a__.shape, dtype=int) # Rim mask.
    msq[comp_field] = a__.mask[comp_field] + da__.mask.sum(axis=-1) # Summed d mask.
    imsq = msq.nonzero()
    day__[imsq] = dax__[imsq] = ma.masked # Apply mask.

    # Compute ga:
    ga__ = ma.hypot(
        ma.arctan2(*day__),
        ma.arctan2(*dax__)
    )[np.newaxis, ...] * SCALER_ga

    try: # dert with m is more common:
        return ma.concatenate( # Concatenate on the first dimension.
            (
                ma.stack((i__, g__, m__, dy__, dx__), axis=0),
                a__, ga__, day__, dax__,
            ),
            axis=0,
        )
    except NameError: # m doesn't exist:
        return ma.concatenate(  # Concatenate on the first dimension.
            (
                ma.stack((i__, g__, dy__, dx__), axis=0),
                a__, ga__, day__, dax__,
            ),
            axis=0,
        )

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
    translated slices and central slice of an array.
    Parameters
    ----------
    a : ndarray
        Input array.
    rng : int
        Range of translations.
    operator : function
        Binary operator of which the result between central
        and translated slices are returned.
    Return
    ------
    out : ndarray
        Array of results where additional dimension correspondent
        to each translated slice.
    """
    out = ma.masked_array([*map(lambda slices:
                                    operator(a[slices],
                                             a[central_slice(rng)]),
                                TRANSLATING_SLICES[rng])])

    # Rearrange axes:
    for dim in range(out.ndim - 1):
        out = out.swapaxes(dim, dim+1)

    return out


def angle_diff(a2, a1):
    """
    Return the vector, of which angle is the angle between a2 and a1.
    Can be applied to arrays.
    Note: This only works for 2D vectors.
    """
    # Extend a1 vector(s) into basis/bases:
    y, x = a1
    bases = [(x, -y), (y, x)]
    transform_mat = ma.array(bases)

    # Apply transformation:
    da = (transform_mat * a2).sum(axis=1)

    return da

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------