'''
Perform comparison of a and then g over predetermined range.
'''

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

def extend_comp(dert__, rng, fig):
    '''
    Compare a and then g over range = rng.
    Parameters
    ----------
    dert__ : MaskedArray
        Contain the arrays: g, m, dy, dx.
    rng : int
        Determine translation between comparands.
    fig : bool
        Indicate whether input is gradient (has an angle).
    Return
    ------
    out : MaskedArray
        The array that contain results from comparison.
    '''
    assert isinstance(dert__, ma.MaskedArray)

    return


def extend_comp_pixel(dert__, rng):
    '''
    Extend comparison of p over predetermined range.
    '''
    pass

def comp_i(dert__, rng):
    '''
    Compare g over predetermined range.
    '''
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
    return ma.array([a1[1]*a2[0] - a1[0]*a2[1],
                     a1[0]*a2[0] + a1[1]*a2[1]])

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