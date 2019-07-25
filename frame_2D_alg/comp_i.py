'''
Comparison of chosen parameter of derts__ (a or g)
over predetermined range (determined by kernel).
'''

import operator as op

import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# Constants

PI_TO_BYTE_SCALE = 114.79033031003102

# Declare comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

# Declare slicing for vectorized rng comparisons:
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

# Declare coefficients for decomposing d into dy and dx:
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

def comp_i(dert___, rng=1, flags=0):
    """
    Determine which parameter from dert__ is the input,
    then compare the input over predetermined range

    Parameters
    ----------
    dert__ : MaskedArray
        Contains input array.
    rng : int
        Determine translation between comparands.
    flags : int, default: 0
        Indicate which params in dert__ being used as input.

    Return
    ------
    i__ : MaskedArray
        Input array for summing in form_P_().
    new_dert___ : list
        Last element is the array of derivatives computed in
        this operation.
    """
    assert isinstance(dert___[-1], ma.MaskedArray)

    # Compare angle flow control:
    if flags & F_ANGLE:
        return comp_a(dert___, rng)

    # Assign input array:
    i__, dy__, dx__, m__ = assign_inputs(dert___, rng, flags)

    # Compare inputs:
    d__ = translated_operation(i__, rng, op.sub)
    valid = central_slice(rng)

    # Decompose and add to corresponding dy and dx:
    dy__[valid] += (d__ * Y_COEFFS[rng]).sum(axis=-1)
    dx__[valid] += (d__ * X_COEFFS[rng]).sum(axis=-1)

    # Compute ms:
    m__[valid] += translated_operation(i__, rng, np.minimum).sum(axis=-1)

    # Compute gs:
    g__ = ma.hypot(dy__, dx__)

    if flags & F_DERIV:
        new_dert___ = dert___ + [ma.stack((g__, m__, dy__, dx__), axis=0)]
    else:
        new_dert___ = dert___[:-1] + [ma.stack((g__, m__, dy__, dx__), axis=0)]

    return new_dert___


def assign_inputs(dert___, rng, flags):
    """Get input and accumulated dx, dy from dert___."""
    if flags & F_DERIV:
        i__ = dert___[-1][0] # Assign g__ of previous layer to i__

        # Accumulate dx__, dy__ starting from 0:
        dy__, dx__, m__ = (ma.array(np.zeros(i__.shape)) for _ in range(3))
    else:
        i__ = dert___[-2][0] # Assign one layer away g__ to i__

        # Accumulated m__, dx__, dy__ of previous layer:
        try: # Most of the time there's m__ (len(dert__) == 5):
            dx__ = dert___[-1][3] # Raise an IndexError is len(dert__) < 5.
            m__, dy__ = dert___[-1][1:3]
        except IndexError: # With dert from frame_blobs (len(dert__) == 4):
            dy__, dx__ = dert___[-1][1:3]
            m__ = ma.array(np.zeros(i__.shape))

    shrink_mask = rim_mask(i__.shape, rng)
    for arr in (dy__, dx__, m__):
        arr[shrink_mask] = ma.masked
    return i__, dy__, dx__, m__


def comp_a(dert___, rng):
    """
    Same functionality as comp_i except for comparands are 2D vectors
    instead of scalars, and differences here are differences in angle.
    """

    # Assign array of comparands:
    a__, day__, dax__ = assign_angle_inputs(dert___, rng)

    # Compute angle differences:
    da__ = translated_operation(a__, rng, angle_diff)
    valid = central_slice(rng)

    # Decompose and add to corresponding day and dax:
    day__[valid] += (da__ * Y_COEFFS[rng]).mean(axis=-1)
    dax__[valid] += (da__ * X_COEFFS[rng]).mean(axis=-1)

    # Compute ga:
    ga__ = (ma.hypot(ma.arctan2(*day__), ma.arctan2(*dax__))
            * PI_TO_BYTE_SCALE)[np.newaxis, ...]

    if rng > 1:
        new_dert___ = dert___[:-1] + [ma.concatenate((ga__,
                                                      day__,
                                                      dax__), axis=0)]
    else:
        new_dert___ = dert___[:-1] + [
            ma.concatenate((dert___[-1][:1], # Keep g__.
                            a__, # a__ Replace dy__, dx__.
                            ), axis=0),
            ma.concatenate((ga__, day__, dax__), axis=0),
        ]

    return new_dert___


def assign_angle_inputs(dert___, rng):
    """Get comparands and accumulated dax, day from dert___."""
    if rng > 1:
        a__ = dert___[-2][-2:]  # Assign a__ of previous layer

        # Accumulated dax__, day__ of previous layer:
        day__ = dert___[-1][-4:-2]
        dax__ = dert___[-1][-2:]
    else:
        # Compute angle from g__, dy__, dx__ of previous layer:
        g__ = dert___[-1][0]
        if len(dert___[-1]) == 3:
            dy__, dx__ =  dert___[-1][-2:]
        elif len(dert___[-1]) == 5:
            dy__, dx__ = ma.arctan2(dert___[-1][-2:-4],
                                    dert___[-1][-4:])
        else:
            raise(ValueError)

        g__[g__ == 0] = ma.masked # To avoid dividing by zero.

        a__ = np.stack((dy__, dx__), axis=0) / g__

        shape = (2,) + g__.shape
        # Accumulate dax__, day__ starting from 0:
        day__ = ma.array(np.zeros(shape), mask=rim_mask(shape, rng))
        dax__ = ma.array(np.zeros(shape))

    shrink_mask = rim_mask(a__.shape, rng)
    day__[shrink_mask] = ma.masked
    dax__[shrink_mask] = ma.masked

    return a__, day__, dax__

# -----------------------------------------------------------------------------
# Utility functions

def central_slice(i):
    """Return central slice objects (last 2 dimensions)."""
    if i < 1:
        return ..., slice(None), slice(None)
    return ..., slice(i, -i), slice(i, -i)


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