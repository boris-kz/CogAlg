'''
Compare a|g of input derts to a|g of derts in the perimeter of surrounding square,
side of the square = rng * 2 + 1
'''

import operator as op

import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
PI_BYTE = 162.33804195373324

# comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

# slices for vectorized rng comparisons:
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
    3:np.array([0.70710678, 0.83205029, 0.9486833 , 1.        , 0.9486833 ,
                0.83205029, 0.70710678, 0.5547002 , 0.31622777, 0.        ,
                0.31622777, 0.5547002 , 0.70710678, 0.83205029, 0.9486833 ,
                1.        , 0.9486833 , 0.83205029, 0.70710678, 0.5547002 ,
                0.31622777, 0.        , 0.31622777, 0.5547002 ]),
    2:np.array([0.70710678, 0.89442719, 1.        , 0.89442719, 0.70710678,
                0.4472136 , 0.        , 0.4472136 , 0.70710678, 0.89442719,
                1.        , 0.89442719, 0.70710678, 0.4472136 , 0.        ,
                0.4472136 ]),
    1:np.array([0.70710678, 1.        , 0.70710678, 0.        , 0.70710678,
                1.        , 0.70710678, 0.        ]),
}

X_COEFFS = {
    3:np.array([0.70710678, 0.5547002 , 0.31622777, 0.        , 0.31622777,
                0.5547002 , 0.70710678, 0.83205029, 0.9486833 , 1.        ,
                0.9486833 , 0.83205029, 0.70710678, 0.5547002 , 0.31622777,
                0.        , 0.31622777, 0.5547002 , 0.70710678, 0.83205029,
                0.9486833 , 1.        , 0.9486833 , 0.83205029]),
    2:np.array([0.70710678, 0.4472136 , 0.        , 0.4472136 , 0.70710678,
                0.89442719, 1.        , 0.89442719, 0.70710678, 0.4472136 ,
                0.        , 0.4472136 , 0.70710678, 0.89442719, 1.        ,
                0.89442719]),
    1:np.array([0.70710678, 0.        , 0.70710678, 1.        , 0.70710678,
                0.        , 0.70710678, 1.        ]),
}

# -----------------------------------------------------------------------------

def comp_i(dert___, rng=1, flags=0):
    """
    Select compared parameter in dert in dert__:
    ----------
    dert__ : MaskedArray  #  Contains input array.
    rng : int,  flags : int, default: 0   # translation into compared parameter
    Return:
    ------
    i__ : MaskedArray   # for summing in form_P_().
    new_dert___ : list  # Last element is array of resulting derivatives
    """

    if flags & F_ANGLE:
        return comp_a(dert___, rng)

    i__, dy__, dx__ = get_new_gdert(dert___, rng, flags)
    d__ = comp_translated(i__, rng, op.sub)  # Compare translated inputs

    # Decompose and add to corresponding dy and dx per extended square:

    dy__ += (d__ * Y_COEFFS[rng]).sum(axis=-1)
    dx__ += (d__ * X_COEFFS[rng]).sum(axis=-1)
    g__ = ma.hypot(dy__, dx__)

    if flags & F_DERIV:
        new_dert___ = dert___ + [ma.stack((g__, dy__, dx__), axis=0)]
    else:
        new_dert___ = dert___[:-1] + [ma.stack((g__, dy__, dx__), axis=0)]

    return i__, new_dert___

def get_new_gdert(dert___, rng, flags):  # Get input g and accumulated dx, dy from dert___

    if flags & F_DERIV:
        assert rng == 1
        i__ = dert___[-1][0]  # i__ is g__ of previous layer
        shape = tuple(np.subtract(i__.shape, 2))

        # accumulate dx__, dy__ starting from 0:
        dy__ = ma.zeros(shape)
        dx__ = ma.zeros(shape)
    else:
        i__ = dert___[-2][0]  # select g__ one layer away from i__
        dy__, dx__ = dert___[-1][-2:][central_slice(1)]  # accumulated dx and dy

    return i__, dy__, dx__

def comp_a(dert___, rng):
    """
    As comp_i but comparands are 2D vectors, forms differences in angle.
    """
    a__, day__, dax__ = get_new_adert(dert___, rng)
    da__ = comp_translated(a__, rng, angle_diff)  # Compute angle differences

    # Decompose and add to corresponding day and dax:
    day__ += (da__ * Y_COEFFS[rng]).sum(axis=-1)
    dax__ += (da__ * X_COEFFS[rng]).sum(axis=-1)

    ga__ = (ma.arctan2(*ma.hypot(day__, dax__))
            * PI_BYTE)[np.newaxis, ...]

    if rng > 1:
        new_dert___ = dert___[:-1] + [ma.concatenate((ga__,
                                                      day__,
                                                      dax__), axis=0)]
    else:
        new_dert___ = dert___[:-1] + [
            ma.concatenate((dert___[-1][:1],
                            a__,
                            dert___[-1][1:]), axis=0),
            ma.concatenate((ga__, day__, dax__), axis=0),
        ]
    return a__, new_dert___

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_new_adert(dert___, rng):  # get angle and accumulated dax, day from dert___

    if rng > 1:  # select angle, dax__, day__ accumulated on previous layer:
        a__ = dert___[-1][1:3]
        day__ = dert___[-1][-4:-2][central_slice(1)]
        dax__ = dert___[-1][-2:][central_slice(1)]

    else:   # compute angle from g__, dy__, dx__:
        g__ = dert___[-1][0]
        dy__, dx__ =  dert___[-1][-2:]
        a__ = np.stack((dy__, dx__), axis=0) / g__

        shape = tuple(np.subtract(dy__.shape, 2))
        # accumulate dax__, day__ starting from 0:
        day__ = ma.zeros((2,)+shape)
        dax__ = ma.zeros((2,)+shape)

    return a__, day__, dax__
# -----------------------------------------------------------------------------

# Utility functions

def central_slice(i):
    """Return central slice objects (last 2 dimensions)."""
    if i < 1:
        return ..., slice(None), slice(None)
    return ..., slice(i, -i), slice(i, -i)

def comp_translated(a, rng, operator):
    """
    Return array of differences between central slice and translated slices
    ----------
    a : ndarray  # Input array
    rng : int   # Range of translations
    operator : function  # Binary operator computing the differences
    out : ndarray  # computed differences are added to each translated slice.
    """
    out = ma.masked_array([*map(lambda slices:
                                    operator(a[slices],
                                             a[central_slice(rng)]),
                                TRANSLATING_SLICES[rng])])
    for dim in range(out.ndim - 1):
        out = out.swapaxes(dim, dim+1)

    return out

def angle_diff(a2, a1):
    """
    Return angle between a2 and a1, works in arrays, for 2D vectors
    """
    y, x = a1
    bases = [(x, -y), (y, x)]
    transform_mat = ma.array(bases)  # Extend a1 vector(s) into basis/bases

    da = (transform_mat * a2).sum(axis=1)

    return da

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------