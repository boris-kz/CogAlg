'''
Comparison of chosen parameter of derts__ (a or g)
over predetermined range (determined by kernel).
'''

import numpy as np

# -----------------------------------------------------------------------------
# Constants

# Declare comparison flags:
F_ANGLE = 0b01
F_DERIVE = 0b10

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
    1:np.array([0.70710678, 0.83205029, 0.9486833 , 1.        , 0.9486833 ,
                0.83205029, 0.70710678, 0.5547002 , 0.31622777, 0.        ,
                0.31622777, 0.5547002 , 0.70710678, 0.83205029, 0.9486833 ,
                1.        , 0.9486833 , 0.83205029, 0.70710678, 0.5547002 ,
                0.31622777, 0.        , 0.31622777, 0.5547002 ]),
    2:np.array([0.70710678, 0.89442719, 1.        , 0.89442719, 0.70710678,
                0.4472136 , 0.        , 0.4472136 , 0.70710678, 0.89442719,
                1.        , 0.89442719, 0.70710678, 0.4472136 , 0.        ,
                0.4472136 ]),
    3:np.array([0.70710678, 1.        , 0.70710678, 0.        , 0.70710678,
                1.        , 0.70710678, 0.        ]),
}

X_COEFFS = {
    1:np.array([0.70710678, 0.5547002 , 0.31622777, 0.        , 0.31622777,
                0.5547002 , 0.70710678, 0.83205029, 0.9486833 , 1.        ,
                0.9486833 , 0.83205029, 0.70710678, 0.5547002 , 0.31622777,
                0.        , 0.31622777, 0.5547002 , 0.70710678, 0.83205029,
                0.9486833 , 1.        , 0.9486833 , 0.83205029]),
    2:np.array([0.70710678, 0.4472136 , 0.        , 0.4472136 , 0.70710678,
                0.89442719, 1.        , 0.89442719, 0.70710678, 0.4472136 ,
                0.        , 0.4472136 , 0.70710678, 0.89442719, 1.        ,
                0.89442719]),
    3:np.array([0.70710678, 0.        , 0.70710678, 1.        , 0.70710678,
                0.        , 0.70710678, 1.        ]),
}

# -----------------------------------------------------------------------------
# Functions

def comp_i(dert__, k, flags): # Need to be coded from scratch
    return