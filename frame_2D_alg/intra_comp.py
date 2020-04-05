"""
Cross-comparison of pixels, angles, or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import numpy.ma as ma
import math

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Coefficients to decompose d into dy and dx, replace with separate sign and magnitude coefs?
Y_COEFFS = [
    np.array([-1, -1, 1, 1]),
    np.array([-0.5, -0.5, -0.5,  0. ,  0.5,  0.5,  0.5,  0. ]),
]
X_COEFFS = [
    np.array([-1, 1, 1, -1]),
    np.array([-0.5,  0. ,  0.5,  0.5,  0.5,  0. , -0.5, -0.5]),
]
''' old scheme:
day__ = ( Y_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
          Y_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
        )
dax__ = ( X_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
          X_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
        )
'''
# ------------------------------------------------------------------------------------
# Functions

def comp_g(dert__):  # add fga if processing in comp_ga is different?
    """
    Cross-comp of g or ga in 2x2 kernels, between derts in ma.stack dert__:
    input dert = (i, g, dy, dx, ga, day, dax, cos_da0, cos_da1)
    output dert = (g, gg, dgy, dgx, gm, ga, day, dax)
    """
    g__, cos_da0__, cos_da1__ = dert__[[1, -2, -1]]  # list of indices indicates top dimension of numpy stack

    # this mask section would need further test later with actual input from frame_blobs
    if isinstance(g__, ma.masked_array):
        g__.data[g__.mask] = np.nan
        g__.mask = ma.nomask

    g_topleft__ = g__[:-1, :-1]
    g_topright__ = g__[:-1, 1:]
    g_botleft__ = g__[1:, :-1]
    g_botright__ = g__[1:, 1:]

    # please check, not sure this in the right order, also need to add sign COEFFS:

    dgy__ = ((g_botleft__ + g_botright__) -
             (g_topleft__ * cos_da0__ + g_topright__ * cos_da1__))
    # y-decomposed difference between gs

    dgx__ = ((g_topright__ + g_botright__) -
             (g_topleft__ * cos_da0__ + g_botleft__ * cos_da1__))
    # x-decomposed difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g_topleft__, (g_botright__ * cos_da0__))  # g match = min(g, _g*cos(da))
    mg1__ = np.minimum(g_topright__, (g_botleft__ * cos_da1__))
    mg__  = mg0__ + mg1__

    gdert = ma.stack(g__, gg__, dgy__, dgx__, mg__, dert__[4], dert__[5], dert__[6])
    # ga__=dert__[5], day_=dert__[6], dax=dert__[7]
    '''
    next comp_r will use g, dgy, dgx   # comp_rg
    next comp_a will use ga, day, dax  # comp_agg
    '''
    return gdert


def comp_r_draft(dert__, fig):
    """
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.

    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 4: 9x9 kernel,
    ...
    skipping rim derts as next-comp central derts forms sparse output dert__,
    hence configuration of input derts in next-rng kernel will always be 3x3.
    Parameters
    ----------
    dert__ : array-like
        dert's structure is (i, g, dy, dx, m, if fig: + idy, idx).
    fig : bool
        True if input is g.
    Returns
    -------
    rdert__ : masked_array
        Output's structure is (i, g, dy, dx, m, if fig: + idy, idx).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fig = 'specific value'
    >>> comp_r(dert__, fig)
    'specific output'
    Notes
    -----
    - Results are accumulated in the input dert.
    - Comparand is dert[0].
    """

    # if input is gdert (g,  gg, gdy, gdx, gm, iga, iday, idax)
    # input is dert  (i,  g,  dy,  dx,  m)
    # input is rdert (ir, gr, dry, drx, mr)
    i__, g__, dy__, dx__, m__ = dert__[0:5]

    # get sparsed value
    ir__ = i__[::2, ::2]
    gr__ = g__[::2, ::2]
    dry__ = dy__[::2, ::2]
    drx__ = dx__[::2, ::2]
    mr__ = m__[::2, ::2]

    # get each direction
    ir__topleft = ir__[:-2, :-2]
    ir__top = ir__[:-2, 1:-1]
    ir__topright = ir__[:-2, 2:]
    ir__right = ir__[1:-1, 2:]
    ir__bottomright = ir__[2:, 2:]
    ir__bottom = ir__[2:, 1:-1]
    ir__bottomleft = ir__[2:, :-2]
    ir__left = ir__[1:-1, :-2]

    if fig:  # input is g (need further update on how to utilize iday and idax on this section)

        #  central of g in sparsed g
        ir__ = ir__[1:-1, 1:-1]

        # difference in 8 directions
        drg__ = np.stack((ir__ - ir__topleft,
                          ir__ - ir__top,
                          ir__ - ir__topright,
                          ir__ - ir__right,
                          ir__ - ir__bottomright,
                          ir__ - ir__bottom,
                          ir__ - ir__bottomleft,
                          ir__ - ir__left))

        # compute range a from the range dy,dx and g
        a__ = [dry__, drx__] / gr__

        # angles per direction:
        a__topleft = a__[:, :-2, :-2]
        a__top = a__[:, :-2, 1:-1]
        a__topright = a__[:, :-2, 2:]
        a__right = a__[:, 1:-1, 2:]
        a__bottomright = a__[:, 2:, 2:]
        a__bottom = a__[:, 2:, 1:-1]
        a__bottomleft = a__[:, 2:, :-2]
        a__left = a__[:, 1:-1, :-2]

        # central of a in sparsed a
        a__ = a__[:, 1:-1, 1:-1]

        # compute da s in 3x3 kernel
        dra__ = np.stack((angle_diff(a__, a__topleft),
                          angle_diff(a__, a__top),
                          angle_diff(a__, a__topright),
                          angle_diff(a__, a__right),
                          angle_diff(a__, a__bottomright),
                          angle_diff(a__, a__bottom),
                          angle_diff(a__, a__bottomleft),
                          angle_diff(a__, a__left)))

        # g difference  = g - g * cos(da) at each opposing
        dri__ = np.stack((drg__[0] - drg__[0] * dra__[0][1],
                          drg__[1] - drg__[1] * dra__[1][1],
                          drg__[2] - drg__[2] * dra__[2][1],
                          drg__[3] - drg__[3] * dra__[3][1],
                          drg__[4] - drg__[4] * dra__[4][1],
                          drg__[5] - drg__[5] * dra__[5][1],
                          drg__[6] - drg__[6] * dra__[6][1],
                          drg__[7] - drg__[7] * dra__[7][1]))

    else:  # input is pixel (should we use diagonal difference or 8 directional difference here?)

        # central of p in sparsed p
        ir__ = ir__[1:-1, 1:-1]

        # difference in 8 directions
        dri__ = np.stack((ir__ - ir__topleft,
                          ir__ - ir__top,
                          ir__ - ir__topright,
                          ir__ - ir__right,
                          ir__ - ir__bottomright,
                          ir__ - ir__bottom,
                          ir__ - ir__bottomleft,
                          ir__ - ir__left))

    dri__ = np.rollaxis(dri__, 0, 3)

    # compute dry and drx
    dry__ = (dri__ * Y_COEFFS[1]).sum(axis=-1)
    drx__ = (dri__ * X_COEFFS[1]).sum(axis=-1)

    # compute gradient magnitudes
    drg__ = ma.hypot(dry__, drx__)

    # pending m computation
    drm__ = []

    # rdert
    rdert = dri__, drg__, dry__, drx__, drm__, dy__, dx__

    return rdert


def comp_a(dert__, fga):
    """
    cross-comp of a or aga in 2x2 kernels
    ----------
    input dert__ : array-like
    fga : bool
        If True, dert structure is interpreted as:
        (g, gg, gdy, gdx, gm, iga, iday, idax)
        else: (i, g, dy, dx, m)
    ----------
    output adert: masked_array of aderts,
    adert structure is (i, g, dy, dx, m, ga, day, dax, cos_da0, cos_da1)
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fga = 'specific value'
    >>> comp_a(dert__, fga)
    'specific output'
    """
    # input dert = (i,  g,  dy,  dx,  m, ?(ga, day, dax))
    i__, g__, dy__, dx__, m__ = dert__[0:5]

    if fga:  # input is adert
        ga__, day__, dax__ = dert__[5:8]
        a__ = [day__, dax__] / ga__  # similar to calc_a
    else:
        a__ = [dy__, dx__] / g__  # similar to calc_a

    # this mask section would need further test later with actual input from frame_blobs
    if isinstance(a__, ma.masked_array):
        a__.data[a__.mask] = np.nan
        a__.mask = ma.nomask

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__botright = a__[:, 1:, 1:]
    a__botleft = a__[:, 1:, :-1]

    # diagonal angle differences

    sin_da0__, cos_da0__ = angle_diff(a__topleft, a__botright)
    sin_da1__, cos_da1__ = angle_diff(a__topright, a__botleft)

    day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed

    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))  # angle gradient:

    adert__ = ma.stack(i__, g__, dy__, dx__, m__, ga__, day__, dax__, cos_da0__, cos_da1__)
    # i, dy, dx, m is for summation in Dert only?
    '''
    next comp_g will use g, cos_da0__, cos_da1__
    next comp_a will use ga, day, dax  # comp_aga
    '''
    return adert__


def calc_a(dert__):
    """
    Compute vector representation of angle of gradient by normalizing (dy, dx).
    Numpy-broadcasted, first dimension of dert__ is a list of parameters: g, dy, dx
    Example
    -------
    >>> dert1 = np.array([0, 5, 3, 4])
    >>> a1 = calc_a(dert1)
    >>> print(a1)
    array([0.6, 0.8])
    >>> # 45 degrees angle
    >>> dert2 = np.array([0, 450**0.5, 15, 15])
    >>> a2 = calc_a(dert2)
    >>> print(a2)
    array([0.70710678, 0.70710678])
    >>> print(np.degrees(np.arctan2(*a2)))
    45.0
    >>> # -30 (or 330) degrees angle
    >>> dert3 = np.array([0, 10, -5, 75**0.5])
    >>> a3 = calc_a(dert3)
    >>> print(a3)
    array([-0.5      ,  0.8660254])
    >>> print(np.rad2deg(np.arctan2(*a3)))
    -29.999999999999996
    """
    return dert__[[2, 3]] / dert__[1]  # np.array([dy, dx]) / g


def angle_diff(a2, a1):
    """
    a1, a2: array-like, each contains sine and cosine of corresponding angle
    """
    sin_1 = a1[0]
    sin_2 = a2[0]

    cos_1 = a1[1]
    cos_2 = a2[1]

    # sine and cosine of difference between angles:
    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (sin_1 * cos_1) + (sin_2 * cos_2)

    return ma.array([sin_da, cos_da])

