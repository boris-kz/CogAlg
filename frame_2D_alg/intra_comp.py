"""
Cross-comparison of pixels, angles, or gradients, in 2x2 or 3x3 kernels
"""
import numpy as np
import numpy.ma as ma

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
'''
# ------------------------------------------------------------------------------------
# Functions
def comp_r_draft(dert__, fig, root_fcr):
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
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3.
    """

    i__ = dert__[0]   # i is ig if fig else pixel
    # sparse aligned i__center and i__rim arrays:

    i__center      = i__[1:-1:2, 1:-1:2]
    i__topleft     = i__[:-2:2, :-2:2]
    i__top         = i__[:-2:2, 1:-1:2]
    i__topright    = i__[:-2:2, 2::2]
    i__right       = i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom      = i__[2::2, 1:-1:2]
    i__bottomleft  = i__[2::2, :-2:2]
    i__left        = i__[1:-1:2, :-2:2]

    if root_fcr:  # root fork is comp_r, all params are present in the input:

        dy__, dx__ = dert__[[2, 3]]  # skip g: recomputed, output for summation only?
        dy__ = dy__[1:-1:2, 1:-1:2]  # sparse to align with i__center
        dx__ = dx__[1:-1:2, 1:-1:2]

    else:  # root fork is comp_g or comp_pixel, initialize sparse derivatives:

        dy__ = np.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
        dx__ = np.zeros((i__center.shape[0], i__center.shape[1]))

    if not fig:  # compare four diametrically opposed pairs of rim pixels:

        dt__ = np.stack((i__topleft - i__bottomright,
                         i__top - i__bottom,
                         i__topright - i__bottomleft,
                         i__right - i__left
                       ))
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs[:4], XCOEFs[:4]):  # is [:4] correct here?

            dy__ += d__ * YCOEF  # decompose differences into dy and dx,
            dx__ += d__ * XCOEF  # accumulate with prior-rng dy, dx

        g__ = np.hypot(dy__, dx__)  # gradient

    else:   # i is ig:
        a__ = [dy__, dx__] / i__  # dy, dx here are idy, idx: not accumulated yet

        a__center =      a__[:, 1:-1:2, 1:-1:2]
        a__topleft =     a__[:, :-2:2, :-2:2]
        a__top =         a__[:, :-2:2, 1:-1: 2]
        a__topright =    a__[:, :-2:2, 2::2]
        a__right =       a__[:, 1:-1:2, 2::2]
        a__bottomright = a__[:, 2::2, 2::2]
        a__bottom =      a__[:, 2::2, 1:-1:2]
        a__bottomleft =  a__[:, 2::2, :-2:2]
        a__left =        a__[:, 1:-1:2, :-2:2]

        # tuple of differences between center and rim angles per direction:

        dat__ = np.stack(angle_diff(a__center, a__topleft),
                         angle_diff(a__center, a__top),
                         angle_diff(a__center, a__topright),
                         angle_diff(a__center, a__right),
                         angle_diff(a__center, a__bottomright),
                         angle_diff(a__center, a__bottom),
                         angle_diff(a__center, a__bottomleft),
                         angle_diff(a__center, a__left)
                        )
        if root_fcr:
            m__, day__, dax__ = dert__[[-4, -2, -1]]  # skip ga: recomputed, output for summation only?
            m__ = m__[1:-1:2, 1:-1:2]  # sparse to align with i__center
        else:
            m__ = np.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
            day__ = np.zeros((a__center.shape[0], a__center.shape[1], a__center.shape[2]))
            dax__ = np.zeros((a__center.shape[0], a__center.shape[1], a__center.shape[2]))

        for dat_, YCOEF, XCOEF in zip(dat__, YCOEFs, XCOEFs):

            day__ += dat_ * YCOEF  # decomposed differences of angle,
            dax__ += dat_ * YCOEF  # accumulate in prior-rng day, dax

        # gradient of angle
        ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))

        # accumulate match (cosine similarity) in prior-rng m (3x3 -> 5x5 -> 9x9):

        m__ +=( np.minimum(i__center, (i__topleft     * dat__[1][:, :, 0]))
              + np.minimum(i__center, (i__top         * dat__[1][:, :, 1]))
              + np.minimum(i__center, (i__topright    * dat__[1][:, :, 2]))
              + np.minimum(i__center, (i__right       * dat__[1][:, :, 3]))
              + np.minimum(i__center, (i__bottomright * dat__[1][:, :, 4]))
              + np.minimum(i__center, (i__bottom      * dat__[1][:, :, 5]))
              + np.minimum(i__center, (i__bottomleft  * dat__[1][:, :, 6]))
              + np.minimum(i__center, (i__left        * dat__[1][:, :, 7]))
              )
        # tuple of cosine differences per direction:

        dt__ = np.stack(((i__center - i__topleft     * dat__[1][:, :, 0]),
                         (i__center - i__top         * dat__[1][:, :, 1]),
                         (i__center - i__topright    * dat__[1][:, :, 2]),
                         (i__center - i__right       * dat__[1][:, :, 3]),
                         (i__center - i__bottomright * dat__[1][:, :, 4]),
                         (i__center - i__bottom      * dat__[1][:, :, 5]),
                         (i__center - i__bottomleft  * dat__[1][:, :, 6]),
                         (i__center - i__left        * dat__[1][:, :, 7])
                       ))
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs, XCOEFs):  # accumulate in prior-rng dy, dx:

            dy__ += d__ * YCOEF  # y-decomposed center-to-rim difference
            dx__ += d__ * XCOEF  # x-decomposed center-to-rim difference

        g__ = np.hypot(dy__, dx__)

    # return dert__ with accumulated derivatives:
    if fig:
        rdert = i__, g__, dy__, dx__, m__, ga__, *day__, *dax__
    else:
        rdert = i__, g__, dy__, dx__
    '''
    next comp_r will use full dert        # comp_rr
    next comp_a will use g__, dy__, dx__  # comp_agr, preserve dy, dx as idy, idx?
    '''
    return rdert


def comp_a(dert__, fga):
    """
    cross-comp of a or aga in 2x2 kernels
    ----------
    input dert__ : array-like
    fga : bool
        If True, dert structure is interpreted as:
        (g, gg, dgy, dgx, gm, iga, iday, idax)
        else: (i, g, dy, dx, m)
    ----------
    output adert: masked_array of aderts,
    adert: (i, g, dy, dx, m, ga, day, dax, cos_da0, cos_da1)
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fga = 'specific value'
    >>> comp_a(dert__, fga)
    """
    # input dert = (i,  g,  dy,  dx, ?( m, ga, day, dax))
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

    # diagonal angle differences:
    sin_da0__, cos_da0__ = angle_diff(a__topleft, a__botright)
    sin_da1__, cos_da1__ = angle_diff(a__topright, a__botleft)

    day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    # angle gradient, a scalar

    adert__ = ma.stack((i__[:-1, :-1],   # for summation in Dert
                        g__[:-1, :-1],   # for summation in Dert
                        dy__[:-1, :-1],  # passed on as idy
                        dx__[:-1, :-1],  # passed on as idx
                        m__[:-1, :-1],   # for summation in Dert
                        ga__,
                        *day__,
                        *dax__,
                        cos_da0__,
                        cos_da1__
                      ))
    '''
    next comp_g will use g, cos_da0__, cos_da1__, dy, dx (passed to comp_rg as idy, idx)
    next comp_a will use ga, day, dax  # comp_aga
    '''
    return adert__


def comp_a_chee(dert__, fga):
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

    a_rim__ = np.stack((a__topleft,
                        a__topright,
                        a__botright,
                        a__botleft))

    # preallocate size of arrays
    sin_da__ = [None] * 2
    cos_da__ = [None] * 2
    day__ = [None] * 2
    day__[0] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))
    day__[1] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))
    dax__ = [None] * 2
    dax__[0] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))
    dax__[1] = np.zeros((a__topleft.shape[1], a__topleft.shape[2]))

    YCOEF_ = -1, -1
    XCOEF_ = -1, 1

    for i in range(2):
        # opposing difference
        sin_da__[i], cos_da__[i] = angle_diff(a_rim__[i], a_rim__[i + 2])

        # sine part of day
        day__[0] += sin_da__[i] * YCOEF_[i]
        # cosine part of day
        day__[1] += cos_da__[i]  # no sign based coefficient for cosine part
        # sine part of dax
        dax__[0] += sin_da__[i] * XCOEF_[i]
        # cosine part of dax
        dax__[1] += cos_da__[i]  # no sign based coefficient for cosine part

    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    # angle gradient, a scalar

    # remove last row and column to solve dimension mismatch issue
    adert__ = ma.stack((i__[:-1, :-1],  # for summation in Dert
                        g__[:-1, :-1],
                        dy__[:-1, :-1],
                        dx__[:-1, :-1],
                        m__[:-1, :-1],   # for summation in Dert
                        ga__,
                        *day__,
                        *dax__,
                        cos_da__[0],
                        cos_da__[1]))
    '''
    next comp_g will use g, cos_da0__, cos_da1__ 
    next comp_a will use ga, day, dax  # comp_aga
    maybe passed to deeper comp_rg: dy as idy, dx as idx? 
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


def comp_g(dert__):  # add fga if processing in comp_ga is different?
    """
    Cross-comp of g or ga in 2x2 kernels, between derts in ma.stack dert__:
    input dert  = (i, g, dy, dx, m, ga, day, dax, cos_da0, cos_da1)
    output dert = (g, gg, dgy, dgx, gm, ga, day, dax, dy, dx)
    """
    g__, cos_da0__, cos_da1__ = dert__[[1, -2, -1]]  # top dimension of numpy stack must be a list

    # this mask section would need further test later with actual input from frame_blobs
    if isinstance(g__, ma.masked_array):
        g__.data[g__.mask] = np.nan
        g__.mask = ma.nomask

    g_topleft__ = g__[:-1, :-1]
    g_topright__ = g__[:-1, 1:]
    g_bottomleft__ = g__[1:, :-1]
    g_bottomright__ = g__[1:, 1:]

    dgy__ = ((g_bottomleft__ + g_bottomright__) -
             (g_topleft__ * cos_da0__ + g_topright__ * cos_da1__))
    # y-decomposed difference between gs

    dgx__ = ((g_topright__ + g_bottomright__) -
             (g_topleft__ * cos_da0__ + g_bottomleft__ * cos_da1__))
    # x-decomposed difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g_topleft__, (g_bottomright__ * cos_da0__))  # g match = min(g, _g*cos(da))
    mg1__ = np.minimum(g_topright__, (g_bottomleft__ * cos_da1__))
    mg__  = mg0__ + mg1__

    gdert = ma.stack(g__[:-1, :-1],  # remove last row and column to align with derived params
                     gg__,
                     dgy__,
                     dgx__,
                     mg__,
                     dert__[5],  # ga__
                     dert__[6],  # day__
                     dert__[7],  # dax__
                     dert__[8][:-1, :-1],  # idy__
                     dert__[9][:-1, :-1]   # idx__
                    )
    '''
    next comp_r will use g, idy, idx   # comp_rg
    next comp_a will use ga, day, dax  # comp_agg, also dgy__, dgx__ as idy, idx?
    '''
    return gdert

''' old comp_a:

day__ = ( Y_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
          Y_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
        )
dax__ = ( X_COEFFS[0][0] * angle_diff(a__topleft, a__bottomright) +
          X_COEFFS[0][1] * angle_diff(a__topright, a__bottomleft)
        )
        
2x2 COEFFS:
Y: np.array([-2, -2, 2, 2])
X: np.array([-2, 2, 2, -2])

roll axis to align COEFFs with dat__: move 1st axis to 4th axis,
for broadcasting 4 pairs of 8 directionals with coefficients:
        
    dat__ = np.rollaxis(dat__, 0, 4)
    day__ = (dat__ * YCOEFs).sum(axis=-1)
    dax__ = (dat__ * XCOEFs).sum(axis=-1)

    dt__ = np.rollaxis(dt__, 0, 3)
    gy__ += (dt__ * YCOEFs).sum(axis=-1)
    gx__ += (dt__ * XCOEFs).sum(axis=-1)
'''

def comp_3x3(image):  # Deprecated, from frame_blobs' comp_pixel, Khanh
    # comparison between pairs of diametrically opposed pixels:

    d___ = np.array(  # center pixels - translated rim pixels:
        [image[ts2] - image[ts1] for ts1, ts2 in TRANSLATING_SLICES_PAIRS_3x3]
    ).swapaxes(0, 2).swapaxes(0, 1)

    # Decompose differences into dy and dx, same as Gy and Gx in conventional edge detection operators:
    dy__ = (d___ * YCOEF).sum(axis=2)
    dx__ = (d___ * XCOEF).sum(axis=2)
    p__ = image[1:-1, 1:-1]
    g__ = np.hypot(dy__, dx__)  # compute gradients per kernel, converted to 0-255 range

    return ma.stack((p__, g__, dy__, dx__))
