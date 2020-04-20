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
def comp_r(dert__, fig, root_fcr):
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
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    """

    i__ = dert__[0]   # i is ig if fig else pixel
    '''
    sparse aligned i__center and i__rim arrays:
    '''
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

        idy__, idx__, m__ = dert__[[2, 3, 4]]  # skip g: recomputed, output for Dert only
        dy__ = idy__[1:-1:2, 1:-1:2]  # sparse to align with i__center
        dx__ = idx__[1:-1:2, 1:-1:2]
        m__  = m__[1:-1:2, 1:-1:2]

    else:  # root fork is comp_g or comp_pixel, initialize sparse derivatives:

        dy__ = np.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
        dx__ = np.zeros((i__center.shape[0], i__center.shape[1]))

    if not fig:  # compare four diametrically opposed pairs of rim pixels:

        dt__ = np.stack((i__topleft - i__bottomright,
                         i__top - i__bottom,
                         i__topright - i__bottomleft,
                         i__right - i__left
                       ))
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs[:4], XCOEFs[:4]):

            dy__ += d__ * YCOEF  # decompose differences into dy and dx,
            dx__ += d__ * XCOEF  # accumulate with prior-rng dy, dx

        g__ = np.hypot(dy__, dx__)  # gradient
        '''
        inverse match = SAD, more precise measure of variation than g, direction doesn't matter:
        (all diagonal derivatives can be imported from prior 2x2 comp)
        '''
        if not root_fcr:
            m__ = np.zeros((i__center.shape[0], i__center.shape[1]))
        m__ += ( abs(i__center - i__topleft)
               + abs(i__center - i__top)
               + abs(i__center - i__topright)
               + abs(i__center - i__right)
               + abs(i__center - i__bottomright)
               + abs(i__center - i__bottom)
               + abs(i__center - i__bottomleft)
               + abs(i__center - i__left)
               )

    else:  # fig is TRUE, compare angle and then magnitude of 8 center-rim pairs
        if not root_fcr:
            idy__, idx__ = dert__[[-2, -1]]  # root fork is comp_g, not sparse

        a__ = [idy__, idx__] / i__  # i = ig
        '''
        sparse aligned a__center and a__rim arrays:
        '''
        a__center =      a__[:, 1:-1:2, 1:-1:2]
        a__topleft =     a__[:, :-2:2, :-2:2]
        a__top =         a__[:, :-2:2, 1:-1: 2]
        a__topright =    a__[:, :-2:2, 2::2]
        a__right =       a__[:, 1:-1:2, 2::2]
        a__bottomright = a__[:, 2::2, 2::2]
        a__bottom =      a__[:, 2::2, 1:-1:2]
        a__bottomleft =  a__[:, 2::2, :-2:2]
        a__left =        a__[:, 1:-1:2, :-2:2]
        '''
        8-tuple of differences between center dert angle and rim dert angle:
        '''
        dat__ = np.stack((angle_diff(a__center, a__topleft),
                          angle_diff(a__center, a__top),
                          angle_diff(a__center, a__topright),
                          angle_diff(a__center, a__right),
                          angle_diff(a__center, a__bottomright),
                          angle_diff(a__center, a__bottom),
                          angle_diff(a__center, a__bottomleft),
                          angle_diff(a__center, a__left)
                        ))
        if root_fcr:
            m__, day__, dax__ = dert__[[-4, -2, -1]]  # skip ga: recomputed, output for summation only?
            m__ = m__[1:-1:2, 1:-1:2]  # sparse to align with i__center
            day__ = day__ [1:-1:2, 1:-1:2]
            dax__ = dax__ [1:-1:2, 1:-1:2]
        else:
            m__   = np.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
            day__ = np.zeros((a__center.shape[0], a__center.shape[1], a__center.shape[2]))
            dax__ = np.zeros((a__center.shape[0], a__center.shape[1], a__center.shape[2]))

        for dat_, YCOEF, XCOEF in zip(dat__, YCOEFs, XCOEFs):
            '''
            accumulate in prior-rng (3x3 -> 5x5 -> 9x9) day, dax:
            '''
            day__ += dat_ * YCOEF  # decomposed differences of angle,
            dax__ += dat_ * YCOEF  # accumulate in prior-rng day, dax
        '''
        gradient of angle:
        '''
        ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
        '''
        accumulate match (cosine similarity) in prior-rng (3x3 -> 5x5 -> 9x9) m:
        '''
        m__ +=( np.minimum(i__center, (i__topleft     * dat__[0][1]))
              + np.minimum(i__center, (i__top         * dat__[1][1]))
              + np.minimum(i__center, (i__topright    * dat__[2][1]))
              + np.minimum(i__center, (i__right       * dat__[3][1]))
              + np.minimum(i__center, (i__bottomright * dat__[4][1]))
              + np.minimum(i__center, (i__bottom      * dat__[5][1]))
              + np.minimum(i__center, (i__bottomleft  * dat__[6][1]))
              + np.minimum(i__center, (i__left        * dat__[7][1]))
              )
        '''
        8-tuple of cosine differences per direction:
        '''
        dt__ = np.stack(((i__center - i__topleft     * dat__[0][1]),
                         (i__center - i__top         * dat__[1][1]),
                         (i__center - i__topright    * dat__[2][1]),
                         (i__center - i__right       * dat__[3][1]),
                         (i__center - i__bottomright * dat__[4][1]),
                         (i__center - i__bottom      * dat__[5][1]),
                         (i__center - i__bottomleft  * dat__[6][1]),
                         (i__center - i__left        * dat__[7][1])
                       ))
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs, XCOEFs):

            dy__ += d__ * YCOEF  # y-decomposed center-to-rim difference
            dx__ += d__ * XCOEF  # x-decomposed center-to-rim difference
            '''
            accumulate in prior-rng (3x3 -> 5x5 -> 9x9) dy, dx
            '''
        g__ = np.hypot(dy__, dx__)

    if fig:
        rdert = ma.stack((i__center, g__, dy__, dx__, m__, ga__, *day__, *dax__))
    else:
        rdert = ma.stack((i__center, g__, dy__, dx__, m__))
    '''
    return input dert__ with accumulated derivatives,
    
    next comp_r will use full dert        # comp_rr
    next comp_a will use g__, dy__, dx__  # comp_agr, preserve dy, dx as idy, idx
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

    dert__ = shape_check(dert__)  # remove derts of incomplete kernels
    i__, g__, dy__, dx__, = dert__[0:4]  # input dert = i, g, dy, dx, ?m, ?(ga, day, dax)

    if fga:  # input is adert
        ga__, day__, dax__ = dert__[4:7]
        a__ = [day__, dax__] / ga__
    else:
        a__ = [dy__, dx__] / g__  # similar to calc_a

    # each shifted a in 2x2 kernel
    a__topleft = a__[:, :-1, :-1]
    a__topright = a__[:, :-1, 1:]
    a__botright = a__[:, 1:, 1:]
    a__botleft = a__[:, 1:, :-1]

    # diagonal angle differences:
    sin_da0__, cos_da0__ = angle_diff(a__topleft, a__botright)
    sin_da1__, cos_da1__ = angle_diff(a__topright, a__botleft)

    # ma__ = np.hypot(sin_da0__, cos_da0__) + np.hypot(sin_da1__, cos_da1__)
    # ma = SAD: angle variation is inverse measure of angle match
    # need to covert sin and cos da to 0->2 range?

    day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    # angle gradient, a scalar, to evaluate for comp_aga

    adert__ = ma.stack((i__[:-1, :-1],   # for summation in Dert
                        g__[:-1, :-1],   # for summation in Dert
                        dy__[:-1, :-1],  # passed on as idy
                        dx__[:-1, :-1],  # passed on as idx
                        # m__[:-1, :-1],   # next fork criterion
                        ga__,
                        day__[0],
                        day__[1],
                        dax__[0],
                        dax__[1],
                        # ma__,
                        cos_da0__,
                        cos_da1__
                      ))
    '''
    next comp_g will use g, cos_da0__, cos_da1__, dy, dx (passed to comp_rg as idy, idx)
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

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (sin_1 * cos_1) + (sin_2 * cos_2)

    return ma.array([sin_da, cos_da])


def comp_g(dert__):  # add fga if processing in comp_ga is different?
    """
    Cross-comp of g or ga in 2x2 kernels, between derts in ma.stack dert__:
    input dert  = (i, g, dy, dx, m, ga, dyy, dxy, dyx, dxx, cos_da0, cos_da1)
    output dert = (g, gg, dgy, dgx, gm, ga, day, dax, dy, dx)
    """

    dert__ = shape_check(dert__)  # remove derts of incomplete kernels
    g__, cos_da0__, cos_da1__ = dert__[[1, -2, -1]]  # top dimension of numpy stack must be a list

    cos_da0__ = cos_da0__[:-1,:-1]
    cos_da1__ = cos_da1__[:-1,:-1]

    g_topleft__ = g__[:-1, :-1]
    g_topright__ = g__[:-1, 1:]
    g_bottomleft__ = g__[1:, :-1]
    g_bottomright__ = g__[1:, 1:]

    dgy__ = ((g_bottomleft__ + g_bottomright__) -
             (g_topleft__ * cos_da0__ + g_topright__ * cos_da1__))
    # y-decomposed cosine difference between gs

    dgx__ = ((g_topright__ + g_bottomright__) -
             (g_topleft__ * cos_da0__ + g_bottomleft__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g_topleft__, (g_bottomright__ * cos_da0__))  # g match = min(g, _g*cos(da))
    mg1__ = np.minimum(g_topright__, (g_bottomleft__ * cos_da1__))
    mg__  = mg0__ + mg1__

    gdert = ma.stack((g__[:-1, :-1],  # remove last row and column to align with derived params
                     gg__,
                     dgy__,
                     dgx__,
                     mg__,
                     dert__[5][:-1, :-1],  # ga__
                     dert__[6][:-1, :-1],  # dyy: dy_dy
                     dert__[7][:-1, :-1],  # dxy: dd_dy
                     dert__[8][:-1, :-1],  # dyx: dy_dx
                     dert__[9][:-1, :-1],  # dxx: dx_dx
                     dert__[10][:-1, :-1],  # ma__
                     dert__[2][:-1, :-1],  # idy__
                     dert__[3][:-1, :-1]   # idx__
                    ))
    '''
    next comp_r will use g, idy, idx   # comp_rg
    next comp_a will use ga, day, dax  # comp_agg, also dgy__, dgx__ as idy, idx?
    '''
    return gdert

def shape_check(dert__):
    # remove derts of 2x2 kernels that are missing some other derts

    if dert__[0].shape[0] % 2 != 0:
        dert__ = dert__[:, :-1, :]
    if dert__[0].shape[1] % 2 != 0:
        dert__ = dert__[:, :, :-1]

    return dert__

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
    
if isinstance(a__, ma.masked_array):
    a__.data[a__.mask] = np.nan
    a__.mask = ma.nomask
'''
