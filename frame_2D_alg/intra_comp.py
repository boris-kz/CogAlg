"""
Cross-comparison of pixels or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import numpy.ma as ma
import functools

# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
'''

def comp_r(dert__, fig, root_fcr):
    '''
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
    '''

    i__ = dert__[0]  # i is ig if fig else pixel
    '''
    sparse aligned i__center and i__rim arrays:
    '''
    i__center =      i__[1:-1:2, 1:-1:2]
    i__topleft =     i__[:-2:2, :-2:2]
    i__top =         i__[:-2:2, 1:-1:2]
    i__topright =    i__[:-2:2, 2::2]
    i__right =       i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom =      i__[2::2, 1:-1:2]
    i__bottomleft =  i__[2::2, :-2:2]
    i__left =        i__[1:-1:2, :-2:2]

    idy__, idx__ = dert__[[1, 2]]

    '''
    Unmasking is not needed here, comp_r is only done in large low-variation areas, where border won't make any difference
     
    i__center.mask = i__topleft.mask = i__top.mask = i__topright.mask = i__right.mask = i__bottomright.mask = i__bottom.mask= \
    i__bottomleft.mask = i__left.mask = \
        mask_SUM(
        [i__center.mask, i__topleft.mask, i__top.mask, i__topright.mask, i__right.mask, i__bottomright.mask,
         i__bottom.mask, i__bottomleft.mask, i__left.mask])
    '''

    if root_fcr:  # root fork is comp_r, accumulate derivatives:

        dy__, dx__, m__ = dert__[[4, 5, 6]]
        dy__ = dy__[1:-1:2, 1:-1:2]  # sparse to align with i__center
        dx__ = dx__[1:-1:2, 1:-1:2]
        m__  =  m__[1:-1:2, 1:-1:2]
        dy__.mask = dx__.mask = m__.mask = i__center.mask

    else:   # root fork is comp_g or comp_pixel, initialize sparse derivatives:

        dy__ = np.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
        dx__ = np.zeros((i__center.shape[0], i__center.shape[1]))
        m__ = np.zeros((i__center.shape[0], i__center.shape[1]))

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
        inverse match = SAD, more precise measure of variation than g, direction-invariant
        (all diagonal derivatives can be imported from prior 2x2 comp)
        '''
        m__ +=( abs(i__center - i__topleft)
              + abs(i__center - i__top)
              + abs(i__center - i__topright)
              + abs(i__center - i__right)
              + abs(i__center - i__bottomright)
              + abs(i__center - i__bottom)
              + abs(i__center - i__bottomleft)
              + abs(i__center - i__left)
              )

    else:  # fig is TRUE, compare angle and then magnitude of 8 center-rim pairs
        # replace float with int

        i__[np.where(i__ == 0)] = .001
        a__ = [idy__, idx__] / i__  # sin, cos;  i = ig
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
        cos_da = np.stack((
                  ((a__center[0] * a__center[1]) + (a__topleft[0] * a__topleft[1])),
                  ((a__center[0] * a__center[1]) + (a__top[0] *  a__top[1])),
                  ((a__center[0] * a__center[1]) + (a__topright[0] * a__topright[1])),
                  ((a__center[0] * a__center[1]) + (a__right[0] * a__right[1])),
                  ((a__center[0] * a__center[1]) + (a__bottomright[0] * a__bottomright[1])),
                  ((a__center[0] * a__center[1]) + (a__bottom[0] * a__bottom[1])),
                  ((a__center[0] * a__center[1]) + (a__bottomleft[0] * a__bottomleft[1])),
                  ((a__center[0] * a__center[1]) + (a__left[0] * a__left[1]))
                ))
        '''
        8-tuple of cosine matches per direction:
        '''
        m__ += (  np.minimum(i__center, i__topleft)    * cos_da[0]
                + np.minimum(i__center, i__top )       * cos_da[1]
                + np.minimum(i__center, i__topright)   * cos_da[2]
                + np.minimum(i__center, i__right)      * cos_da[3]
                + np.minimum(i__center, i__bottomright)* cos_da[4]
                + np.minimum(i__center, i__bottom)     * cos_da[5]
                + np.minimum(i__center, i__bottomleft) * cos_da[6]
                + np.minimum(i__center, i__left)       * cos_da[7]
                )
        '''
        8-tuple of cosine differences per direction:
        '''
        dt__ = np.stack(((i__center - i__topleft     * cos_da[0]),
                         (i__center - i__top         * cos_da[1]),
                         (i__center - i__topright    * cos_da[2]),
                         (i__center - i__right       * cos_da[3]),
                         (i__center - i__bottomright * cos_da[4]),
                         (i__center - i__bottom      * cos_da[5]),
                         (i__center - i__bottomleft  * cos_da[6]),
                         (i__center - i__left        * cos_da[7])
                         ))

        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs, XCOEFs):

            dy__ += d__ * YCOEF  # decompose differences into dy and dx,
            dx__ += d__ * XCOEF  # accumulate with prior-rng dy, dx
            '''
            accumulate in prior-rng (3x3 -> 5x5 -> 9x9) dy, dx
            '''
        g__ = np.hypot(dy__, dx__)
    '''
    next comp_r will use full dert       
    next comp_g will use g__, dy__, dx__
    '''
    return ma.stack((i__center,
                     idy__[1:-1:2, 1:-1:2],  # i__center.shape
                     idx__[1:-1:2, 1:-1:2],  # i__center.shape
                     g__,
                     dy__,
                     dx__,
                     m__
                     ))

def comp_g(dert__):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__

    dert__ = shape_check(dert__)  # remove derts of incomplete kernels

    g__, dy__, dx__ = dert__[[3, 4, 5]]  # g, dy, dx -> local i, idy, idx

    g0__, dy0__, dx0__ = g__[:-1, :-1], dy__[:-1, :-1], dx__[:-1, :-1]  # top left
    g1__, dy1__, dx1__ = g__[:-1, 1:],  dy__[:-1, 1:],  dx__[:-1, 1:]   # top right
    g2__, dy2__, dx2__ = g__[1:, 1:],   dy__[1:, 1:],   dx__[1:, 1:]    # bottom right
    g3__, dy3__, dx3__ = g__[1:, :-1],  dy__[1:, :-1],  dx__[1:, :-1]   # bottom left

    g0__.mask  = mask_SUM([g0__.mask, g1__.mask, g2__.mask, g3__.mask])
    # aren't the masks below identical?:
    dy0__.mask = mask_SUM([dy0__.mask, dy1__.mask, dy2__.mask, dy3__.mask])
    dx0__.mask = mask_SUM([dx0__.mask, dx1__.mask, dx2__.mask, dx3__.mask])

    g1__.mask = g2__.mask = g3__.mask = g0__.mask
    dy1__.mask = dy2__.mask = dy3__.mask = dy0__.mask
    dx1__.mask = dx2__.mask = dx3__.mask = dx0__.mask

    g0__[np.where(g0__ == 0)] = .001
    g1__[np.where(g1__ == 0)] = .001
    g2__[np.where(g2__ == 0)] = .001
    g3__[np.where(g3__ == 0)] = .001

    sin0__ = dy0__ / g0__;  cos0__ = dx0__ / g0__
    sin1__ = dy1__ / g1__;  cos1__ = dx1__ / g1__
    sin2__ = dy2__ / g2__;  cos2__ = dx2__ / g2__
    sin3__ = dy3__ / g3__;  cos3__ = dx3__ / g3__

    '''
    cosine of difference between diagonally opposed angles, in vector representation
    print(cos_da1__.shape, type(cos_da1__))
    '''
    cos_da0__ = (sin0__ * cos0__) + (sin2__ * cos2__)  # top left to bottom right
    cos_da1__ = (sin1__ * cos1__) + (sin3__ * cos3__)  # top right to bottom left

    dgy__ = ((g3__ + g2__) - (g0__ * cos_da0__ + g1__ * cos_da1__))
    # y-decomposed cosine difference between gs

    dgx__ = ((g1__ + g2__) - (g0__ * cos_da0__ + g3__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g0__, g2__) * cos_da0__  # g match = min(g, _g) *cos(da)
    mg1__ = np.minimum(g1__, g3__) * cos_da1__
    mg__  = mg0__ + mg1__

    '''
    next comp_rg will use g, dy, dx     
    next comp_gg will use gg, dgy, dgx  
    '''
    # Do we need shape adjustment after mask_SUM?:

    return  ma.stack((g__ [:-1, :-1],  # remove last row and column to align with derived params
                      dy__[:-1, :-1],
                      dx__[:-1, :-1],  # -> idy, idx to compute cos for comp rg
                      gg__,
                      dgy__,
                      dgx__,
                      mg__
                      ))

def shape_check(dert__):
    # remove derts of 2x2 kernels that are missing some other derts

    if dert__[0].shape[0] % 2 != 0:
        dert__ = dert__[:, :-1, :]
    if dert__[0].shape[1] % 2 != 0:
        dert__ = dert__[:, :, :-1]

    return dert__


def normalization(array):
    start = 1
    end = 255
    width = end - start
    res = (array - array.min()) / (array.max() - array.min()) * width + start
    return res


def mask_SUM(list_of_arrays):  # sum of masks converted to int

    sum = functools.reduce(lambda x1, x2: x1.astype('int') + x2.astype('int'), list_of_arrays)
    mask = sum > 1  # mask output if more than 1 input is masked

    return mask

'''
OR_mask=  functools.reduce(np.logical_or, (i__center, i__topleft, i__top,\
                                           i__topright, i__right, i__bottomright,\
                                           i__bottom, i__bottomleft, i__left ))
                                           
a__center.mask = a__topleft.mask = a__top.mask = a__topright.mask = a__right.mask = a__bottomright.mask = \
a__bottom.mask = a__bottomleft.mask = a__left.mask = \
    mask_SUM(
    [a__center.mask, a__topleft.mask, a__top.mask, a__topright.mask, a__right.mask, a__bottomright.mask,
     a__bottom.mask, a__bottomleft.mask, a__left.mask])
'''