"""
Cross-comparison of pixels or gradients, in 2x2 or 3x3 kernels
"""

import numpy as np
import functools

# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
            
| Scharr coefs:
# YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
# XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
'''

def comp_r(dert__, fig, root_fcr, mask=None):
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
    i__center =      i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
    i__topleft =     i__[:-2:2, :-2:2]
    i__top =         i__[:-2:2, 1:-1:2]
    i__topright =    i__[:-2:2, 2::2]
    i__right =       i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom =      i__[2::2, 1:-1:2]
    i__bottomleft =  i__[2::2, :-2:2]
    i__left =        i__[1:-1:2, :-2:2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask is not None:
        majority_mask = ( mask[1:-1:2, 1:-1:2]
                        + mask[:-2:2, :-2:2]
                        + mask[:-2:2, 1:-1: 2]
                        + mask[:-2:2, 2::2]
                        + mask[1:-1:2, 2::2]
                        + mask[2::2, 2::2]
                        + mask[2::2, 1:-1:2]
                        + mask[2::2, :-2:2]
                        + mask[1:-1:2, :-2:2]
                        ) > 1
    else:
        majority_mask = None
    # majority_mask is returned at the end of function

    idy__, idx__ = dert__[1][1:-1:2, 1:-1:2], dert__[2][1:-1:2, 1:-1:2]

    if root_fcr:  # root fork is comp_r, accumulate derivatives:

        dy__ = dert__[4][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        dx__ = dert__[5][1:-1:2, 1:-1:2].copy()
        m__ = dert__[6][1:-1:2, 1:-1:2].copy()
    else:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    if not fig:  # compare four diametrically opposed pairs of rim pixels:

        d_tl_br = i__topleft - i__bottomright
        d_t_b   = i__top - i__bottom
        d_tr_bl = i__topright - i__bottomleft
        d_r_l   = i__right - i__left

        dy__ += (d_tl_br * YCOEFs[0] +
                 d_t_b   * YCOEFs[1] +
                 d_tr_bl * YCOEFs[2] +
                 d_r_l   * YCOEFs[3])

        dx__ += (d_tl_br * XCOEFs[0] +
                 d_t_b   * XCOEFs[1] +
                 d_tr_bl * XCOEFs[2] +
                 d_r_l   * XCOEFs[3])

        g__ = np.hypot(dy__, dx__)  # gradient
        '''
        inverse match = SAD, direction-invariant and more precise measure of variation than g
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

        i__[np.where(i__ == 0)] = 1  # to avoid / 0
        sin__ = dert__[1] / i__
        cos__ = dert__[2] / i__
        '''
        sparse aligned a__center and a__rim arrays:
        '''
        sin__center      = sin__[1:-1:2, 1:-1:2]
        sin__topleft     = sin__[:-2:2, :-2:2]
        sin__top         = sin__[:-2:2, 1:-1: 2]
        sin__topright    = sin__[:-2:2, 2::2]
        sin__right       = sin__[1:-1:2, 2::2]
        sin__bottomright = sin__[2::2, 2::2]
        sin__bottom      = sin__[2::2, 1:-1:2]
        sin__bottomleft  = sin__[2::2, :-2:2]
        sin__left        = sin__[1:-1:2, :-2:2]

        cos__center      = cos__[1:-1:2, 1:-1:2]
        cos__topleft     = cos__[:-2:2, :-2:2]
        cos__top         = cos__[:-2:2, 1:-1: 2]
        cos__topright    = cos__[:-2:2, 2::2]
        cos__right       = cos__[1:-1:2, 2::2]
        cos__bottomright = cos__[2::2, 2::2]
        cos__bottom      = cos__[2::2, 1:-1:2]
        cos__bottomleft  = cos__[2::2, :-2:2]
        cos__left        = cos__[1:-1:2, :-2:2]
        ''' 
        only mask kernels with more than one masked dert, for all operations below: 
        '''
        # (mask is summed above

        '''
        8-tuple of differences between central dert angle and rim dert angle:
        '''
        cos_da = [
            ((cos__topleft     * cos__center) + (sin__center * sin__topleft)),
            ((cos__top         * cos__center) + (sin__center * sin__top)),
            ((cos__topright    * cos__center) + (sin__center * sin__topright)),
            ((cos__right       * cos__center) + (sin__center * sin__right)),
            ((cos__bottomright * cos__center) + (sin__center * sin__bottomright)),
            ((cos__bottom      * cos__center) + (sin__center * sin__bottom)),
            ((cos__bottomleft  * cos__center) + (sin__center * sin__bottomleft)),
            ((cos__left        * cos__center) + (sin__center * sin__left))
        ]
        '''
        8-tuple of cosine matches per direction:
        '''
        m__ += (  np.minimum(i__center, i__topleft     * cos_da[0])
                + np.minimum(i__center, i__top         * cos_da[1])
                + np.minimum(i__center, i__topright    * cos_da[2])
                + np.minimum(i__center, i__right       * cos_da[3])
                + np.minimum(i__center, i__bottomright * cos_da[4])
                + np.minimum(i__center, i__bottom      * cos_da[5])
                + np.minimum(i__center, i__bottomleft  * cos_da[6])
                + np.minimum(i__center, i__left        * cos_da[7])
                  )
        '''
        8-tuple of cosine differences per direction:
        '''
        dt__ = [
            (i__center - i__topleft     * cos_da[0]),
            (i__center - i__top         * cos_da[1]),
            (i__center - i__topright    * cos_da[2]),
            (i__center - i__right       * cos_da[3]),
            (i__center - i__bottomright * cos_da[4]),
            (i__center - i__bottom      * cos_da[5]),
            (i__center - i__bottomleft  * cos_da[6]),
            (i__center - i__left        * cos_da[7])
        ]
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs, XCOEFs):

            dy__ += d__ * YCOEF  # decompose differences into dy and dx,
            dx__ += d__ * XCOEF  # accumulate with prior-rng dy, dx
            '''
            accumulate in prior-range dy, dx: 3x3 -> 5x5 -> 9x9 
            '''
        g__ = np.hypot(dy__, dx__)

    '''
    next comp_r will use full dert       
    next comp_g will use g__, dy__, dx__
    '''
    return (i__center, idy__, idx__, g__, dy__, dx__, m__), majority_mask


def comp_g(dert__, mask=None):  # cross-comp of g in 2x2 kernels, between derts in np.stack dert__

    # Unpack relevant params
    g__, dy__, dx__ = dert__[3], dert__[4], dert__[5]    # g, dy, dx -> local i, idy, idx
    g__[np.where(g__ == 0)] = 1  # replace 0 values with 1 to avoid error, not needed in high-g blobs?
    ''' 
    for all operations below: only mask kernels with more than one masked dert 
    '''
    if mask is not None:
        majority_mask = (mask[:-1, :-1] +
                         mask[:-1, 1:] +
                         mask[1:, 1:] +
                         mask[1:, :-1]
                         ) > 1
    else:
        majority_mask = None

    g0__, dy0__, dx0__ = g__[:-1, :-1], dy__[:-1, :-1], dx__[:-1, :-1]  # top left
    g1__, dy1__, dx1__ = g__[:-1, 1:],  dy__[:-1, 1:],  dx__[:-1, 1:]   # top right
    g2__, dy2__, dx2__ = g__[1:, 1:],   dy__[1:, 1:],   dx__[1:, 1:]    # bottom right
    g3__, dy3__, dx3__ = g__[1:, :-1],  dy__[1:, :-1],  dx__[1:, :-1]   # bottom left

    sin0__ = dy0__ / g0__;  cos0__ = dx0__ / g0__
    sin1__ = dy1__ / g1__;  cos1__ = dx1__ / g1__
    sin2__ = dy2__ / g2__;  cos2__ = dx2__ / g2__
    sin3__ = dy3__ / g3__;  cos3__ = dx3__ / g3__

    '''
    cosine of difference between diagonally opposite angles, in vector representation
    print(cos_da1__.shape, type(cos_da1__))
    '''
    cos_da0__ = (cos2__ * cos0__) + (sin2__ * sin0__)  # top left to bottom right
    cos_da1__ = (cos3__ * cos1__) + (sin3__ * sin1__)  # top right to bottom left

    dgy__ = ((g3__ + g2__) - (g0__ * cos_da0__ + g1__ * cos_da0__))
    # y-decomposed cosine difference between gs
    dgx__ = ((g1__ + g2__) - (g0__ * cos_da0__ + g3__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g0__, g2__) * (cos_da1__+1)  # +1 to make all positive
    mg1__ = np.minimum(g1__, g3__) * (cos_da1__+1)
    mg__  = mg0__ + mg1__  # match of gradient

    ig__ = g__ [:-1, :-1]  # remove last row and column to align with derived params
    idy__ = dy__[:-1, :-1]
    idx__ = dx__[:-1, :-1]  # -> idy, idx to compute cos for comp rg

    '''
    next comp_rg will use g, dy, dx
    next comp_gg will use gg, dgy, dgx
    '''
    return (ig__, idy__, idx__, gg__, dgy__, dgx__, mg__), majority_mask  # return new dert, along with summed mask