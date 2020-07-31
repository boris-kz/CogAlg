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
    # initialize new dert structure
    new_dert__ = ma.zeros((dert__.shape[0],
                          (dert__.shape[1] - 1) // 2,
                          (dert__.shape[2] - 1) // 2),
                          dtype=dert__.dtype)
    new_dert__.mask = True
    # extract new_dert__ 'views', use [:] to 'update' views and new_dert__ at the same time
    i__center, idy__, idx__, g__, dy__, dx__, m__ = new_dert__

    i__ = dert__[0]  # i is ig if fig else pixel

    '''
    sparse aligned i__center and i__rim arrays:
    '''
    i__center[:] =   i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
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
    majority_mask = ( i__[:, 1:-1:2, 1:-1:2].mask.astype(int)
                    + i__[:, :-2:2, :-2:2].mask.astype(int)
                    + i__[:, :-2:2, 1:-1: 2].mask.astype(int)
                    + i__[:, :-2:2, 2::2].mask.astype(int)
                    + i__[:, 1:-1:2, 2::2].mask.astype(int)
                    + i__[:, 2::2, 2::2].mask.astype(int)
                    + i__[:, 2::2, 1:-1:2].mask.astype(int)
                    + i__[:, 2::2, :-2:2].mask.astype(int)
                    + i__[:, 1:-1:2, :-2:2].mask.astype(int)
                    ) > 1
    i__center.mask = i__topleft.mask = i__top.mask = i__topright.mask = i__right.mask = i__bottomright.mask = \
    i__bottom.mask = i__bottomleft.mask = i__left.mask = majority_mask  # not only i__center

    idy__[:], idx__[:] = dert__[[1, 2], 1:-1:2, 1:-1:2]

    if root_fcr:  # root fork is comp_r, accumulate derivatives:

        dy__[:] = dert__[4, 1:-1:2, 1:-1:2]  # sparse to align with i__center
        dx__[:] = dert__[5, 1:-1:2, 1:-1:2]
        m__[:] = dert__[6, 1:-1:2, 1:-1:2]

    dy__.mask = dx__.mask = m__.mask = majority_mask

    if not fig:  # compare four diametrically opposed pairs of rim pixels:

        d_tl_br = i__topleft.data - i__bottomright.data
        d_t_b   = i__top.data - i__bottom.data
        d_tr_bl = i__topright.data - i__bottomleft.data
        d_r_l   = i__right.data - i__left.data

        dy__ += (d_tl_br * YCOEFs[0] +
                 d_t_b   * YCOEFs[1] +
                 d_tr_bl * YCOEFs[2] +
                 d_r_l   * YCOEFs[3])

        dx__ += (d_tl_br * XCOEFs[0] +
                 d_t_b   * XCOEFs[1] +
                 d_tr_bl * XCOEFs[2] +
                 d_r_l   * XCOEFs[3])

        g__[:] = ma.hypot(dy__, dx__)  # gradient
        '''
        inverse match = SAD, direction-invariant and more precise measure of variation than g
        (all diagonal derivatives can be imported from prior 2x2 comp)
        '''
        m__ +=( abs(i__center.data - i__topleft.data)
              + abs(i__center.data - i__top.data)
              + abs(i__center.data - i__topright.data)
              + abs(i__center.data - i__right.data)
              + abs(i__center.data - i__bottomright.data)
              + abs(i__center.data - i__bottom.data)
              + abs(i__center.data - i__bottomleft.data)
              + abs(i__center.data - i__left.data)
              )

    else:  # fig is TRUE, compare angle and then magnitude of 8 center-rim pairs

        i__[ma.where(i__ == 0)] = 1  # to avoid / 0
        a__ = dert__[[1, 2]] / i__  # sin = idy / i, cos = idx / i, i = ig
        '''
        sparse aligned a__center and a__rim arrays:
        '''
        a__center      = a__[:, 1:-1:2, 1:-1:2]
        a__topleft     = a__[:, :-2:2, :-2:2]
        a__top         = a__[:, :-2:2, 1:-1: 2]
        a__topright    = a__[:, :-2:2, 2::2]
        a__right       = a__[:, 1:-1:2, 2::2]
        a__bottomright = a__[:, 2::2, 2::2]
        a__bottom      = a__[:, 2::2, 1:-1:2]
        a__bottomleft  = a__[:, 2::2, :-2:2]
        a__left        = a__[:, 1:-1:2, :-2:2]
        ''' 
        only mask kernels with more than one masked dert, for all operations below: 
        '''
        majority_mask_a = ( a__[:, 1:-1:2, 1:-1:2].mask.astype(int)
                          + a__[:, :-2:2, :-2:2].mask.astype(int)
                          + a__[:, :-2:2, 1:-1: 2].mask.astype(int)
                          + a__[:, :-2:2, 2::2].mask.astype(int)
                          + a__[:, 1:-1:2, 2::2].mask.astype(int)
                          + a__[:, 2::2, 2::2].mask.astype(int)
                          + a__[:, 2::2, 1:-1:2].mask.astype(int)
                          + a__[:, 2::2, :-2:2].mask.astype(int)
                          + a__[:, 1:-1:2, :-2:2].mask.astype(int)
                          ) > 1
        a__center.mask = a__topleft.mask = a__top.mask = a__topright.mask = a__right.mask = a__bottomright.mask = \
        a__bottom.mask = a__bottomleft.mask = a__left.mask = majority_mask_a

        assert (majority_mask_a[0] == majority_mask_a[1]).all()
        dy__.mask = dx__.mask = m__.mask = majority_mask_a[0]

        '''
        8-tuple of differences between central dert angle and rim dert angle:
        '''
        cos_da = [
            ((a__topleft[1].data     * a__center[1].data) + (a__center[0].data * a__topleft[0].data)),
            ((a__top[1].data         * a__center[1].data) + (a__center[0].data * a__top[0].data)),
            ((a__topright[1].data    * a__center[1].data) + (a__center[0].data * a__topright[0].data)),
            ((a__right[1].data       * a__center[1].data) + (a__center[0].data * a__right[0].data)),
            ((a__bottomright[1].data * a__center[1].data) + (a__center[0].data * a__bottomright[0].data)),
            ((a__bottom[1].data      * a__center[1].data) + (a__center[0].data * a__bottom[0].data)),
            ((a__bottomleft[1].data  * a__center[1].data) + (a__center[0].data * a__bottomleft[0].data)),
            ((a__left[1].data        * a__center[1].data) + (a__center[0].data * a__left[0].data))
        ]
        '''
        8-tuple of cosine matches per direction:
        '''
        m__ += (  ma.minimum(i__center.data, i__topleft.data)     * cos_da[0]
                + ma.minimum(i__center.data, i__top.data)         * cos_da[1]
                + ma.minimum(i__center.data, i__topright.data)    * cos_da[2]
                + ma.minimum(i__center.data, i__right.data)       * cos_da[3]
                + ma.minimum(i__center.data, i__bottomright.data) * cos_da[4]
                + ma.minimum(i__center.data, i__bottom.data)      * cos_da[5]
                + ma.minimum(i__center.data, i__bottomleft.data)  * cos_da[6]
                + ma.minimum(i__center.data, i__left.data)        * cos_da[7]
                )
        '''
        8-tuple of cosine differences per direction:
        '''
        dt__ = [
            (i__center.data - i__topleft.data     * cos_da[0]),
            (i__center.data - i__top.data         * cos_da[1]),
            (i__center.data - i__topright.data    * cos_da[2]),
            (i__center.data - i__right.data       * cos_da[3]),
            (i__center.data - i__bottomright.data * cos_da[4]),
            (i__center.data - i__bottom.data      * cos_da[5]),
            (i__center.data - i__bottomleft.data  * cos_da[6]),
            (i__center.data - i__left.data        * cos_da[7])
        ]
        for d__, YCOEF, XCOEF in zip(dt__, YCOEFs, XCOEFs):

            dy__ += d__ * YCOEF  # decompose differences into dy and dx,
            dx__ += d__ * XCOEF  # accumulate with prior-rng dy, dx
            '''
            accumulate in prior-range dy, dx: 3x3 -> 5x5 -> 9x9 
            '''
        g__[:] = ma.hypot(dy__, dx__)

    '''
    next comp_r will use full dert       
    next comp_g will use g__, dy__, dx__
    '''
    return new_dert__  # new_dert__ has been updated along with 'view' arrays: i__center, idy__, idx__, g__, dy__, dx__, m__


def comp_g(dert__, mask=None):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__
    new_shape = np.subtract(dert__[0].shape, 1)
    # initialize return variable
    new_dert__ = tuple(np.zeros(new_shape) for derts in dert__)
        # ma.zeros((dert__.shape[0], dert__.shape[1] - 1, dert__.shape[2] - 1))
    ig__, idy__, idx__, gg__, dgy__, dgx__, mg__ = new_dert__  # assign 'views'. Use [:] to update views

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
        new_mask = majority_mask
    else:
        new_mask = None

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

    dgy__[:] = ((g3__ + g2__) - (g0__ * cos_da0__ + g1__ * cos_da0__))
    # y-decomposed cosine difference between gs
    dgx__[:] = ((g1__ + g2__) - (g0__ * cos_da0__ + g3__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__[:] = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g0__, g2__) * (cos_da1__+1)  # +1 to make all positive
    mg1__ = np.minimum(g1__, g3__) * (cos_da1__+1)
    mg__[:]  = mg0__ + mg1__  # match of gradient

    ig__[:] = g__ [:-1, :-1]  # remove last row and column to align with derived params
    idy__[:] = dy__[:-1, :-1]
    idx__[:] = dx__[:-1, :-1]  # -> idy, idx to compute cos for comp rg

    '''
    next comp_rg will use g, dy, dx
    next comp_gg will use gg, dgy, dgx
    '''
    return new_dert__, new_mask  # new_dert__ has been updated along with 'view' arrays: ig__, idy__, idx__, gg__, dgy__, dgx__, mg__