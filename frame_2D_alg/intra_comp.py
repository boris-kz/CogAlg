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
    unmask all derts in kernels with only one masked dert, can be set to any number of masked derts,
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels.
    unmasked derts were already computed due to extend_dert() in intra_blob   
    '''
    mask_i = mask_SUM([i__center.mask, i__topleft.mask, i__top.mask,
                       i__topright.mask, i__right.mask, i__bottomright.mask,
                       i__bottom.mask, i__bottomleft.mask, i__left.mask])

    i__center.mask = mask_i
    idy__[:], idx__[:] = dert__[[1, 2], 1:-1:2, 1:-1:2]

    if root_fcr:  # root fork is comp_r, accumulate derivatives:

        dy__[:] = dert__[4, 1:-1:2, 1:-1:2]  # sparse to align with i__center
        dx__[:] = dert__[5, 1:-1:2, 1:-1:2]
        m__[:] = dert__[6, 1:-1:2, 1:-1:2]

    dy__.mask = dx__.mask = m__.mask = mask_i

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
        inverse match = SAD, more precise measure of variation than g, direction-invariant
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
        only mask kernels with more than one masked dert 
        '''
        mask_a = mask_SUM([a__center.mask, a__topleft.mask, a__top.mask,
                           a__topright.mask, a__right.mask, a__bottomright.mask,
                           a__bottom.mask, a__bottomleft.mask, a__left.mask])

        assert (mask_a[0] == mask_a[1]).all()
        dy__.mask = dx__.mask = m__.mask = mask_a[0]

        '''
        8-tuple of differences between central dert angle and rim dert angle:
        '''
        cos_da = [  # abs because cos_da should only reduce magnitude of projected d | m, not alter it's sign:

            abs(((a__topleft[1].data     * a__center[1].data) + (a__center[0].data * a__topleft[0].data))),
            abs(((a__top[1].data         * a__center[1].data) + (a__center[0].data * a__top[0].data))),
            abs(((a__topright[1].data    * a__center[1].data) + (a__center[0].data * a__topright[0].data))),
            abs(((a__right[1].data       * a__center[1].data) + (a__center[0].data * a__right[0].data))),
            abs(((a__bottomright[1].data * a__center[1].data) + (a__center[0].data * a__bottomright[0].data))),
            abs(((a__bottom[1].data      * a__center[1].data) + (a__center[0].data * a__bottom[0].data))),
            abs(((a__bottomleft[1].data  * a__center[1].data) + (a__center[0].data * a__bottomleft[0].data))),
            abs(((a__left[1].data        * a__center[1].data) + (a__center[0].data * a__left[0].data)))
        ]
        '''
        8-tuple of cosine matches per direction:
        '''
        m__ += (  ma.minimum(i__center.data, i__topleft.data)    * cos_da[0]
                + ma.minimum(i__center.data, i__top .data)       * cos_da[1]
                + ma.minimum(i__center.data, i__topright.data)   * cos_da[2]
                + ma.minimum(i__center.data, i__right.data)      * cos_da[3]
                + ma.minimum(i__center.data, i__bottomright.data)* cos_da[4]
                + ma.minimum(i__center.data, i__bottom.data)     * cos_da[5]
                + ma.minimum(i__center.data, i__bottomleft.data) * cos_da[6]
                + ma.minimum(i__center.data, i__left.data)       * cos_da[7]
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


def comp_g(dert__):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__

    # initialize return variable
    new_dert__ = ma.zeros((dert__.shape[0], dert__.shape[1] - 1, dert__.shape[2] - 1))
    new_dert__.mask = True  # initialize mask
    ig__, idy__, idx__, gg__, dgy__, dgx__, mg__ = new_dert__  # assign 'views'. Use [:] to update views

    # Unpack relevant params
    g__, dy__, dx__ = dert__[[3, 4, 5]]    # g, dy, dx -> local i, idy, idx
    g__.data[np.where(g__.data == 0)] = 1  # replace 0 values with 1 to avoid error, not needed in high-g blobs?

    g0__, dy0__, dx0__ = g__[:-1, :-1].data, dy__[:-1, :-1].data, dx__[:-1, :-1].data  # top left
    g1__, dy1__, dx1__ = g__[:-1, 1:].data,  dy__[:-1, 1:].data,  dx__[:-1, 1:].data   # top right
    g2__, dy2__, dx2__ = g__[1:, 1:].data,   dy__[1:, 1:].data,   dx__[1:, 1:].data    # bottom right
    g3__, dy3__, dx3__ = g__[1:, :-1].data,  dy__[1:, :-1].data,  dx__[1:, :-1].data   # bottom left
    ''' 
    only mask kernels with more than one masked dert 
    '''
    summed_mask = (g__[:-1, :-1].mask.astype(int) +
                   g__[:-1, 1:].mask.astype(int) +
                   g__[1:, 1:].mask.astype(int) +
                   g__[1:, :-1].mask.astype(int)) > 1

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

    dgy__[:] = ((g3__ + g2__) - (g0__ * cos_da0__ + g1__ * cos_da1__))
    # y-decomposed cosine difference between gs
    dgx__[:] = ((g1__ + g2__) - (g0__ * cos_da0__ + g3__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__[:] = ma.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = ma.minimum(g0__, g2__) * abs(cos_da0__)  # g match = min(g, _g) *cos(da)
    mg1__ = ma.minimum(g1__, g3__) * abs(cos_da1__)
    mg__[:]  = mg0__ + mg1__

    ig__[:] = g__ [:-1, :-1]  # remove last row and column to align with derived params
    idy__[:] = dy__[:-1, :-1]
    idx__[:] = dx__[:-1, :-1]  # -> idy, idx to compute cos for comp rg
    gg__.mask = mg__.mask = dgy__.mask = dgx__.mask = summed_mask

    '''
    next comp_rg will use g, dy, dx
    next comp_gg will use gg, dgy, dgx
    '''
    return new_dert__  # new_dert__ has been updated along with 'view' arrays: ig__, idy__, idx__, gg__, dgy__, dgx__, mg__

"""
Cross-comparison of pixels or gradients, in 2x2 or 3x3 kernels
"""

def comp_r_old(dert__, fig, root_fcr):

    i__ = dert__[0]  # i is ig if fig else pixel
    '''
    sparse aligned i__center and i__rim arrays:
    '''
    i__center =      i__[1:-1:2, 1:-1:2].copy()
    i__topleft =     i__[:-2:2, :-2:2].copy()
    i__top =         i__[:-2:2, 1:-1:2].copy()
    i__topright =    i__[:-2:2, 2::2].copy()
    i__right =       i__[1:-1:2, 2::2].copy()
    i__bottomright = i__[2::2, 2::2].copy()
    i__bottom =      i__[2::2, 1:-1:2].copy()
    i__bottomleft =  i__[2::2, :-2:2].copy()
    i__left =        i__[1:-1:2, :-2:2].copy()
    ''' 
    remove mask from kernels with only one masked dert 
    '''
    mask_i = mask_SUM([i__center.mask, i__topleft.mask, i__top.mask,
                       i__topright.mask, i__right.mask, i__bottomright.mask,
                       i__bottom.mask, i__bottomleft.mask, i__left.mask])

    i__center.mask = i__topleft.mask = i__top.mask = i__topright.mask = i__right.mask = i__bottomright.mask = \
    i__bottom.mask = i__bottomleft.mask = i__left.mask = mask_i

    idy__, idx__ = dert__[[1, 2]]

    if root_fcr:  # root fork is comp_r, accumulate derivatives:

        dy__, dx__, m__ = dert__[[4, 5, 6]]
        dy__ = dy__[1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        dx__ = dx__[1:-1:2, 1:-1:2].copy()
        m__  =  m__[1:-1:2, 1:-1:2].copy()
        dy__.mask = dx__.mask = m__.mask = mask_i

    else:   # root fork is comp_g or comp_pixel, initialize sparse derivatives:

        dy__ = ma.zeros((i__center.shape[0], i__center.shape[1]))  # row, column
        dx__ = ma.zeros((i__center.shape[0], i__center.shape[1]))
        m__ = ma.zeros((i__center.shape[0], i__center.shape[1]))

    if not fig:  # compare four diametrically opposed pairs of rim pixels:

        dt__ = np.stack((i__topleft - i__bottomright,
                         i__top - i__bottom,
                         i__topright - i__bottomleft,
                         i__right - i__left
                         ))
        dt__.mask = mask_i  # not needed?

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

        i__[ma.where(i__ == 0)] = 1  # if g is int
        a__ = [idy__, idx__] / i__  # sin, cos;  i = ig
        '''
        sparse aligned a__center and a__rim arrays:
        '''
        a__center      = a__[:, 1:-1:2, 1:-1:2].copy()
        a__topleft     = a__[:, :-2:2, :-2:2].copy()
        a__top         = a__[:, :-2:2, 1:-1: 2].copy()
        a__topright    = a__[:, :-2:2, 2::2].copy()
        a__right       = a__[:, 1:-1:2, 2::2].copy()
        a__bottomright = a__[:, 2::2, 2::2].copy()
        a__bottom      = a__[:, 2::2, 1:-1:2].copy()
        a__bottomleft  = a__[:, 2::2, :-2:2].copy()
        a__left        = a__[:, 1:-1:2, :-2:2].copy()

        ''' 
        mask kernels with more than one masked dert 
        '''
        mask_a = mask_SUM([a__center.mask, a__topleft.mask, a__top.mask,
                           a__topright.mask, a__right.mask, a__bottomright.mask,
                           a__bottom.mask, a__bottomleft.mask, a__left.mask])

        a__center.mask = a__topleft.mask = a__top.mask = a__topright.mask = a__right.mask = a__bottomright.mask = \
            a__bottom.mask = a__bottomleft.mask = a__left.mask = mask_a

        '''
        8-tuple of differences between central dert angle and rim dert angle:
        '''
        cos_da = ma.stack((
                  ((a__topleft[1]     * a__center[1]) + (a__center[0] * a__topleft[0])),
                  ((a__top[1]         * a__center[1]) + (a__center[0] * a__top[0])),
                  ((a__topright[1]    * a__center[1]) + (a__center[0] * a__topright[0])),
                  ((a__right[1]       * a__center[1]) + (a__center[0] * a__right[0])),
                  ((a__bottomright[1] * a__center[1]) + (a__center[0] * a__bottomright[0])),
                  ((a__bottom[1]      * a__center[1]) + (a__center[0] * a__bottom[0])),
                  ((a__bottomleft[1]  * a__center[1]) + (a__center[0] * a__bottomleft[0])),
                  ((a__left[1]        * a__center[1]) + (a__center[0] * a__left[0]))
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
            accumulate in prior-range dy, dx: 3x3 -> 5x5 -> 9x9 
            '''
        g__ = np.hypot(dy__, dx__)

    idy__ = idy__[1:-1:2, 1:-1:2].copy()  # i__center.shape, add .copy()?
    idx__ = idx__[1:-1:2, 1:-1:2].copy()  # i__center.shape
    idy__.mask = idx__.mask = i__center.mask  # align shifted masks
    '''
    next comp_r will use full dert       
    next comp_g will use g__, dy__, dx__
    '''
    return ma.stack((i__center, idy__, idx__, g__, dy__, dx__, m__))


def comp_g_old(dert__):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__

    g__, dy__, dx__ = dert__[[3, 4, 5]]  # g, dy, dx -> local i, idy, idx
    g__[ma.where(g__ == 0)] = 1  # replace 0 values with 1 to avoid error, not needed in high-g blobs?

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

    dgy__ = ((g3__ + g2__) - (g0__ * cos_da0__ + g1__ * cos_da1__))
    # y-decomposed cosine difference between gs
    dgx__ = ((g1__ + g2__) - (g0__ * cos_da0__ + g3__ * cos_da1__))
    # x-decomposed cosine difference between gs

    gg__ = np.hypot(dgy__, dgx__)  # gradient of gradient

    mg0__ = np.minimum(g0__, g2__) * cos_da0__  # g match = min(g, _g) *cos(da)
    mg1__ = np.minimum(g1__, g3__) * cos_da1__
    mg__  = mg0__ + mg1__

    g__ = g__ [:-1, :-1]  # remove last row and column to align with derived params
    dy__= dy__[:-1, :-1]
    dx__= dx__[:-1, :-1]  # -> idy, idx to compute cos for comp rg

    # no longer needed: g__.mask = dy__.mask = dx__.mask = gg__.mask?
    '''
    next comp_rg will use g, dy, dx     
    next comp_gg will use gg, dgy, dgx  
    '''
    return ma.stack((g__, dy__, dx__, gg__, dgy__, dgx__, mg__))


def shape_check(dert__):  # deprecated,
    # discussion: https://github.com/boris-kz/CogAlg/commit/b2da96480423704bd419766148d8fd2b89650c75#commitcomment-38341233
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
Unpack in code:
        a__center.mask = a__topleft.mask = a__top.mask = a__topright.mask = a__right.mask = a__bottomright.mask = \
        a__bottom.mask = a__bottomleft.mask = a__left.mask = \
        functools.reduce(lambda x1, x2:
                         x1.astype('int') + x2.astype('int'),
                         [a__center, a__topleft, a__top,
                         a__topright, a__right, a__bottomright,
                         a__bottom, a__bottomleft, a__left]
                         ) > 1
'''