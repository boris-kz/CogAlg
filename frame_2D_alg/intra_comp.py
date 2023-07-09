"""
Cross-comparison of pixels or gradient angles in 2x2 kernels
"""

import numpy as np
from utils import kernel_slice_3x3 as ks
# no ave_ga = .78, ave_ma = 2  # at 22.5 degrees
# https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png

def comp_r(dert__, rng, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient in shorter-range cross-comp: input intensity didn't vary much.
    Such input is predictable enough for selective sampling: skipping current rim in following comparison kernels.
    Skipping forms increasingly sparse dert__ for next-range cross-comp,
    hence kernel width increases as 2^rng: 1: 2x2 kernel, 2: 4x4 kernel, 3: 8x8 kernel
    There is also skipping within greater-rng rims, so configuration of compared derts is always 2x2
    '''

    i__ = dert__[0]  # pixel intensity, should be separate from i__sum
    # sparse aligned rim arrays:
    i__topleft = i__[:-1:2, :-1:2]  # also assignment to new_dert__[0]
    i__topright = i__[:-1:2, 1::2]
    i__bottomleft = i__[1::2, :-1:2]
    i__bottomright = i__[1::2, 1::2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[:-1:2, :-1:2].astype(int)
                          + mask__[:-1:2, 1::2].astype(int)
                          + mask__[1::2, 1::2].astype(int)
                          + mask__[1::2, :-1:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function

    d_upleft__ = dert__[1][:-1:2, :-1:2].copy()  # sparse step=2 sampling
    d_upright__= dert__[2][:-1:2, :-1:2].copy()
    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 8x8, *4 for 16x16
    # combined distance and extrapolation coeffs, or separate distance coef: ave * (rave / dist), rave = ave abs d / ave i?
    # compare pixels diagonally:
    d_upright__+= (i__bottomleft - i__topright) * rngSkip
    d_upleft__ += (i__bottomright - i__topleft) * rngSkip

    g__ = np.hypot(d_upright__, d_upleft__)  # match = inverse of abs gradient (variation), recomputed at each comp_r
    ri__ = i__topleft + i__topright + i__bottomleft + i__bottomright

    return (i__topleft, d_upleft__, d_upright__, g__, ri__), majority_mask__


def comp_a(dert__, mask__=None):  # cross-comp of gradient angle in 3x3 kernels

    if mask__ is not None:
        majority_mask__ = np.sum(
            (
                mask__[ks.tl], mask__[ks.tc], mask__[ks.tr],
                mask__[ks.ml], mask__[ks.mc], mask__[ks.mr],
                mask__[ks.bl], mask__[ks.bc], mask__[ks.br],
            ),
            axis=0) > 2.25  # 1/4 of maximum values?
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, ri__ = dert__[:5]  # day__,dax__,ma__ are recomputed

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        uv__ = dert__[1:3] / g__
        uv__[np.where(np.isnan(uv__))] = 0  # set nan to 0, to avoid error later

    # uv__ comparison in 3x3 kernels:
    lcol = angle_diff(uv__[ks.br], uv__[ks.tr])  # left col
    ccol = angle_diff(uv__[ks.bc], uv__[ks.tc])  # central col
    rcol = angle_diff(uv__[ks.bl], uv__[ks.tl])  # right col
    trow = angle_diff(uv__[ks.tr], uv__[ks.tl])  # top row
    mrow = angle_diff(uv__[ks.mr], uv__[ks.ml])  # middle row
    brow = angle_diff(uv__[ks.br], uv__[ks.bl])  # bottom row

    # compute mean vectors
    mday__ = 0.25*lcol + 0.5*ccol + 0.25*rcol
    mdax__ = 0.25*trow + 0.5*mrow + 0.25*brow

    # normalize mean vectors into unit vectors
    uday__, vday__ = mday__ / np.hypot(*mday__)
    udax__, vdax__ = mdax__ / np.hypot(*mdax__)

    # v component of mean unit vector represents similarity of angles
    # between compared vectors, goes from -1 (opposite) to 1 (same)
    ga__ = np.hypot(1-vday__, 1-vday__)     # +1 for all positives
    # or ga__ = np.hypot( np.arctan2(*day__), np.arctan2(*dax__)?

    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    in conventional notation: G = (Ix, Iy), A = (Ix, Iy) / hypot(G), DA = (dAdx, dAdy), abs_GA = hypot(DA)?
    '''
    i__ = i__[ks.mc]
    dy__ = dy__[ks.mc]
    dx__ = dx__[ks.mc]
    g__ = g__[ks.mc]
    ri__ = ri__[ks.mc]

    return (i__, g__, ga__, ri__, dy__, dx__, uday__, vday__, udax__, vdax__), majority_mask__

def angle_diff(uv2, uv1):  # compare angles of uv1 to uv2 (uv1 to uv2)

    u2, v2 = uv2[:]
    u1, v1 = uv1[:]

    # sine and cosine of difference between angles of uv1 and uv2:
    u3 = (v1 * u2) - (u1 * v2)
    v3 = (v1 * v2) + (u1 * u2)

    return np.stack((u3, v3))

'''
alternative versions below:
'''
def comp_r_odd(dert__, ave, rng, root_fia, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel,
    ...
    Sobel coefficients to decompose ds into dy and dx:
    YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
    XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
        |--(clockwise)--+  |--(clockwise)--+
        YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
                0       0  ¦          -2       2  ¦
                1   2   1  ¦          -1   0   1  ¦
    Scharr coefs:
    YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
    XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, using Sobel coeffs, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_d.drawio
    '''

    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    rotate in first call only: same orientation as from frame_blobs?
    '''
    i__center = i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
    i__topleft = i__[:-2:2, :-2:2]
    i__top = i__[:-2:2, 1:-1:2]
    i__topright = i__[:-2:2, 2::2]
    i__right = i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom = i__[2::2, 1:-1:2]
    i__bottomleft = i__[2::2, :-2:2]
    i__left = i__[1:-1:2, :-2:2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[1:-1:2, 1:-1:2].astype(int)
                          + mask__[:-2:2, :-2:2].astype(int)
                          + mask__[:-2:2, 1:-1: 2].astype(int)
                          + mask__[:-2:2, 2::2].astype(int)
                          + mask__[1:-1:2, 2::2].astype(int)
                          + mask__[2::2, 2::2].astype(int)
                          + mask__[2::2, 1:-1:2].astype(int)
                          + mask__[2::2, :-2:2].astype(int)
                          + mask__[1:-1:2, :-2:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function
    '''
    can't happen:
    if root_fia:  # initialize derivatives:  
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)
    else: 
    '''
     # root fork is comp_r, accumulate derivatives:
    dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
    dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
    m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs * rim skip ratio:

    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 9x9, *4 for 17x17

    dy__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * -2  * rngSkip +
             (i__topright - i__bottomleft) * -1 * rngSkip +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 * rngSkip+
             (i__right - i__left) * 2 * rngSkip)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    '''
    m__ += ( abs(i__center - i__topleft) * 1 * rngSkip
           + abs(i__center - i__top) * 2 * rngSkip
           + abs(i__center - i__topright) * 1 * rngSkip
           + abs(i__center - i__right) * 2 * rngSkip
           + abs(i__center - i__bottomright) * 1 * rngSkip
           + abs(i__center - i__bottom) * 2 * rngSkip
           + abs(i__center - i__bottomleft) * 1 * rngSkip
           + abs(i__center - i__left) * 2 * rngSkip
           )

    return (i__center, dy__, dx__, g__, m__), majority_mask__