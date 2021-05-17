"""
Cross-comparison of pixels 3x3 kernels or gradient angles in 2x2 kernels
"""

import numpy as np
import functools

''' 
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
'''

def comp_r(dert__, ave, root_fia, mask__=None):
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
    rng = 4: 9x9 kernel,
    ...
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, and we use Sobel coeffs,
    see:
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

    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
        m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs:

    dy__ += ((i__topleft - i__bottomright) * -1 +
             (i__top - i__bottom) * -2 +
             (i__topright - i__bottomleft) * -1 +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 +
             (i__right - i__left) * 2)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    ave SAD = ave g * 1.2:
    '''
    m__ += int(ave * 1.2) - ( abs(i__center - i__topleft)
                            + abs(i__center - i__top) * 2
                            + abs(i__center - i__topright)
                            + abs(i__center - i__right) * 2
                            + abs(i__center - i__bottomright)
                            + abs(i__center - i__bottom) * 2
                            + abs(i__center - i__bottomleft)
                            + abs(i__center - i__left) * 2
                            )

    return (i__center, dy__, dx__, g__, m__), majority_mask__


def comp_a(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels
    '''
    More concise but also more opaque version
    https://github.com/khanh93vn/CogAlg/commit/1f3499c4545742486b89e878240d5c291b81f0ac
    '''
    if mask__ is not None:
        majority_mask__ = (mask__[:-1, :-1].astype(int) +
                           mask__[:-1, 1:].astype(int) +
                           mask__[1:, 1:].astype(int) +
                           mask__[1:, :-1].astype(int)
                           ) > 1
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    az__ = dx__ + 1j * dy__  # take the complex number (z), phase angle is now atan2(dy, dx)

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        az__ /= np.absolute(az__)  # normalized, cosine = a__.real, sine = a__.imag

    # a__ shifted in 2x2 kernel, rotate 45 degrees counter-clockwise to cancel clockwise rotation in frame_blobs:
    az__left = az__[:-1, :-1]  # was topleft
    az__top = az__[:-1, 1:]  # was topright
    az__right = az__[1:, 1:]  # was botright
    az__bottom = az__[1:, :-1]  # was botleft

    dazx__ = angle_diff(az__right, az__left)
    dazy__ = angle_diff(az__bottom, az__top)
    # (a__ is rotated 45 degrees counter-clockwise)

    dax__ = np.angle(dazx__)  # phase angle of the complex number, same as np.atan2(dazx__.imag, dazx__.real)
    day__ = np.angle(dazy__)

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        ma__ = 1 - (np.abs(dax__) + np.abs(day__)) / 2 * np.pi   # the result is in range in 0-1
        ma__
    '''
    ma = 1 - (np.abs(dax__) + np.abs(day__)) / (2*π):  in range [0, 1]. 
    ma @ 22.5 deg = 0.875: 1 - (π/8 + π/8)/(2*π)
    ma @ 45 deg   = 0.75:  1 - (π/4 + π/4)/(2*π)
    
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(day__, dax__) - 0.2777  # same as old formula, atan2 and angle are equivalent
    '''
    ga value is deviation from ave: 0.2777 @ 22.5 deg, 0.5554 @ 45 degrees = π/4 radians, sqrt(0.5)*π/4 
    
    interruption | wave is sign-agnostic: expected reversion, same for d sign?
    extended-kernel gradient from decomposed diffs: np.hypot(dydy, dxdy) + np.hypot(dydx, dxdx)?
    '''
    # if root fork is frame_blobs, recompute orthogonal dy and dx

    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):
        i__topleft = i__[:-1, :-1]
        i__topright = i__[:-1, 1:]
        i__botright = i__[1:, 1:]
        i__botleft = i__[1:, :-1]
        dy__ = (i__botleft + i__botright) - (i__topleft + i__topright)  # decomposition of two diagonal differences
        dx__ = (i__topright + i__botright) - (i__topleft + i__botleft)  # decomposition of two diagonal differences
    else:
        dy__ = dy__[:-1, :-1]  # passed on as idy, not rotated
        dx__ = dx__[:-1, :-1]  # passed on as idx, not rotated

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]

    return (i__, dy__, dx__, g__, m__, dazy__, dazx__, ga__, ma__), majority_mask__  # dazx__, dazy__ may not be needed


def angle_diff(az2, az1):  # compare phase angle of az1 to that of az2
    '''
    az1 = cos_1 + j*sin_1
    az2 = cos_2 + j*sin_2
    (sin_1, cos_1, sin_2, cos_2 below in angle_diff2)
    Assuming that the formula in angle_diff is correct, the result is:
    daz = cos_da + j*sin_da

    Substitute cos_da, sin_da (from angle_diff below):

    daz = (cos_1*cos_2 + sin_1*sin_2) + j*(cos_1*sin_2 - sin_1*cos_2)
        = (cos_1 + j*sin_1)*(cos_2 - j*sin_2)

    Substitute (1) and (2) into the above eq:
    daz = az1 * complex_conjugate_of_(az2)

    az1 = a + bj; az2 = c + dj
    daz = (a + bj)(c - dj)
        = (ac + bd) + (ad - bc)j
        (same as old formula, in angle_diff2() below)
     '''

    return az1*az2.conj()  # imags and reals of the result are sines and cosines of difference between angles

'''
old version:
'''
def comp_a_simple(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels

    # angles can't be summed: https://rosettacode.org/wiki/Averages/Mean_angle

    if mask__ is not None:
        majority_mask__ = (mask__[:-1, :-1].astype(int) +
                           mask__[:-1, 1:].astype(int) +
                           mask__[1:, 1:].astype(int) +
                           mask__[1:, :-1].astype(int)
                           ) > 1
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    a__ = [dy__, dx__] / (g__ + ave + 0.001)  # + ave to restore abs g, + .001 to avoid / 0
    # g and m are rotation invariant, but da is more accurate with rot_a__:

    # a__ shifted in 2x2 kernel, rotate 45 degrees counter-clockwise to cancel clockwise rotation in frame_blobs:
    a__left   = a__[:, :-1, :-1]  # was topleft
    a__top    = a__[:, :-1, 1:]   # was topright
    a__right  = a__[:, 1:, 1:]    # was botright
    a__bottom = a__[:, 1:, :-1]   # was botleft

    sin_da0__, cos_da0__ = angle_diff(a__right, a__left)
    sin_da1__, cos_da1__ = angle_diff(a__bottom, a__top)

    ''' 
    match of angle = inverse deviation rate of SAD of angles from ave ma of all possible angles.
    we use ave 2: (2 + 2) / 2, 2 is average not-deviation ma, when da is 90 degree (because da varies from 0-180 degree). 
    That's just a rough guess, as all filter initializations, actual average will be lower because adjacent angles don't vary as much, 
    there is general correlation between proximity and similarity.
    Normally, we compute match as inverse deviation: ave - value. Here match is defined directly(?), so it's value - ave
    '''
    ma__ = (cos_da0__ + 1.001) + (cos_da1__ + 1.001) - 2  # +1 to convert to all positives, +.001 to avoid / 0, ave ma = 2

    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines
    day__ = [-sin_da0__ - sin_da1__, cos_da0__ + cos_da1__]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    dax__ = [-sin_da0__ + sin_da1__, cos_da0__ + cos_da1__]
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot( np.arctan2(*day__), np.arctan2(*dax__) )
    '''
    ga value is a deviation; interruption | wave is sign-agnostic: expected reversion, same for d sign?
    extended-kernel gradient from decomposed diffs: np.hypot(dydy, dxdy) + np.hypot(dydx, dxdx)?
    '''
    # if root fork is frame_blobs, recompute orthogonal dy and dx
    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):
        i__topleft = i__[:-1, :-1]
        i__topright = i__[:-1, 1:]
        i__botright = i__[1:, 1:]
        i__botleft = i__[1:, :-1]
        dy__ = (i__botleft + i__botright) - (i__topleft + i__topright)  # decomposition of two diagonal differences
        dx__ = (i__topright + i__botright) - (i__topleft + i__botleft)  # decomposition of two diagonal differences
    else:
        dy__ = dy__[:-1, :-1]  # passed on as idy, not rotated
        dx__ = dx__[:-1, :-1]  # passed on as idx, not rotated

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]


    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask__


def angle_diff_simple(a2, a1):  # compare angle_1 to angle_2

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]


