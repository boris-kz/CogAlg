from collections import deque
import numpy.ma as ma
import numpy as np

'''
comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over the whole frame.
Coefs scale down pixel dy and dx contribution to kernel g in proportion to the ratio of that pixel distance and angle 
to ortho-pixel distance and angle. This is a proximity-ordered search, comparing ortho-pixels first, thus their coef = 1.  

This is a more precise equivalent of Sobel operator, but works in reverse, the latter sets diagonal pixel coef = 1 and scales 
contribution of other pixels up, in proportion to the same ratios (relative contribution of each rim pixel to g in Sobel is 
similar but lower resolution). This forms integer coefs, vs our fractional coefs, which makes computation a lot faster. 
We may switch to integer coefs for speed, and are open to using Scharr operator in the future.

kwidth = 3: input-centered, low resolution kernel: frame | blob shrink by 2 pixels per row,
kwidth = 2: co-centered, grid shift, 1-pixel row shrink, no deriv overlap, 1/4 chance of boundary pixel in kernel?
kwidth = 2: quadrant g = ((dx + dy) * .705 + d_diag) / 2, no i res decrement, ders co-location, + orthogonal quadrant for full rep?
'''

def comp_pixel(image):  # current version of 2x2 pixel cross-correlation within image

    # following four slices provide inputs to a sliding 2x2 kernel:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    botleft__ = image[1:, :-1]
    botright__ = image[1:, 1:]

    dy__ = ((botleft__ + botright__) - (topleft__ + topright__))  # same as diagonal from left
    dx__ = ((topright__ + botright__) - (topleft__ + botleft__))  # same as diagonal from right
    g__ = np.hypot(dy__, dx__)  # gradient per kernel

    return ma.stack((topleft__, g__, dy__, dx__))


def comp_pixel_m(image):  # current version of 2x2 pixel cross-correlation within image

    # following four slices provide inputs to a sliding 2x2 kernel:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    botleft__ = image[1:, :-1]
    botright__ = image[1:, 1:]

    dy__ = ((botleft__ + botright__) - (topleft__ + topright__))  # same as diagonal from left
    dx__ = ((topright__ + botright__) - (topleft__ + botleft__))  # same as diagonal from right
    g__ = np.hypot(dy__, dx__)  # gradient per kernel

    # inverse match = SAD: measure of variation within kernel
    m__ = ( abs(topleft__ - botright__) + abs(topright__ - botleft__))

    return ma.stack((topleft__, g__, dy__, dx__, m__))


def comp_pixel_old(image):  # 2x2 pixel cross-correlation within image

    dy__ = image[1:] - image[:-1]        # orthogonal vertical com
    dx__ = image[:, 1:] - image[:, :-1]  # orthogonal horizontal comp

    p__ = image[:-1, :-1]  #  top-left pixel
    mean_dy__ = (dy__[:, 1:] + dy__[:, :-1]) * 0.5  # mean dy per kernel
    mean_dx__ = (dx__[1:, :] + dx__[:-1, :]) * 0.5  # mean dx per kernel
    g__ = ma.hypot(mean_dy__, mean_dx__)  # central gradient of four rim pixels
    dert__ = ma.stack((p__, g__, mean_dy__, mean_dx__))

    return dert__


def comp_pixel_skip(image):  # 2x2 pixel cross-comp without kernel overlap

    p1__ = image[:-1, :-1, 2]
    p2__ = image[:-1, 1:, 2]
    p3__ = image[1:, :-1, 2]
    p4__ = image[1:, 1:, 2]

    dy__ = ((p3__ + p4__) - (p1__ + p2__)) * 0.5
    dx__ = ((p2__ + p4__) - (p1__ + p3__)) * 0.5

    g__ = np.hypot(dy__, dx__)
    return (p1__, p2__, p3__, p4__), g__, dy__, dx__


def comp_pixel_ternary(image):  # 3x3 and 2x2 pixel cross-correlation within image
    # orthogonal comp
    orthdy__ = image[1:] - image[:-1]       # vertical
    orthdx__ = image[:, 1:] - image[:, :-1] # horizontal

    # compute gdert__
    p__ = (image[:-2, :-2] + image[:-2, 1:-1] + image[1:-1, :-2] + image[1:-1, 1:-1]) * 0.25
    dy__ = (orthdy__[:-1, 1:-1] + orthdy__[:-1, :-2]) * 0.5
    dx__ = (orthdx__[1:-1, :-1] + orthdx__[:-2, :-1]) * 0.5
    g__ = ma.hypot(dy__, dx__)
    gdert__ = ma.stack((p__, g__, dy__, dx__))

    # diagonal comp
    diag1__ = image[2:, 2:] - image[:-2, :-2]
    diag2__ = image[2:, :-2] - image[:-2, 2:]

    # compute rdert__
    p3__ = image[1:-1, 1:-1]
    dy3__ = (orthdy__[1:, 1:-1] + orthdy__[:-1, 1:-1]) * 0.25 +\
            (diag1__ + diag2__) * 0.125
    dx3__ = (orthdx__[1:-1, 1:] + orthdx__[1:-1, :-1]) * 0.25 + \
            (diag1__ - diag2__) * 0.125
    g3__ = ma.hypot(dy3__, dx3__)
    rdert__ = ma.stack((p3__, g3__, dy3__, dx3__))

    # gdert__ = comp_2x2(image)  # cross-compare four adjacent pixels diagonally
    # rdert__ = comp_3x3(image)  # compare each pixel to 8 rim pixels

    return gdert__, rdert__


def comp_pixel_diag(image):  # 3x3 and 2x2 pixel cross-correlation within image
    gdert__ = comp_2x2(image)  # cross-compare four adjacent pixels diagonally
    rdert__ = comp_3x3(image)  # compare each pixel to 8 rim pixels

    return gdert__, rdert__

def comp_2x2(image):
    """Deprecated."""
    dy__ = (image[1:-1, 1:-1] + image[1:-1, :-2]
            - image[:-2, 1:-1] - image[:-2, :-2]) * 0.5
    dx__ = (image[1:-1, 1:-1] + image[:-2, 1:-1]
            - image[1:-1, :-2] - image[:-2, :-2]) * 0.5
    # sum pixel values and reconstruct central pixel as their average:
    p__ = (image[:-2, :-2] + image[:-2, 1:-1]
           + image[1:-1, :-2] + image[1:-1, 1:-1]) * 0.25
    g__ = np.hypot(dy__, dx__)  # compute gradients per kernel, converted to 0-255 range
    return ma.stack((p__, g__, dy__, dx__))


def comp_3x3(image):
    """Deprecated."""
    d___ = np.array(  # subtract centered image from translated image:
        [image[ts2] - image[ts1] for ts1, ts2 in TRANSLATING_SLICES_PAIRS_3x3]
    ).swapaxes(0, 2).swapaxes(0, 1)
    # 3rd dimension: sequence of differences between pairs of
    # diametrically opposed pixels corresponding to:
    #          |--(clockwise)--+              |--(clockwise)--+
    # YCOEF: -0.5    -1  -0.5  ¦  XCOEF:    -0.5   0    0.5   ¦
    #          0           0   ¦             -1          1    ¦
    #         0.5     1   0.5  ¦            -0.5   0    0.5   ¦
    #                    <<----+                        <<----+
    # Decompose differences into dy and dx, same as Gy and Gx in conventional edge detection operators:
    dy__ = (d___ * YCOEF).sum(axis=2)
    dx__ = (d___ * XCOEF).sum(axis=2)
    p__ = image[1:-1, 1:-1]
    g__ = np.hypot(dy__, dx__)  # compute gradients per kernel, converted to 0-255 range

    return ma.stack((p__, g__, dy__, dx__))


def comp_3x3_loop(image):
    buff_ = deque() # buffer for derts
    Y, X = image.shape

    for p_ in image: # loop through rows
        for p in p_: # loop through each pixel
            dx = dy = 0  # uni-lateral differences
            # loop through buffers:
            for k, xcoeff, ycoeff in zip([  0 ,   X-1 ,   X ,  X+1 ], # indices of _dert
                                         [0.25, -0.125,   0 , 0.125], # x axis coefficient
                                         [  0 ,  0.125, 0.25, 0.125]): # y axis coefficient
                try:
                    _p, _dy, _dx = buff_[k]  # unpack buff_[k]
                    d = p - _p  # compute difference
                    dx_buff = d * xcoeff  # decompose difference
                    dy_buff = d * ycoeff
                    dx += dx_buff  # accumulate fuzzy difference over the kernel
                    _dx += dx_buff
                    dy += dy_buff
                    _dy += dy_buff

                    buff_[k] = _p, _dy, _dx  # repack buff_[k]

                except TypeError: # buff_[k] is None
                    pass
                except IndexError: # k >= len(buff_)
                    break

            buff_.appendleft((p, dy, dx))  # initialize dert with uni-lateral differences
        buff_.appendleft(None)  # add empty dert at the end of each row

    # reshape data and compute g (temporary, to perform tests)
    p__, dy__, dx__ = np.array([buff for buff in reversed(buff_)
                                if buff is not None])\
        .reshape(Y, X, 3).swapaxes(1, 2).swapaxes(0, 1)[:, 1:-1, 1:-1]

    g__ = ma.hypot(dy__, dx__)

    return ma.stack((p__, g__, dy__, dx__))