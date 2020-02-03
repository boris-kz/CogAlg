import numpy.ma as ma
import numpy as np
'''
comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over the whole frame.
Coefs scale down pixel dy and dx contribution to kernel g in proportion to the ratio of that pixel distance and angle 
to ortho-pixel distance and angle. This is a proximity-ordered search, comparing ortho-pixels first, thus their coef = 1.  

This is a more precise equivalent to Sobel operator, but works in reverse, the latter sets diagonal pixel coef = 1, and scales 
contribution of other pixels up, in proportion to the same ratios (relative contribution of each rim pixel to g in Sobel is 
similar but lower resolution). This forms integer coefs, vs our fractional coefs, which makes computation a lot faster. 
We will probably switch to integer coefs for speed, and are open to using Scharr operator in the future.

kwidth = 3: input-centered, low resolution kernel: frame | blob shrink by 2 pixels per row,
kwidth = 2: co-centered, grid shift, 1-pixel row shrink, no deriv overlap, 1/4 chance of boundary pixel in kernel?
kwidth = 2: quadrant g = ((dx + dy) * .705 + d_diag) / 2, no i res decrement, ders co-location, + orthogonal quadrant for full rep?
'''
# Constants:
MAX_G = 255  # 721.2489168102785 without normalization.

G_NORMALIZER_3x3 = 255.9 / (255 * 2**0.5 * 2)
G_NORMALIZER_2x2 = 255.9 / (255 * 2**0.5)

# Coefficients and translating slices sequence for 3x3 window comparison
YCOEF = np.array([-0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0])
XCOEF = np.array([-0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1])

TRANSLATING_SLICES_SEQUENCE = [
    (slice(None, -2), slice(None, -2)),
    (slice(None, -2), slice(1, -1)),
    (slice(None, -2), slice(2, None)),
    (slice(1, -1), slice(2, None)),
    (slice(2, None), slice(2, None)),
    (slice(2, None), slice(1, -1)),
    (slice(2, None), slice(None, -2)),
    (slice(1, -1), slice(None, -2)),
]

def comp_pixel(image):  # 3x3 and 2x2 pixel cross-correlation within image

    gdert__ = comp_2x2(image)  # cross-compare four adjacent pixels diagonally
    rdert__ = comp_3x3(image)  # compare each pixel to 8 rim pixels

    return gdert__, rdert__


def comp_2x2(image):
    dy__ = (image[1:, 1:] + image[1:, :-1] - image[:-1, 1:] - image[:-1, :-1]) * 0.5
    dx__ = (image[1:, 1:] + image[:-1, 1:] - image[1:, :-1] - image[:-1, :-1]) * 0.5
    # sum pixel values and reconstruct central pixel as their average:
    p__ = (image[:-1, :-1] + image[:-1, 1:] + image[1:, :-1] + image[1:, 1:]) * 0.25
    g__ = np.hypot(dy__, dx__) * G_NORMALIZER_2x2  # compute gradients per kernel, converted to 0-255 range
    return ma.around(ma.stack((p__, g__, dy__, dx__), axis=0))


def comp_3x3(image):
    d___ = np.array(  # subtract centered image from translated image:
        [image[trans_slices] - image[1:-1, 1:-1] for trans_slices in TRANSLATING_SLICES_SEQUENCE]
    ).swapaxes(0, 2).swapaxes(0, 1)  # 3rd dimension: sequence of differences corresponding to:
    #          |--(clockwise)--+              |--(clockwise)--+
    # YCOEF: -0.5    -1  -0.5  ¦  XCOEF:    -0.5   0    0.5   ¦
    #          0           0   ¦             -1          1    ¦
    #         0.5     1   0.5  ¦            -0.5   0    0.5   ¦
    #                    <<----+                        <<----+
    # Decompose differences into dy and dx, same as Gy and Gx in conventional edge detection operators:
    dy__ = (d___ * YCOEF).sum(axis=2)
    dx__ = (d___ * XCOEF).sum(axis=2)
    p__ = image[1:-1, 1:-1]
    g__ = np.hypot(dy__, dx__) * G_NORMALIZER_3x3  # compute gradients per kernel, converted to 0-255 range

    return ma.around(ma.stack((p__, g__, dy__, dx__), axis=0))