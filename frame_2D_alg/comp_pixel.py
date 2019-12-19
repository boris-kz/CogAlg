import numpy.ma as ma
import numpy as np
'''
comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over the whole frame.
Coefs scale down pixel dy and dx contribution to kernel g in proportion to the ratio of that pixel distance and angle 
to ortho-pixel distance and angle. This is a proximity-ordered search, comparing ortho-pixels first, thus their coef = 1.  

This is equivalent to Sobel operator, but it and other conventional kernels do the opposite: set diagonal pixel coef = 1, 
and scale contribution of other pixels up, in proportion to the same ratios (relative contribution of each rim pixel to g 
is the same in Sobel). This forms integer coefs, vs our fractional coefs, which makes computation a lot faster. 
We will probably switch to integer coefs for speed, and are open to using Scharr operator in the future.
'''
# Constants: MAX_G = 256  # 721.2489168102785 without normalization.
# Adjustable parameters:

kwidth = 3  # input-centered, low resolution kernel: frame | blob shrink by 2 pixels per row,
# kwidth = 2  # co-centered, grid shift, 1-pixel row shrink, no deriv overlap, 1/4 chance of boundary pixel in kernel?
# kwidth = 2 quadrant: g = ((dx + dy) * .705 + d_diag) / 2, signed-> gPs? no i res-, ders co-location, + orthogonal quadrant for full rep?

def comp_pixel(image):  # 3x3 or 2x2 pixel cross-correlation within image

    if kwidth == 2:  # cross-compare four adjacent pixels diagonally:

        dy__ = (image[1:, 1:] - image[:-1, 1:]) + (image[1:, :-1] - image[:-1, :-1]) * 0.5
        dx__ = (image[1:, 1:] - image[1:, :-1]) + (image[:-1, 1:] - image[:-1, :-1]) * 0.5

        # or no coef: distance 1.41 * angle .705 -> 1? and conversion only for extended kernel, if centered?
        # sum pixel values and reconstruct central pixel as their average:

        p__ = (image[:-1, :-1] + image[:-1, 1:] + image[1:, :-1] + image[1:, 1:]) * 0.25

    else:  # kwidth == 3, compare central pixel to 8 rim pixels, current default option

        ycoef = np.array([-0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0])
        xcoef = np.array([-0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1])

        d___ = np.array(list(  # subtract centered image from translated image:
            map(lambda trans_slices: image[trans_slices] - image[1:-1, 1:-1],
                [
                    (slice(None, -2), slice(None, -2)),
                    (slice(None, -2), slice(1, -1)),
                    (slice(None, -2), slice(2, None)),
                    (slice(1, -1), slice(2, None)),
                    (slice(2, None), slice(2, None)),
                    (slice(2, None), slice(1, -1)),
                    (slice(2, None), slice(None, -2)),
                    (slice(1, -1), slice(None, -2)),
                ]
            )
        )).swapaxes(0, 2).swapaxes(0, 1)

        # Decompose differences into dy and dx, same as Gy and Gx in conventional edge detection operators:

        dy__ = (d___ * ycoef).sum(axis=2)
        dx__ = (d___ * xcoef).sum(axis=2)

        p__ = image[1:-1, 1:-1]

    g__ = np.hypot(dy__, dx__) * 0.354801226089485  # compute gradients per kernel, converted to 0-255 range

    return ma.around(ma.stack((p__, g__, dy__, dx__), axis=0))

