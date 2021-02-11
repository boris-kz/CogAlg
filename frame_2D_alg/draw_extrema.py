"""
Visualize output of first 3 layers of intra_comp forks, for testing
"""

from alternative_versions.comp_pixel_versions import comp_pixel_m
from intra_comp import *

import cv2
import argparse
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/toucan.jpg"
# Outputs:
OUTPUT_PATH = "./images/intra_comp0/"

# -----------------------------------------------------------------------------
# Functions


# rng = 2
def comp_r_rng2(dert__, ave, root_fia, mask__=None):
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
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    '''
    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    rotate in first call only: same orientation as from frame_blobs?
    '''


    i__center      = i__[2:-2:3, 2:-2:3]  # also assignment to new_dert__[0]
    i__topleft     = i__[:-4:3, :-4:3]
    i__top         = i__[:-4:3, 2:-2:3]
    i__topright    = i__[:-4:3, 4::3]
    i__right       = i__[2:-2:3, 4::3]
    i__bottomright = i__[4::3, 4::3]
    i__bottom      = i__[4::3, 2:-2:3]
    i__bottomleft  = i__[4::3, :-4:3]
    i__left        = i__[2:-2:3, :-4:3]

    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[2:-2:3, 2:-2:3].astype(int)
                          + mask__[:-4:3, :-4:3].astype(int)
                          + mask__[:-4:3, 2:-2:3].astype(int)
                          + mask__[:-4:3, 4::3].astype(int)
                          + mask__[2:-2:3, 4::3].astype(int)
                          + mask__[4::3, 4::3].astype(int)
                          + mask__[4::3, 2:-2:3].astype(int)
                          + mask__[4::3, :-4:3].astype(int)
                          + mask__[2:-2:3, :-4:3].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function

    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        dy__ = dert__[1][2:-2:3, 2:-2:3].copy()  # sparse to align with i__center
        dx__ = dert__[2][2:-2:3, 2:-2:3].copy()
        m__ = dert__[4][2:-2:3, 2:-2:3].copy()

    # compare four diametrically opposed pairs of rim pixels:

    d_tl_br = i__topleft - i__bottomright
    d_t_b = i__top - i__bottom
    d_tr_bl = i__topright - i__bottomleft
    d_r_l = i__right - i__left

    dy__ += (d_tl_br * YCOEFs[0] +
             d_t_b * YCOEFs[1] +
             d_tr_bl * YCOEFs[2] +
             d_r_l * YCOEFs[3])

    dx__ += (d_tl_br * XCOEFs[0] +
             d_t_b * XCOEFs[1] +
             d_tr_bl * XCOEFs[2] +
             d_r_l * XCOEFs[3])

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    ave SAD = ave g * 1.41:
    '''

    m__abs = m__.copy()
    m__ += int(ave * 1.41) - ( abs(i__center - i__topleft)
                             + abs(i__center - i__top)
                             + abs(i__center - i__topright)
                             + abs(i__center - i__right)
                             + abs(i__center - i__bottomright)
                             + abs(i__center - i__bottom)
                             + abs(i__center - i__bottomleft)
                             + abs(i__center - i__left)
                             )

    g__abs = np.hypot(dy__, dx__)
    m__abs += ( abs(i__center - i__topleft)
              + abs(i__center - i__top)
              + abs(i__center - i__topright)
              + abs(i__center - i__right)
              + abs(i__center - i__bottomright)
              + abs(i__center - i__bottom)
              + abs(i__center - i__bottomleft)
              + abs(i__center - i__left)
              )

    e__ = m__abs - 1.41*g__abs


    return (i__center, dy__, dx__, g__, m__, e__), majority_mask__


def draw_g(img_out, g_):
    endy = min(img_out.shape[0], g_.shape[0])
    endx = min(img_out.shape[1], g_.shape[1])
    img_out[:endy, :endx] = (g_[:endy, :endx] * 255) / g_.max()  # scale to max=255, less than max / 255 is 0

    return img_out

def draw_gr(img_out, g_):

    img_out[:] = cv2.resize((g_[:] * 255) / g_.max(),  # normalize g to uint
                            (img_out.shape[1], img_out.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    return img_out

def imread(filename, raise_if_not_read=True):
    "Read an image in grayscale, return array."
    try:
        return cv2.imread(filename, 0).astype(int)
    except AttributeError:
        if raise_if_not_read:
            raise SystemError('image is not read')
        else:
            print('Warning: image is not read')
            return None

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default=IMAGE_PATH)
    argument_parser.add_argument('-o', '--output', help='path to output folder', default=OUTPUT_PATH)
    arguments = argument_parser.parse_args()

    print('Reading image...')
    image = imread(arguments.image)

    dert_ = comp_pixel_m(image)

    ave = 50

    print('Processing first layer comps...')
    # comp_p ->
    gr_dert_, _ = comp_r_rng2(dert_, ave, root_fia = 0)


    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]), 'uint8')  # initialize image y, x

    # 0th layer
    g_ = draw_g(ini_.copy(), dert_[3])
    m_ = draw_g(ini_.copy(), dert_[6])
    # 1st layer
    gr_ = draw_gr(ini_.copy(), gr_dert_[3])
    mr_ = draw_gr(ini_.copy(), gr_dert_[4])
    e_ = draw_gr(ini_.copy(),  gr_dert_[5])


    # save to disk
    cv2.imwrite(arguments.output + '0_g.jpg',  g_)
    cv2.imwrite(arguments.output + '1_m.jpg',  m_)
    cv2.imwrite(arguments.output + '2_gr.jpg',  gr_)
    cv2.imwrite(arguments.output + '3_mr.jpg',  mr_)
    cv2.imwrite(arguments.output + '4_e.jpg',  e_)

    print('Done...')


def add_colour(img_comp,size_y,size_x):
    img_colour = np.zeros((3,size_y,size_x))
    img_colour[2] = img_comp
    img_colour[2][img_colour[2]<255] = 0
    img_colour[2][img_colour[2]>0] = 205
    img_colour[1] = img_comp
    img_colour[1][img_colour[1]==255] = 0
    img_colour[1][img_colour[1]>0] = 255
    img_colour = np.rollaxis(img_colour,0,3).astype('uint8')

    return img_colour