"""
Visualize output of first 3 layers of intra_comp forks, for testing
"""

from frame_2D_alg.comp_pixel_versions import comp_pixel_m
from frame_2D_alg.intra_comp_ma import *

import cv2
import argparse
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "../images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./images/intra_comp0/"

# -----------------------------------------------------------------------------
# Functions

def draw_g(img_out, g_):
    endy = min(img_out.shape[0], g_.shape[0])
    endx = min(img_out.shape[1], g_.shape[1])
    img_out[:endy, :endx] = (g_[:endy, :endx] * 255) / g_.max()  # scale to max=255, less than max / 255 is 0

    return img_out

def draw_gr(img_out, g_, rng):

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

    print('Processing first layer comps...')
    # comp_p ->
    gr_dert_, _ = comp_r(dert_, fig = 0, root_fcr = 0)         # if   +M
    gg_dert_, _ = comp_g(dert_)                                # elif +G

    print('Processing second layer comps...')
    # comp_g ->
    grg_dert_, _ = comp_r(gg_dert_, fig = 1, root_fcr = 0)    # if   +Mg
    ggg_dert_, _ = comp_g(gg_dert_)                           # elif +Gg
    # comp_r ->
    grr_dert_, _ = comp_r(gr_dert_, fig = 0, root_fcr = 1)    # if   +Mr
    ggr_dert_, _ = comp_g(gr_dert_)                           # elif +Gr

    print('Processing third layer comps...')
    # comp_gg ->
    grgg_dert_, _ = comp_r(ggg_dert_, fig = 1, root_fcr = 0)   # if   +Mgg
    gggg_dert_, _ = comp_g(ggg_dert_)                          # elif +Ggg
    # comp_rg ->
    grrg_dert_, _ = comp_r(grg_dert_, fig = 0, root_fcr = 1)  # if   +Mrg
    ggrg_dert_, _ = comp_g(grg_dert_)                          # elif +Grg
    # comp_gr ->
    grgr_dert_, _ = comp_r(ggr_dert_, fig = 1, root_fcr = 0)   # if   +Mgr
    gggr_dert_, _ = comp_g(ggr_dert_)                       # elif +Ggr
    # comp_rrp ->
    grrr_dert_, _ = comp_r(grr_dert_, fig = 0, root_fcr = 1)   # if   +Mrr
    ggrr_dert_, _ = comp_g(grr_dert_)                         # elif +Grr

    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]), 'uint8')  # initialize image y, x

    # 0th layer
    g_ = draw_g(ini_.copy(), dert_[3])
    m_ = draw_g(ini_.copy(), dert_[6])
    # 1st layer
    gg_ = draw_g(ini_.copy(), gg_dert_[3])
    mg_ = draw_g(ini_.copy(), gg_dert_[6])
    gr_ = draw_gr(ini_.copy(), gr_dert_[3], rng=2)
    mr_ = draw_gr(ini_.copy(), gr_dert_[6], rng=2)
    # 2nd layer
    ggg_ = draw_g(ini_.copy(), ggg_dert_[3])
    mgg_ = draw_g(ini_.copy(), ggg_dert_[6])
    grg_ = draw_gr(ini_.copy(), grg_dert_[3], rng=2)
    mrg_ = draw_gr(ini_.copy(), grg_dert_[6], rng=2)
    ggr_ = draw_gr(ini_.copy(), ggr_dert_[3], rng=2)
    mgr_ = draw_gr(ini_.copy(), ggr_dert_[6], rng=2)
    grr_ = draw_gr(ini_.copy(), grr_dert_[3], rng=4)
    mrr_ = draw_gr(ini_.copy(), grr_dert_[6], rng=4)

    # 3rd layer
    gggg_ = draw_g(ini_.copy(), gggg_dert_[3])
    mggg_ = draw_g(ini_.copy(), gggg_dert_[6])
    grgg_ = draw_gr(ini_.copy(), grgg_dert_[3], rng=2)
    mrgg_ = draw_gr(ini_.copy(), grgg_dert_[6], rng=2)
    ggrg_ = draw_gr(ini_.copy(), ggrg_dert_[3], rng=2)
    mgrg_ = draw_gr(ini_.copy(), ggrg_dert_[6], rng=2)
    grrg_ = draw_gr(ini_.copy(), grrg_dert_[3], rng=4)
    mrrg_ = draw_gr(ini_.copy(), grrg_dert_[6], rng=4)
    gggr_ = draw_gr(ini_.copy(), gggr_dert_[3], rng=2)
    mggr_ = draw_gr(ini_.copy(), gggr_dert_[6], rng=2)
    grgr_ = draw_gr(ini_.copy(), grgr_dert_[3], rng=4)
    mrgr_ = draw_gr(ini_.copy(), grgr_dert_[6], rng=4)
    ggrr_ = draw_gr(ini_.copy(), ggrr_dert_[3], rng=4)
    mgrr_ = draw_gr(ini_.copy(), ggrr_dert_[6], rng=4)
    grrr_ = draw_gr(ini_.copy(), grrr_dert_[3], rng=8)
    mrrr_ = draw_gr(ini_.copy(), grrr_dert_[6], rng=8)

    # save to disk
    cv2.imwrite(arguments.output + '0_g.jpg',  g_)
    cv2.imwrite(arguments.output + '1_m.jpg',  m_)

    cv2.imwrite(arguments.output + '2_gg.jpg',  gg_)
    cv2.imwrite(arguments.output + '3_mg.jpg',  mg_)
    cv2.imwrite(arguments.output + '4_gr.jpg',  gr_)
    cv2.imwrite(arguments.output + '5_mr.jpg',  mr_)

    cv2.imwrite(arguments.output + '6_ggg.jpg',  ggg_)
    cv2.imwrite(arguments.output + '7_mgg.jpg',  mgg_)
    cv2.imwrite(arguments.output + '8_grg.jpg',  grg_)
    cv2.imwrite(arguments.output + '9_mrg.jpg',  mrg_)
    cv2.imwrite(arguments.output + '10_ggr.jpg',  ggr_)
    cv2.imwrite(arguments.output + '11_mgr.jpg',  mgr_)
    cv2.imwrite(arguments.output + '12_grr.jpg',  grr_)
    cv2.imwrite(arguments.output + '13_mrr.jpg',  mrr_)

    cv2.imwrite(arguments.output + '14_gggg.jpg',  gggg_)
    cv2.imwrite(arguments.output + '15_mggg.jpg',  mggg_)
    cv2.imwrite(arguments.output + '16_grgg.jpg',  grgg_)
    cv2.imwrite(arguments.output + '17_mrgg.jpg',  mrgg_)
    cv2.imwrite(arguments.output + '18_ggrg.jpg',  ggrg_)
    cv2.imwrite(arguments.output + '19_mgrg.jpg',  mgrg_)
    cv2.imwrite(arguments.output + '20_grrg.jpg',  grrg_)
    cv2.imwrite(arguments.output + '21_mrrg.jpg',  mrrg_)
    cv2.imwrite(arguments.output + '22_gggr.jpg',  gggr_)
    cv2.imwrite(arguments.output + '23_mggr.jpg',  mggr_)
    cv2.imwrite(arguments.output + '24_grgr.jpg',  grgr_)
    cv2.imwrite(arguments.output + '25_mrgr.jpg',  mrgr_)
    cv2.imwrite(arguments.output + '26_ggrr.jpg',  ggrr_)
    cv2.imwrite(arguments.output + '27_mgrr.jpg',  mgrr_)
    cv2.imwrite(arguments.output + '28_grrr.jpg',  grrr_)
    cv2.imwrite(arguments.output + '29_mrrr.jpg',  mrrr_)

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