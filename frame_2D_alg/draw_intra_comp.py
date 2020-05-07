"""
For testing intra_comp operations and 3 layers of intra_comp's forks
Visualize each comp's output with image output
"""

from frame_2D_alg.comp_pixel import comp_pixel_m
from frame_2D_alg.intra_comp import *
from frame_2D_alg.utils import imread, imwrite
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./images/intra_comp0/"

# -----------------------------------------------------------------------------
# Functions

def draw_g(img_out, g_):

    for y in range(g_.shape[0]):  # loop rows, skip last row
        for x in range(g_.shape[1]):  # loop columns, skip last column
            img_out[y,x] = g_[y,x]

    return img_out.astype('uint8')

def draw_gr(img_out, g_, rng):

    for y in range(g_.shape[0]):
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+1+rng,(x*rng)+1:(x*rng)+1+rng] = g_[y,x]

    return img_out.astype('uint8')

# -----------------------------------------------------------------------------
# Main


if __name__ == "__main__":

    print('Reading image...')
    image = imread(IMAGE_PATH)

    dert_ = comp_pixel_m(image)

    print('Processing first layer comps...')
    # comp_p ->
    gr_dert_ = comp_r(dert_, fig = 0, root_fcr = 0)         # if   +M
    gg_dert_ = comp_g(dert_)                                # elif +G

    print('Processing second layer comps...')
    # comp_g ->
    grg_dert_  = comp_r(gg_dert_, fig = 1, root_fcr = 0)    # if   +Mg
    ggg_dert_  = comp_g(gg_dert_)                           # elif +Gg
    # comp_r ->
    grr_dert_  = comp_r(gr_dert_, fig = 0, root_fcr = 1)    # if   +Mr
    ggr_dert_  = comp_g(gr_dert_)                           # elif +Gr

    print('Processing third layer comps...')
    # comp_gg ->
    grgg_dert_ = comp_r(ggg_dert_, fig = 1, root_fcr = 0)   # if   +Mgg
    gggg_dert_ = comp_g(ggg_dert_)                          # elif +Ggg
    # comp_rg ->
    grrg_dert_  = comp_r(grg_dert_, fig = 0, root_fcr = 1)  # if   +Mrg
    ggrg_dert_ = comp_g(grg_dert_)                          # elif +Grg
    # comp_gr ->
    grgr_dert_ = comp_r(ggr_dert_, fig = 1, root_fcr = 0)   # if   +Mgr
    gggr_dert_    = comp_g(ggr_dert_)                       # elif +Ggr
    # comp_rrp ->
    grrr_dert_ = comp_r(grr_dert_, fig = 0, root_fcr = 1)   # if   +Mrr
    ggrr_dert_  = comp_g(grr_dert_)                         # elif +Grr

    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]))  # initialize image y, x

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
    cv2.imwrite(OUTPUT_PATH+'0_g.png', g_)
    cv2.imwrite(OUTPUT_PATH+'1_m.png', m_)

    cv2.imwrite(OUTPUT_PATH+'2_gg.png', gg_)
    cv2.imwrite(OUTPUT_PATH+'3_mg.png', mg_)
    cv2.imwrite(OUTPUT_PATH+'4_gr.png', gr_)
    cv2.imwrite(OUTPUT_PATH+'5_mr.png', mr_)

    cv2.imwrite(OUTPUT_PATH+'6_ggg.png', ggg_)
    cv2.imwrite(OUTPUT_PATH+'7_mgg.png', mgg_)
    cv2.imwrite(OUTPUT_PATH+'8_grg.png', grg_)
    cv2.imwrite(OUTPUT_PATH+'9_mrg.png', mrg_)
    cv2.imwrite(OUTPUT_PATH+'10_ggr.png', ggr_)
    cv2.imwrite(OUTPUT_PATH+'11_mgr.png', mgr_)
    cv2.imwrite(OUTPUT_PATH+'12_grr.png', grr_)
    cv2.imwrite(OUTPUT_PATH+'13_mrr.png', mrr_)

    cv2.imwrite(OUTPUT_PATH+'14_gggg.png', gggg_)
    cv2.imwrite(OUTPUT_PATH+'15_mggg.png', mggg_)
    cv2.imwrite(OUTPUT_PATH+'16_grgg.png', grgg_)
    cv2.imwrite(OUTPUT_PATH+'17_mrgg.png', mrgg_)
    cv2.imwrite(OUTPUT_PATH+'18_ggrg.png', ggrg_)
    cv2.imwrite(OUTPUT_PATH+'19_mgrg.png', mgrg_)
    cv2.imwrite(OUTPUT_PATH+'20_grrg.png', grrg_)
    cv2.imwrite(OUTPUT_PATH+'21_mrrg.png', mrrg_)
    cv2.imwrite(OUTPUT_PATH+'22_gggr.png', gggr_)
    cv2.imwrite(OUTPUT_PATH+'23_mggr.png', mggr_)
    cv2.imwrite(OUTPUT_PATH+'24_grgr.png', grgr_)
    cv2.imwrite(OUTPUT_PATH+'25_mrgr.png', mrgr_)
    cv2.imwrite(OUTPUT_PATH+'26_ggrr.png', ggrr_)
    cv2.imwrite(OUTPUT_PATH+'27_mgrr.png', mgrr_)
    cv2.imwrite(OUTPUT_PATH+'28_grrr.png', grrr_)
    cv2.imwrite(OUTPUT_PATH+'29_mrrr.png', mrrr_)

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