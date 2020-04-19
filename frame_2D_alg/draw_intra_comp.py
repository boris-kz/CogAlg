"""
For testing intra_comp operations and 3 layers of intra_comp's forks
Visualize each comp's output with image output
"""

import frame_blobs
from intra_comp import *
from utils import imread, imwrite
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./visualization/images/"

# -----------------------------------------------------------------------------
# Functions

def draw_g(img_out, g_):
    # loop in each row
    for y in range(g_.shape[0]):
        # loop across each dert
        for x in range(g_.shape[1]):
            # assign g to image
            img_out[y,x] = g_[y,x]

    return img_out.astype('uint8')

def draw_ga(img_out, g_):
    # loop in each row
    for y in range(g_.shape[0]):
        # loop across each dert
        for x in range(g_.shape[1]):
            # assign g to image
            img_out[y,x] = g_[y,x]

    # convert to degree
    img_out = img_out * 180/np.pi
    # scale 0 to 180 degree into 0 to 255
    img_out = (img_out /180 )*255

    return img_out.astype('uint8')

def draw_gr(img_out, g_, rng):
    # loop in each row, skip 1st row
    for y in range(g_.shape[0]):
        # loop across each dert, skip 1st dert
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+1+rng,(x*rng)+1:(x*rng)+1+rng] = g_[y,x]

    return img_out.astype('uint8')

def draw_gar(img_out, g_, rng):
    # loop in each row, skip 1st row
    for y in range(g_.shape[0]):
        # loop across each dert, skip 1st dert
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+3,(x*rng)+1:(x*rng)+3] = g_[y,x]

   # convert to degree
    img_out = img_out * 180/np.pi
    # scale 0 to 180 degree into 0 to 255
    img_out = (img_out /180 )*255

    return img_out.astype('uint8')

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(IMAGE_PATH)
    print('Done!')

    ## root layer #############################################################

    print('Doing first comp...')
    dert_tem_ = frame_blobs.comp_pixel(image)

    # add extra m channel to make the code run-able, comp pixel doesn't have m
    dert_ = np.zeros((5,dert_tem_.shape[1],dert_tem_.shape[2]))
    dert_[0:4] = dert_tem_[:]
    print('Done!')

    print('Processing first layer comps...')

    ga_dert_ = comp_a(dert_, fga = 0)  # if +G
    gr_dert_ = comp_r(dert_, fig = 0, root_fcr = 0)  # if -G

    print('Done!')
    print('Processing second layer comps...')

    # comp_a ->
    gaga_dert_ = comp_a(ga_dert_, fga = 1)  # if +Ga
    gg_dert_   = comp_g(ga_dert_)           # if -Ga
    # comp_r ->
    gagr_dert_ = comp_a(gr_dert_, fga = 0)                # if +Gr
    grr_dert_  = comp_r(gr_dert_, fig = 0, root_fcr = 1)  # if -Gr

    print('Done!')
    print('Processing third layer comps...')

    # comp_aga ->
    ga_gaga_dert_ = comp_a(gaga_dert_, fga = 1) # if +Gaga
    g_ga_dert_    = comp_g(gaga_dert_)          # if -Gaga
    # comp_g ->
    ga_gg_dert_ = comp_a(gg_dert_, fga = 0)                # if +Gg
    g_rg_dert_  = comp_r(gg_dert_, fig = 1, root_fcr = 0)  # if -Gg
    # comp_agr ->
    ga_gagr_dert_ = comp_a(gagr_dert_, fga = 1)  # if +Gagr
    g_gr_dert_    = comp_g(gagr_dert_)           # if -Gagr
    # comp_rr ->
    ga_grr_dert_ = comp_a(grr_dert_, fga = 0)                # if +Grr
    g_rrr_dert_  = comp_r(grr_dert_, fig = 0, root_fcr = 1)  # if -Grr：

    print('Done!')
    print('Drawing forks...')

    # initialize images:
    size_y = image.shape[0]
    size_x = image.shape[1]

    g = np.zeros((size_y,size_x))
    # 1st layer
    ga = np.zeros((size_y,size_x))
    gr = np.zeros((size_y,size_x))
    # 2nd layer
    gaga = np.zeros((size_y,size_x))
    gg = np.zeros((size_y,size_x))
    gagr = np.zeros((size_y,size_x))
    grr = np.zeros((size_y,size_x))
    # 3rd layer
    ga_gaga = np.zeros((size_y,size_x))
    g_ga = np.zeros((size_y,size_x))
    ga_gg = np.zeros((size_y,size_x))
    g_rg = np.zeros((size_y,size_x))
    ga_gagr = np.zeros((size_y,size_x))
    g_gr = np.zeros((size_y,size_x))
    ga_grr = np.zeros((size_y,size_x))
    g_rrr  = np.zeros((size_y,size_x))

    # draw each dert
    img_g = draw_g(g,dert_[1])
    # 1st layer
    img_ga = draw_ga(ga, ga_dert_[5])
    img_gr = draw_gr(gr, gr_dert_[1], rng=2)
    # 2nd layer
    img_gaga = draw_ga(gaga, gaga_dert_[5])
    img_gg =   draw_g(gg, gaga_dert_[1])
    img_gagr = draw_gar(gagr, gagr_dert_[5], rng=2)
    img_grr =  draw_gr(grr, gagr_dert_[1], rng=4)
    # 3rd layer
    img_ga_gaga = draw_ga(ga_gaga, gaga_dert_[5])
    img_g_ga    = draw_g(g_ga, gaga_dert_[1])
    img_ga_gg   = draw_ga(ga_gg, gg_dert_[5])
    img_g_rg    = draw_gr(g_rg, gg_dert_[1], rng=2)
    img_ga_gagr = draw_gar(ga_gagr, gagr_dert_[5], rng=2)
    img_g_gr    = draw_gr(g_gr, gagr_dert_[1], rng=2)
    img_ga_grr  = draw_gar(ga_grr, grr_dert_[5], rng=4)
    img_g_rrr   = draw_gr(g_rrr, grr_dert_[1], rng=8)

    # save to disk
    cv2.imwrite('./images/intra_comp/g.png', img_g)
    cv2.imwrite('./images/intra_comp/ga.png', img_ga)
    cv2.imwrite('./images/intra_comp/gr.png', img_gr)
    cv2.imwrite('./images/intra_comp/gaga.png', img_gaga)
    cv2.imwrite('./images/intra_comp/gg.png', img_gg)
    cv2.imwrite('./images/intra_comp/gagr.png', img_gagr)
    cv2.imwrite('./images/intra_comp/grr.png', img_grr)
    cv2.imwrite('./images/intra_comp/ga_gaga.png', img_ga_gaga)
    cv2.imwrite('./images/intra_comp/g_ga.png', img_g_ga)
    cv2.imwrite('./images/intra_comp/ga_gg.png', img_ga_gg)
    cv2.imwrite('./images/intra_comp/g_rg.png', img_g_rg)
    cv2.imwrite('./images/intra_comp/ga_gagr.png', img_ga_gagr)
    cv2.imwrite('./images/intra_comp/g_gr.png', img_g_gr)
    cv2.imwrite('./images/intra_comp/ga_grr.png', img_ga_grr)
    cv2.imwrite('./images/intra_comp/g_rrr.png', img_g_rrr)

    print('Done!')
    print('Terminating...')


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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#    # size of image
#    size_y = image.shape[0]
#    size_x = image.shape[1]
#
#
#    from matplotlib import pyplot as plt
#
#    plt.figure()
#    plt.imshow(dert_)
#
#    # initialize each image for visualization
#    img_comp_pixel = np.zeros((size_y,size_x))
#    img_adert_ag = np.zeros((size_y,size_x))
#    img_adert_aga = np.zeros((size_y,size_x))
#    img_gdert_g = np.zeros((size_y,size_x))
#
#    # loop in each row
#    for y in range(size_y):
#        # loop across each dert
#        for x in range(size_x):
#
#            try:
#                # derts' g
#                g = dert_[1][y,x] - ave
#                # +g fork
#                if g>=0:
#
#                    # add value to +g
#                    img_comp_pixel[y,x] = 255;
#                    # adert's ga
#                    ga = adert_ag[1][y,x] - ave_adert_ag
#
#                    # +ga fork
#                    if ga>=0:
#
#                        # add value to +ga
#                        img_adert_ag[y,x] = 255
#                        # adert's gaga
#                        gaga = adert_aga[1][y,x] - ave_adert_aga
#
#                        # +gaga fork
#                        if gaga>=0:
#
#                            # add value to +gaga
#                            img_adert_aga[y,x] = 255
#                            # comp_aga_ga
#
#                        # -gaga fork
#                        else:
#
#                            # add value to -gaga
#                            img_adert_aga[y,x] = 128
#
#                            # comp_ga
#
#                    # -ga fork
#                    else:
#
#                        # add value to -ga
#                        img_adert_ag[y,x] = 128
#
#                        # adert's gaga
#                        gg = gdert_g[1][y,x] - ave_gdert_g
#
#                        # +gg fork
#                        if gg>=0:
#
#                            # add value to +gg
#                            img_gdert_g[y,x] = 255
#                            # comp_agg
#
#                        # -gg fork
#                        else:
#
#                            # add value to -gg
#                            img_gdert_g[y,x] = 128
#                            # comp_rng_g
#
#                # -g fork
#                else:
#
#                    # add value to -g
#                    img_comp_pixel[y,x] = 128;
#
#                    # comp_rng_p
#                    # comp_agr
#                    # comp_rrp
#
#            except:
#                pass
#
#    print('Done!')
#    print('Saving images to disk...')
#
#    # add colour
#    # where red = +g, green = -g
#    img_colour_comp_pixel = add_colour(img_comp_pixel,size_y,size_x)
#    img_colour_adert_ag = add_colour(img_adert_ag,size_y,size_x)
#    img_colour_adert_aga = add_colour(img_adert_aga,size_y,size_x)
#    img_colour_gdert_g = add_colour(img_gdert_g,size_y,size_x)
#
#    # save to disk
#    cv2.imwrite('./images/image_colour_comp_pixel.png',img_colour_comp_pixel)
#    cv2.imwrite('./images/image_colour_adert_ag.png',img_colour_adert_ag)
#    cv2.imwrite('./images/image_colour_adert_aga.png',img_colour_adert_aga)
#    cv2.imwrite('./images/image_colour_gdert_g.png',img_colour_gdert_g)
#    cv2.imwrite('./images/image_comp_pixel.png',img_comp_pixel)
#    cv2.imwrite('./images/image_adert_ag.png',img_adert_ag)
#    cv2.imwrite('./images/image_adert_aga.png',img_adert_aga)
#    cv2.imwrite('./images/image_gdert_g.png',img_gdert_g)
#
#    print('Done!')
#    print('Terminating...')