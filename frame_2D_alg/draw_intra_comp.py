"""
For testing intra_comp operations and 3 layers of intra_comp's forks
Visualize each comp's output with image output
"""
import frame_blobs
from intra_comp import *
from utils import imread, imwrite
import cv2

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/raccoon.jpg"

# Outputs:
OUTPUT_PATH = "./visualization/images/"

# aves
ave = 14
ave_a = 18
ave_aga = 22
ave_g = 22

# -----------------------------------------------------------------------------
# Functions

# add colour
def add_colour(img_comp, size_y, size_x):
    img_colour = np.zeros((3, size_y, size_x))
    img_colour[2] = img_comp
    img_colour[2][img_colour[2] < 255] = 0
    img_colour[2][img_colour[2] > 0] = 205
    img_colour[1] = img_comp
    img_colour[1][img_colour[1] == 255] = 0
    img_colour[1][img_colour[1] > 0] = 255
    img_colour = np.rollaxis(img_colour, 0, 3).astype('uint8')

    return img_colour

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(IMAGE_PATH)
    print('Done!')

    ## root layer #############################################################

    print('Doing first comp...')
    dert_ = frame_blobs.comp_pixel(image)
    print('Done!')

    ## 1st layer ##############################################################

    print('Processing first layer comps...')
    ## 2 forks from comp_pixel ##
    # if +G：
    # comp_a (comp_ag)
    adert_ag = comp_a(dert_, fga=0)
    # if -G：
    # comp_r (comp_rng_p)
    rdert_rng_p = comp_r(dert_, fig=0, root_fcr=0)
    print('Done!')

    ## 2nd layer ##############################################################

    print('Processing second layer comps...')
    ## 2 forks from comp_ag ##
    # if +Ga：
    # comp_a (comp_aga)
    adert_aga = comp_a(adert_ag, fga=1)
    # if -Ga：
    # comp_g (comp_g)
    gdert_g = comp_g(adert_ag)

    ## 2 forks from comp_rng_p ##
    # if +Gr:
    # comp_a(comp_agr)
    adert_agr = comp_a(rdert_rng_p, fga=0)
    # if -Gr:
    # comp_r (comp_rrp)
    rdert_rrp = comp_r(rdert_rng_p, fig=0, root_fcr=1)
    print('Done!')

    ## 3rd layer ##############################################################

    print('Processing third layer comps...')
    ## 2 forks from comp_aga ##
    # if +Gaga：
    # comp_a (comp_aga_ga)
    adert_aga_ga = comp_a(adert_aga, fga=1)
    # if -Gaga：
    # comp_g (comp_ga)
    gdert_ga = comp_g(adert_aga)

    ## 2 forks from comp_g ##
    # if +Gg:
    # comp_a (comp_agg)
    adert_agg = comp_a(gdert_g, fga=0)
    # if -Gg:
    # comp_r (comp_rng_g)
    rdert_rng_g = comp_r(gdert_g, fig=1, root_fcr=0)

    ## 2 forks from comp_agr ##
    # if +Gagr：
    # comp_a (comp_aga_gr)
    adert_aga_gr = comp_a(adert_agr, fga=1)
    # if -Gagr：
    # comp_g (comp_gr)
    gdert_gr = comp_g(adert_agr)

    ## 2 forks from comp_rrp ##
    # if +Grr：
    # comp_a (comp_a_grr)
    adert_a_grr = comp_a(rdert_rrp, fga=0)
    # if -Grr：
    # comp_r (comp_rrrp)
    rdert_rrrp = comp_r(gdert_g, fig=0, root_fcr=1)
    print('Done!')

    ###########################################################################

    print('Drawing each comps...')

    # size of image
    size_y = image.shape[0]
    size_x = image.shape[1]

    # initialize each image for visualization
    img_comp_pixel = np.zeros((size_y, size_x))
    img_adert_ag = np.zeros((size_y, size_x))
    img_adert_aga = np.zeros((size_y, size_x))
    img_gdert_g = np.zeros((size_y, size_x))

    # loop in each row
    for y in range(size_y):
        # loop across each dert
        for x in range(size_x):
            try:
                # derts' g
                g = dert_[1][y, x] - ave
                # +g fork
                if g >= 0:
                    # add value to +g
                    img_comp_pixel[y, x] = 255;
                    # adert's ga
                    ga = adert_ag[1][y, x] - ave_adert_ag
                    # +ga fork
                    if ga >= 0:
                        # add value to +ga
                        img_adert_ag[y, x] = 255
                        # adert's gaga
                        gaga = adert_aga[1][y, x] - ave_adert_aga
                        # +gaga fork
                        if gaga >= 0:
                            # add value to +gaga
                            img_adert_aga[y, x] = 255
                            # comp_aga_ga
                        # -gaga fork
                        else:
                            # add value to -gaga
                            img_adert_aga[y, x] = 128
                            # comp_ga
                    # -ga fork
                    else:
                        # add value to -ga
                        img_adert_ag[y, x] = 128
                        # adert's gaga
                        gg = gdert_g[1][y, x] - ave_gdert_g
                        # +gg fork
                        if gg >= 0:
                            # add value to +gg
                            img_gdert_g[y, x] = 255
                            # comp_agg
                        # -gg fork
                        else:
                            # add value to -gg
                            img_gdert_g[y, x] = 128
                            # comp_rng_g
                # -g fork
                else:
                    # add value to -g
                    img_comp_pixel[y, x] = 128;
                    # comp_rng_p
                    # comp_agr
                    # comp_rrp
            except:
                pass

    print('Done!')
    print('Saving images to disk...')
    # add colour
    # where red = +g, green = -g
    img_colour_comp_pixel = add_colour(img_comp_p, size_y, size_x)
    img_colour_adert_ag = add_colour(img_comp_a, size_y, size_x)
    img_colour_adert_aga = add_colour(img_comp_aga, size_y, size_x)
    img_colour_gdert_g = add_colour(img_comp_g, size_y, size_x)

    # save to disk
    cv2.imwrite('./images/image_colour_comp_p.png', img_colour_comp_pixel)
    cv2.imwrite('./images/image_colour_comp_a.png', img_colour_adert_ag)
    cv2.imwrite('./images/image_colour_comp_aga.png', img_colour_adert_aga)
    cv2.imwrite('./images/image_colour_comp_g.png', img_colour_gdert_g)
    cv2.imwrite('./images/image_comp_p.png', img_comp_p)
    cv2.imwrite('./images/image_comp_a.png', img_comp_a)
    cv2.imwrite('./images/image_comp_aga.png', img_comp_aga)
    cv2.imwrite('./images/image_comp_g.png', img_comp_g)

    print('Done!')
    print('Terminating...')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------