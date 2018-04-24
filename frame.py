import cv2
import argparse


def image_to_patterns(image):
    t_ = horisontal_comparison(image[0, :])

    t2__ = []

    for t in t_:
        p, d, m = t
        t2__.append([p, d, 0, m, 0])

    frame_ = []

    for y in range(1, height):
        p_ = image[y, :]
        t_ = horisontal_comparison(p_)
        t2__, _vp_, _dp_, vb_, db_, vn_, dn_ = vertical_comparison(t_, t2__)

        frame_.append([vn_, dn_])

    return frame_


def horisontal_comparison(p_):
    t_ = []
    it_ = []

    for p in p_:
        for index, it in enumerate(it_):
            pri_p, fd, fm = it

            fd += p - pri_p
            fm += min(p, pri_p)

            it_[index] = [pri_p, fd, fm]

        if len(it_) == pixel_range:
            t_.append([pri_p, fd, fm])

        it_.append([p, 0, 0])

    t_ += it_

    return t_


def vertical_comparison(t_, t2__, _vp_=[], _dp_=[], vb_=[], db_=[], vn_=[], dn_=[]):
    return t2__, _vp_, _dp_, vb_, db_, vn_, dn_


# initialize arguments to be parsed from console
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())

# read image as 2d-array (gray scale) with height and width
image = cv2.imread(arguments['image'], 0).astype(int)
height, width = image.shape

# initialize pattern filters here as constants for high-level feedback
pixel_range = 1
accuracy_match = 127
accuracy_coefficient = 0.25

# input image in root function to get list of patterns
patterns = image_to_patterns(image)
