import cv2
import argparse


def image_to_patterns(i):
    frame_ = []

    _vp_, vb_, vn_ = [], [], []
    _dp_, db_, dn_ = [], [], []

    t2__ = []

    t_ = horisontal_comparison(i[0, :])

    for t in t_:
        p, d, m = t
        t2__.append((p, d, 0, m, 0))

    for y in range(1, ih):
        p_ = i[y, :]
        t_ = horisontal_comparison(p_)
        t2__, _vp_, _dp_, vb_, db_, vn_, dn_ = vertical_comparison(t_, t2__, _vp_, _dp_, vb_, db_, vn_, dn_)

        frame_.append((vn_, dn_))

    return frame_


def horisontal_comparison(p_):
    return p_


def vertical_comparison(t_, t2__, _vp_, _dp_, vb_, db_, vn_, dn_):
    return t2__, _vp_, _dp_, vb_, db_, vn_, dn_


# initialize arguments to be parsed from console
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
args = vars(ap.parse_args())

# read image as 2d-array (gray scale) with height and width
i = cv2.imread(args['image'], 0).astype(int)
ih, iw = i.shape

# initialize pattern filters here as constants for high-level feedback
pr = 1
am = 127
ac = 0.25

# input image in root function to get list of patterns
patterns = image_to_patterns(i)
