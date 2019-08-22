import cv2
import numpy as np

from math import hypot
from collections import deque


def lateral_comp(dert_, rng, x_start=0):

    new_dert_ = []

    dert_buff_ = deque(maxlen=rng)

    max_index = rng - 1

    for x, (i, ncomp, dx, dy) in enumerate(dert_):

        if len(dert_buff_) == rng:  # xd == rng

            _i, _ncomp, _dx, _dy = dert_buff_[max_index]

            d = i - _i      # lateral comparison

            ncomp += 1      # bilateral accumulation
            dx += d         # bilateral accumulation
            _ncomp += 1     # bilateral accumulation
            _dx += d        # bilateral accumulation

            new_dert_.append((_i, _ncomp, _dx, _dy))

        dert_buff_.appendleft((i, ncomp, dx, dy))

    while dert_buff_:
        new_dert_.append(dert_buff_.pop())

    return new_dert_

def vertical_comp(dert__, dert_, dert_buff__, rng=1):

    for yd, _dert_ in enumerate(dert_buff__, start=1):  # yd: distance between dert_ and _dert_

        if yd == rng:
            new_dert_ = []
            for x, ((i, ncomp, dx, dy), (_i, _ncomp, _dx, _dy)) in enumerate(zip(dert_, _dert_)):

                d = i - _i      # vertical comparison

                ncomp += 1      # bilateral accumulation
                dy += d         # bilateral accumulation
                _ncomp += 1     # bilateral accumulation
                _dy += d        # bilateral accumulation

                new_dert_.append((_i, _ncomp, _dx, _dy))
                dert_[x] = i, ncomp, dx, dy

            dert__.append(new_dert_)
        else:
            xd = rng - yd

            for (x, (i, ncomp, dx, dy)), (_x, (_i, _ncomp, _dx, _dy)) in zip(enumerate(dert_[xd:], xd), enumerate(_dert_[:-xd])):

                d = i - _i      # vertical comparison

                x_coef = xd / hypot(xd, yd)
                y_coef = yd / hypot(xd, yd)

                ncomp += 1      # bilateral accumulation
                dx += int(d * x_coef)
                dy += int(d * y_coef)
                _ncomp += 1     # bilateral accumulation
                _dx += int(d * x_coef)
                _dy += int(d * y_coef)

                _dert_[_x] = _i, _ncomp, _dx, _dy
                dert_[x] = i, ncomp, dx, dy

            xd = -xd

            for (x, (i, ncomp, dx, dy)), (_x, (_i, _ncomp, _dx, _dy)) in zip(enumerate(dert_[:xd]), enumerate(_dert_[-xd:], xd)):

                d = i - _i      # vertical comparison

                x_coef = xd / hypot(xd, yd)
                y_coef = yd / hypot(xd, yd)

                ncomp += 1      # bilateral accumulation
                dx += int(d * x_coef)
                dy += int(d * y_coef)
                _ncomp += 1     # bilateral accumulation
                _dx += int(d * x_coef)
                _dy += int(d * y_coef)

                _dert_[_x] = _i, _ncomp, _dx, _dy
                dert_[x] = i, ncomp, dx, dy

    dert_buff__.appendleft(dert_)

image = cv2.imread('../images/raccoon_eye.jpg', 0).astype(int)

Y, X = image.shape

rng = 2

ave = 5

dert__ = []

dert_buff__ = deque(maxlen=rng) # vertical buffer for dert_

for y in range(Y):

    dert_ = [(p, 0, 0, 0) for p in image[y, :]]

    dert_ = lateral_comp(dert_, rng=rng, x_start=0)

    vertical_comp(dert__, dert_, dert_buff__, rng=rng)

while dert_buff__:
    dert__.append(dert_buff__.pop())

image = np.array([[(abs(dx) + abs(dy) - ave * ncomp > 0) * 255 for p, ncomp, dx, dy in dert_] for dert_ in dert__])

cv2.imwrite('./../visualization/images/out.bmp', image)

print('Done!')