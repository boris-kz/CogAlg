from File_IO import Read_Image
from Operations import Compare

input_ = Read_Image('../images/raccoon_eye.jpg')
rng1_horizontal = Compare(input_, offset=(0, 1))
rng1_vertical = Compare(input_, offset=(1, 0))

import cv2
import numpy as np

map_image = np.array([[128] * 64] * 64)

dert_ = []

for inp, out1, out2 in zip(input_, rng1_horizontal, rng1_vertical):

    ncomp1, dy1, dx1 = out1
    ncomp2, dy2, dx2 = out2
    y, x, p          = inp

    ncomp = ncomp1 + ncomp2
    dy = dy1 + dy2
    dx = dx1 + dx2

    g = abs(dy) + abs(dx) - ncomp * 4

    dert = y, x, p, (ncomp, dy, dx), g

    s = g > 0

    # Form_P_

    if map_image[y, x] != 128:
        print("overlap at (%d, %d)!\n" %(y, x))

    if ncomp == 0:
        print("ncomp = 0 at (%d, %d)!\n" % (y, x))

    map_image[y, x] = s * 255

cv2.imwrite('../debug/map.bmp', map_image)