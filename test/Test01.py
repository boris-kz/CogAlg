from File_IO import Read_Image
from Operations import Compare

input_ = Read_Image('../images/raccoon_eye.jpg')
rng1_horizontal = Compare(input_, offset=(0, 1))
rng1_vertical = Compare(input_, offset=(1, 0))

import cv2
import numpy as np

map_image = np.empty((64, 64), dtype=int)

for inp, out1, out2 in zip(input_, rng1_horizontal, rng1_vertical):

    ncomp1, dy1, dx1 = out1
    ncomp2, dy2, dx2 = out2

    s = abs(dy1 + dy2) + abs(dx1 + dx2) > (ncomp1 + ncomp2) * 15

    y, x, p = inp

    map_image[y, x] = s * 255

cv2.imwrite('../debug/map.bmp', map_image)