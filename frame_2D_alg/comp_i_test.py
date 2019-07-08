"""
For testing comparison operations and sub-blobs of different depths.
Requirements: numpy, frame_blobs, comp_i, utils.
Comparison is performed on i vars in derts, with non-i vars skipped.
"""

import numpy as np
from frame_blobs import comp_pixel
from comp_i import comp_i
from utils import imread, draw

# -----------------------------------------------------------------------------
# Adjustable parameters

image_path = "../images/raccoon.jpg"
init_ave = 20
angle_ave = 60
increase_ave = lambda ave, rng: ave * ((rng * 2 + 1) ** 2 - 1) / 2

# -----------------------------------------------------------------------------
# comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Initial comp:
    image = imread(image_path)
    input, dert = comp_pixel(image)
    ave = init_ave
    draw("../debug/g0", (dert[0] > ave) * 255)

    # Convert dert to derts:
    derts = [input[np.newaxis, ...], dert]

    # Angle comp from derts (forking):
    input, aderts = comp_i(derts, flags=F_ANGLE)
    draw("../debug/ga1", (aderts[-1][0] > angle_ave) * 255)

    # Rng comp from derts:
    for rng in range(1, 4):
        input, derts = comp_i(derts, rng=rng)
        ave *= ((rng * 2 + 1) ** 2 - 1) / 2
        draw("../debug/g%d" % (rng), (derts[-1][0] > ave) * 255)

    # Gradient comp from derts (forking):
    input, gderts = comp_i(derts, flags=F_DERIV)
    ave *= ((1 * 2 + 1) ** 2 - 1) / 2
    draw("../debug/gg1", (gderts[-1][0] > ave) * 255)
    for rng in range(2, 4):
        input, gderts = comp_i(gderts, rng=rng)
        ave *= ((rng * 2 + 1) ** 2 - 1) / 2
        draw("../debug/gg%d" % (rng), (gderts[-1][0] > ave) * 255)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------