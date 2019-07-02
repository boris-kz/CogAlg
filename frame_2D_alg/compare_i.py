'''
Comparison of chosen parameter of derts__ (a or g)
over predetermined range (determined by kernel).
'''

import numpy as np

from frame_blobs import convolve

# flags:
f_angle = 0b01
f_derive = 0b10

# ************ FUNCTIONS ************************************************************************************************
# -compare_i()
# ***********************************************************************************************************************

def compare_i(derts__, kernel, flags):    # Currently under revision

    if flag & f_derive:
        g__ = derts__[0]


    # ---------- compare_i() end --------------------------------------------------------------------------------------------