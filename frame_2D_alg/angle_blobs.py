import numpy as np
from collections import deque
# Filters ------------------------------------------------------------------------
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
'''
    angle_blob is a component of intra_blob
'''
# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -blob_to_ablobs()
# -get_angle()
# -comp_angle()
# ***********************************************************************************************************************

def blob_to_ablobs(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    sub_blob = []  # initialize sub_blob

    seg_ = deque()

    for y in range(1, Y - 1):

    blob.sub_blob_.append(sub_blob)
    return sub_blob
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------

def get_gradient(dert__, map):  # default = False: no higher-line for first line
    " compute angle of gradient in and adjacent to selected gblob "

    dy = dert__[:, :, 1]
    dx = dert__[:, :, 2]

    a__ = np.empty(map.shape, dtype=int)

    a__[map] = np.arctan2(dy, dx, where=[map])[map] * angle_coef + 128

    return a__
    # ---------- get_gradient() end -------------------------------------------------------------------------------------

def comp_angle(a__, sub_blob):
    " compare angle of adjacent gradients within frame per gblob "

    # ---------- comp_angle() end ---------------------------------------------------------------------------------------