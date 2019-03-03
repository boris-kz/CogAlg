import numpy as np
from collections import deque
import generic_functions
from frame_2D_alg_classes.misc import get_filters
get_filters(globals()) # imports all filters at once
# --------------------------------------------------------------------------------

# ***************************************************** INC_DERIV FUNCTIONS *********************************************
# Functions:
# -inc_deriv()
# -comp_g()
# ***********************************************************************************************************************

def inc_deriv(blob):
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    sub_blob = [0, 0, 0, 0, []]

    comp_g(sub_blob, blob.dert__, blob.map)

    seg_ = deque()

    for y in range(Y - 1):


    blob.sub_blob_.append(sub_blob)

    return sub_blob
    # ---------- inc_deriv() end ----------------------------------------------------------------------------------------
def comp_g(sub_blob, dert__, map):
    " compare g within sub blob "

    # ---------- comp_g() end -------------------------------------------------------------------------------------------