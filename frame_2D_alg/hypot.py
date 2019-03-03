from collections import deque
# Filters ------------------------------------------------------------------------
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------

# ***************************************************** ANGLE BLOBS FUNCTIONS *******************************************
# Functions:
# -refine_gblobs()
# -get_angle()
# -comp_angle()
# ***********************************************************************************************************************

def refine_gblobs(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    sub_blob = []  # initialize sub_blob
    seg_ = deque()
    a__ = get_angle(blob.dert__, blob.map)  # compute maximal gradients' magnitude and angle within gblob
    comp_angle(sub_blob, a__)

    for y in range(1, Y - 1):


    y = Y - 1   # sub_blob ends, merge segs of last line into their blobs:
    while seg_: form_blob(y, seg_.popleft(), sub_blob)

    blob.sub_blob_.append(sub_blob)
    return sub_blob
    # ---------- refine_gblobs() end ------------------------------------------------------------------------------------
    # ---------- blob_to_ablobs() end -----------------------------------------------------------------------------------