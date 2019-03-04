import numpy as np
from collections import deque
# Filters ------------------------------------------------------------------------
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------

def hypot_gradient(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    sub_blob = [0, 0, 0, 0, []]  # initialize sub_blob
    seg_ = deque()
    dert__ = blob.dert__

    recalc_g(sub_blob, blob.dert__)

    for y in range(1, height - 1):
        P_ = generic_functions.form_P_(y, sub_blob)  # horizontal clustering
        P_ = generic_functions.scan_P_(P_, seg_, sub_blob)
        seg_ = generic_functions.form_seg_(P_, sub_blob)

    while seg_: generic_functions.form_blob(seg_.popleft(), sub_blob)
    blob.sub_blob_.append(sub_blob)

    return sub_blob
    # ---------- hypot_gradient() end -----------------------------------------------------------------------------------

def recalc_g(sub_blob, dert__):

    dert__.mask = ~blob.map
    dert__[:, :, 3] = np.hypot(dert__[:, :, 1], dert__[:, :, 2]) - ave
    sub_blob.append(dert__)
    # ---------- recalc_g() end -----------------------------------------------------------------------------------------