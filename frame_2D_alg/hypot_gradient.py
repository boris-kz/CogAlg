import numpy as np
import numpy.ma as ma
from collections import deque
# Filters ------------------------------------------------------------------------
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------

def hypot_gradient(blob):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape

    for i in range(4):
        blob.params.append(0)

    seg_ = deque()

    recalc_g(blob)

    for y in range(1, height - 1):
        P_ = generic_functions.form_P_(y, blob)  # horizontal clustering
        P_ = generic_functions.scan_P_(P_, seg_, blob)
        seg_ = generic_functions.form_seg_(P_, blob)

    while seg_: generic_functions.form_blob(seg_.popleft(), blob)

    return blob.sub_blob_[-1]
    # ---------- hypot_gradient() end -----------------------------------------------------------------------------------

def recalc_g(blob):

    blob.new_dert__ = ma.array(blob.dert__, mask=~blob.map)
    blob.new_dert__[:, :, 3] = np.hypot(blob.new_dert__[:, :, 1], blob.new_dert__[:, :, 2]) - ave
    # ---------- recalc_g() end -----------------------------------------------------------------------------------------