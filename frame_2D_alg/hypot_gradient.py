import numpy as np
from collections import deque
from misc import get_filters
from frame_2D_alg import generic
get_filters(globals())   # imports all filters at once
# --------------------------------------------------------------------------------

def hypot_gradient(blob):  # compute hypot_g, define sub_blobs by incremented filters, extend blob params with hypot flag?
    ''' same functionality as image_to_blobs() in frame_blobs.py'''

    global height, width
    height, width = blob.map.shape
    sub_blob = [0, 0, 0, 0, []]  # explain elements?
    sub_blob_ = []
    seg_ = deque()
    dert__ = blob.dert__

    dert__.mask = ~blob.map
    dert__[:, :, 3] = np.hypot(dert__[:, :, 1], dert__[:, :, 2]) - ave * 2  # ave increased to reflect angle_blobs + eval cost
    sub_blob.append(dert__)  # ref to external map, redundant to dert_ in Ps

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, sub_blob)  # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, sub_blob)
        seg_ = generic.form_seg_(P_, sub_blob)
        
        if sub_blob.open_segs == 0:  # please check, where is open_segs?  
            sub_blob_.append(sub_blob)  # sub_blob is terminated

    y = height - 1   # sub_blob ends, merge segs of last line into their blobs:
    while seg_: 
        generic.form_blob(y, seg_.popleft(), sub_blob)
        if sub_blob.open_segs == 0:  # please check, where is open_segs?  
            sub_blob_.append(sub_blob)  # sub_blob is terminated

    blob.seg_ = sub_blob_  # replace top-level element_ with sub_blob_, appended in form_blob?
    return blob
    # ---------- hypot_gradient() end -----------------------------------------------------------------------------------
