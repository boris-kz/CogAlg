from time import time
import frame_blobs
from angle_blobs import comp_angle
from misc import draw_blobs
'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm

    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?

    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?
    intra_blob rdn is eliminated by merging blobs, reduced by full inclusion: mediated access?

    dCx = max_x - min_x + 1;  dCy = max_y - min_y + 1
    rC = dCx / dCy  # width / height, vs shift / height: abs(xD) / Ly for oriented blobs only?
    rD = max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)  # lateral variation / vertical variation, for flip and comp_P eval
'''
def blob_eval(blob, dert__):
    comp_angle(blob, dert__)  # angle comp, ablob def; a, da, sda accum in higher-composition reps
    return blob
def intra_blob(frame):  # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_, dert__ = frame
    new_blob_ = []
    for blob in blob_:
        if blob[0]:  # positive g sign
            new_blob_.append(blob_eval(blob, dert__))
    frame = I, G, Dx, Dy, xD, abs_xD, Ly, new_blob_, dert__
    return frame
# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
from misc import get_filters
get_filters(globals())          # imports all filters at once
# --------------------------------------------------------------------------------
# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)
draw_blobs('./debug', frame[7], (frame_blobs.Y, frame_blobs.X), oablob=1, debug=0)
# Check for redundant sda:
print '\nchecking for sda redundancy...\n'
for y, dert_ in enumerate(frame[-1][1:]):
    for x, dert in enumerate(dert_[1:]):
        if len(dert) > 6: print 'redundancy detected at (%d, %d)!\n' %(y + 1, x + 1)