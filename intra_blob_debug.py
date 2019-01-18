from collections import deque
import math as math
from time import time
import frame_blobs
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
def comp_angle(blob, dert__):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' - Sort list of segments (root_) based on their top line P's coordinate (segment's min_y)    <---------------------------------------|
        - Iterate through each line in the blob (from blob's min_y to blob's max_y):                                                        |
            + Have every segment that contains current-line P in a list (runningSegment_). This action is simplified by sorting step above -|
            + Extract current-line slice of the blob - or the list of every P of this line (P_)
            + Have every out-of-bound segment removed from list (runningSegment_)
            + Perform angle computing, comparison and clustering in every dert in P_ '''
    root_ = blob[3]
    blob[3] = sorted(root_, key=lambda segment: segment[1][2])  # sorted by min_y
    ablob_ = []
    global y
    y = blob[1][2]      # min_y of this blob
    runningSegment_ = []
    haP_ = []
    segmentIndex = 0    # iterator
    while segmentIndex < len(root_):
        P_ = []
        while root_[segmentIndex][1][2] == y:
            runningSegment_.append([root_[segmentIndex], 0])    # runningSegment consists of segments that contains y-line P and that P's index
        for i, (segment, PIndex) in enumerate(runningSegment_): # for every segment that contains y-line P
            P_.append(segment[3][PIndex][0])    # P = Py_[PIndex][0] = segment[3][PIndex][0]
            if y = segment[1][3]:               # if y has reached segment's bottom
                runningSegment_.pop(i)          # remove from list
            else:
                runningSegment_[i] += 1         # index point to next-line P
        # actual comp_angle:


        # buffers for next line
        haP_ = aP_

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
ave = 15  # g value that coincides with average match: gP filter
div_ave = 1023  # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000  # cost of form_P and deeper?
ave_rate = 0.25  # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
dim = 2  # number of dimensions
rng = 2  # number of pixels compared to each pixel in four directions
min_coord = rng * 2 - 1  # min x and y for form_P input: ders2 from comp over rng*2 (bidirectional: before and after pixel p)
degree = 128 / math.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15
# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)