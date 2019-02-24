from time import time
import numpy as np
# Recursion branches -------------------------------------------------------------
from frame_2D_alg.angle_blobs import blob_to_ablobs
from frame_2D_alg.comp_inc_deriv import inc_deriv
from frame_2D_alg.comp_inc_range import bilateral, inc_range
# from comp_P_ import comp_P_
'''
    - intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob
      combined with frame_blobs(), it forms a 2D version of first-level algorithm
      
    - inter_subb() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    - inter_level() will compare between blob levels, where lower composition level is integrated by inter_subb
      match between levels'edges may form composite blob, axis comp if margin sub_blobs over super blobs?
       
    - inter_blob() comp will be second-level 2D algorithm, and a prototype for recursive meta-level algorithm
'''

def eval_blob(blob):  # evaluate blob for comp_angle, comp_inc_range, comp_inc_deriv, comp_P_

    global rdn  # redundant representation counter, or branch-specific cost ratio?
    L, I, G = blob.params[:3]  # Ly = blob.orientation_params[2]
    Ave = ave * L  # whole-blob reprocessing filter
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional (edge) gradient
        if G > Ave:  # + fixed costs of hypot_g, angle_blobs, evaluation?
            rdn += 1
            blob.dert__[blob.map, 1] = np.hypot(blob.dert__[blob.map, 2], blob.dert__[blob.map, 3])
            # max g is more precisely estimated as hypot(dx, dy)
            blob_ablobs = blob_to_ablobs(blob)
            if blob.Ga > Ave:
                blob_ablobs = intra_blob(blob_ablobs)  # eval for recursion within ablobs

            val_deriv = (G / Ave) * -blob_ablobs.params[5]  # rG * -Ga: angle match, likely edge
            val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

    # val_PP_ = (L + I + G) * (L / Ly / Ly) * (Dy / Dx)
    # 1st term: proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant hyp_g_P calc, comp, no indiv comp: rdn
    # 2nd term: elongation: >ave Pm? ~ box elong: (x_max - x_min) / (y_max - y_min)?
    # 3rd term: dimensional variation bias

    return [(val_deriv, 0, blob), (val_range, 1, blob)]  # + (val_PP_, 2, blob), estimated values per branch


def eval_layer(val_):  # val_: estimated values of active branches in current layer across recursion tree per blob

    global rdn
    val_ = sorted(val_, key=lambda val: val[0])
    sub_val_ = []  # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob maps of stronger branches in val_, appended for next val evaluation

    while val_:
        val, typ, blob = val_.pop()
        for map in map_:
            olp = np.sum( np.logical_and( blob.map, map))  # if box overlap?
            rdn += 1 * (olp / blob.L())   # redundancy to previously formed representations

        if val > ave * blob.L() * rdn:
            if typ == 0:
                if blob.ncomp == 1:  # add ncomp param init=1, incr per bilateral | inc_range
                    blob_sub_blobs = bilateral(blob)  # recursive comp over p_ of incremental distance
                else:
                    blob_sub_blobs = inc_range(blob)  # recursive comp over p_ of incremental distance
            else:
                blob_sub_blobs = inc_deriv(blob)  # recursive comp over g_ of incremental derivation
                # dderived, but min_g is not necessary because +gblob already selected for it

            # else: blob_sub_blobs = comp_P_(val, 0, blob, rdn)  # -> comp_P
            # val-= sub_blob and branch switch cost: added map?  only after g,a calc: no rough g comp?

            map_.append(blob.map)
            for blob in blob_sub_blobs.blob_:
                sub_val_ += eval_blob(blob)  # returns estimated recursion values of the next layer:
                # [(val_deriv, 0, blob), (val_range, 1, blob), (val_PP_, 2, blob)] per sub_blob, may include deeper angle_blobs?
        else:
            break

    if sub_val_:
        rdn += 1
        eval_layer(sub_val_)  # evaluation of sub_val_ for recursion


def intra_blob(frame, redundancy=0.0):  # evaluate blobs for comp_angle, inc_range comp, inc_deriv comp, comp_P_

    global rdn
    for blob in frame.blob_:
        rdn = redundancy
        eval_layer( eval_blob( blob))  # eval_blob returns val_

        # for debugging:
        # if blob.sign:
        #     blob_to_ablobs(blob)
        #     inc_range(blob)
        #     inc_deriv(blob)
    return frame  # frame of 2D patterns, to be outputted to level 2

# ************ PROGRAM BODY *********************************************************************************************

from frame_2D_alg.misc import get_filters
get_filters(globals())  # imports all filters at once

# Main ---------------------------------------------------------------------------
from frame_2D_alg import frame_blobs

Y, X = frame_blobs.Y, frame_blobs.X
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------

from DEBUG import draw_blob
draw_blob('../debug', frame, typ=0, debug_parts=0, debug_local=0, show=0)