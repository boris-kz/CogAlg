from time import time
import numpy as np
# Recursion branches -------------------------------------------------------------
from frame_2D_alg.compare_derts import inc_range
# from comp_P_ import comp_P_

'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm

    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?

    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?   
    intra_blob rdn is reduced by full inclusion: mediated access, also by cross-derivation blob comp?
'''


def eval_blob(blob):  # evaluate blob for comp_angle, inc_range comp, inc_deriv comp, comp_P_

    global rdn
    L, I, G = blob.params[:3]
    Ly = blob.orientation_params[2]
    Ave = ave * L  # whole-blob reprocessing filter
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional gradient
        if G > Ave:  # likely edge, angle comp, ablobs definition

            rdn += 1  # redundant representation counter, or branch-specific cost ratio?
            blob.dert__[blob.map, 1] = np.hypot(blob.dert__[blob.map, 2], blob.dert__[blob.map, 3])
            blob_ablobs = blob_to_ablobs(blob)
            blob_ablobs = intra_blob(blob_ablobs)
            val_deriv = (G / Ave) * -blob_ablobs.params[5]  # relative_G * -sDa: angle Match

        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match
    val_PP_ = (L + I + G) * (L / Ly / Ly)

    # first term is proj P match; + abs_Dx and abs_Dy: more accurate but not needed for most blobs?
    # last term is elongation: above-average P match? ~ box elongation: (x_max - x_min) / (y_max - y_min)?

    # * D bias: A deviation from vertical: y_dev = A - 128*L, if y_dev > 63:
    # A adjust -> ave / each sum?

    # vs Dx = (Dx * hyp + Dy / hyp) / 2 / hyp  # est D over ver_L, Ders summed in ver / lat ratio
    #    Dy = (Dy / hyp - Dx * hyp) / 2 / hyp  # for flip and comp_P_ eval only, no comp?

    # return [(val_deriv, 0, blob), (val_range, 1, blob), (val_PP_, 2, blob)]  # estimated values per branch
    return [(val_deriv, 0, blob), (val_range, 1, blob)]  # estimated values per branch


def eval_layer(val_):  # val_: estimated values of active branches in current layer across recursion tree per blob

    global rdn
    val_ = sorted(val_, key=lambda val: val[0])
    sub_val_ = []  # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob maps of stronger branches in val_, appended for next val evaluation

    while val_:
        val, typ, blob = val_.pop()
        for map in map_:
            olp = np.sum(np.logical_and(blob.map, map))
            rdn += 1 * (olp / blob.L())           # redundancy to previously formed representations

        if val > ave * blob.L() * rdn:
            if typ == 0:
                blob_sub_blobs = inc_range(blob)  # recursive comp over p_ of incremental distance, also diagonal?
            else:
                blob_sub_blobs = inc_deriv(blob)  # recursive comp over g_ of incremental derivation
            # else:
            #     blob_sub_blobs = comp_P_(val, 0, blob, rdn)  # -> comp_P

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
        eval_layer(eval_blob(blob))  # calls eval_sub_blob()

        # for debugging:
        # if blob.sign:
        #     blob_to_ablobs(blob)
        #     inc_range(blob)
        #     inc_deriv(blob)
    return frame  # frame of 2D patterns, to be outputted to level 2


# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

from old.frame_2D_alg_classes.misc import get_filters

get_filters(globals())  # imports all filters at once

# Main ---------------------------------------------------------------------------
from old.frame_2D_alg_classes import frame_blobs

Y, X = frame_blobs.Y, frame_blobs.X
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
from DEBUG import draw_blob
draw_blob('../debug', frame, typ=0, debug_parts=0, debug_local=0, show=0)