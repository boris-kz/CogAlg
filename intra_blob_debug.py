from collections import deque
import math as math
from time import time
import frame_blobs
from angle_blobs import comp_angle
from comp_Py_ import comp_Py_

'''
    intra_blob() is an extension to frame_blobs, it performs evaluation for comp_P and recursive frame_blobs within each blob.
    Currently it's mostly a draft, combined with frame_blobs it will form a 2D version of first-level algorithm
    inter_blob() will be second-level 2D algorithm, and a prototype for meta-level algorithm
    
    colors will be defined as color / sum-of-colors, color Ps are defined within sum_Ps: reflection object?
    relative colors may match across reflecting objects, forming color | lighting objects?     
    comp between color patterns within an object: segmentation?
    
    inter_olp_blob: scan alt_typ_ ) alt_color, rolp * mL > ave * max_L?   
    intra_blob rdn is eliminated by merging blobs, reduced by full inclusion: mediated access?
'''

def eval_blob(blob, dert__):  # evaluate blob for comp_angle, incr_rng_comp, incr_der_comp, comp_Py_, orthogonal blob flip

    s, [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, I, G, Dx, Dy, abs_Dx, abs_Dy], root_ = blob
    Ave = ave * L   # whole-blob reprocessing filter, fixed: no if L?
    rdn = 1  # redundant representation counter
    val_deriv, val_range = 0, 0

    if s:  # positive blob, primary orientation match eval: noisy or directional gradient
        if G > Ave:   # likely edge, ave d_angle = ave g?
            rdn += 1  # or greater?
            comp_angle(blob, dert__)  # angle comparison, ablob definition; A, sDa accumulation in aP, aseg, ablob, blob
            sDa = blob[2][7]

            val_deriv = G * -sDa  # -sDa indicates proximate angle match -> directional d match, dderived?
        val_range = G  # G without angle is not directional, thus likely d reversal and match among distant pixels
    val_PP_ = L + I + G + Dx + Dy  # max P match -> PP_, also abs_Dx, abs_Dy: more accurate but not needed for most blobs?

    # Three branches of recursion start with three generic function calls:

    values      = [val_range, val_deriv, val_PP_]   # projected values of three branches of recursion
    branches    = [inc_range, inc_deriv, comp_Py_]   # functions of each branch
    arguments   = [[blob, dert__], [blob, dert__], [val_PP_, 0, blob, xD]]   # arguments of each branch
    eval_queue  = sorted(zip(values, branches, arguments), key= lambda item: item[0], reverse=True)  # sort by value
    recursion(eval_queue, Ave, rdn)
    return blob

def recursion(eval_queue, Ave, rdn):  # initial evaluation of recursion branches

    val, branch, arg = eval_queue.pop(0)
    if val > Ave * rdn:
        branch( *arg, rdn=rdn )   # specific recursive function call
        if eval_queue:
            deep_recursion(eval_queue, Ave, rdn+1)  # or rdn += rel_rdn: cost per dert per branch / cost per comp pixel?

def deep_recursion(eval_queue, Ave, rdn):
    # result of evaluation is also evaluated for insertion in eval_queue, which determines next step of recursion

    val, branch, arg = eval_queue.pop(0)
    if val > Ave * rdn:
        new_val, new_branch, new_arg = branch(*arg, rdn=rdn)  # insert new branch into eval_queue, ordered by value
        new_eval_queue = []

        while eval_queue and eval_queue[0][0] > new_val:
            new_eval_queue.append(eval_queue.pop(0))
        new_eval_queue.append((new_val, new_branch, new_arg))
        while eval_queue:
            new_eval_queue.append(eval_queue.pop(0))
        eval_queue = new_eval_queue
        if eval_queue:
            deep_recursion(eval_queue, Ave, rdn+1)

'''
    values = val_deriv, val_range, val_PP_
    c, b, a = sorted(values)
    # three instances of evaluation for three branches of recursion:

    if a > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
        rdn += 1
        if a is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
        elif a is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
        else:
            if val_PP_ * ((max_x - min_x + 1) / (max_y - min_y + 1)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                flip(blob)  # vertical blob rescan -> comp_Px_
            comp_Py_(0, blob, xD, rdn)  #-> comp_P

        if b > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
            rdn += 1
            if b is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
            elif b is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
            else:
                if val_PP_ * ((max_x - min_x + 1) / (max_y - min_y + 1)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                    flip(blob)  # vertical blob rescan -> comp_Px_
                comp_Py_(0, blob, xD, rdn)  #-> comp_P

            if c > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
                rdn += 1
                if c is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
                elif c is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
                else:
                    if val_PP_ * ((max_x - min_x + 1) / (max_y - min_y + 1)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                        flip(blob)  # vertical blob rescan -> comp_Px_
                    comp_Py_(0, blob, xD, rdn)  #-> comp_P
'''

def inc_range(blob, dert__, rdn):  # frame_blobs recursion if G
    return 0, inc_range, [blob]

def inc_deriv(blob, dert__, rdn):  # frame_blobs recursion if Dx + Dy: separately, or abs_Dx + abs_Dy: directional, but for both?
    return 0, inc_deriv, [blob]

def intra_blob(frame):   # evaluate blobs for orthogonal flip, incr_rng_comp, incr_der_comp, comp_P
    I, G, Dx, Dy, xD, abs_xD, Ly, blob_, dert__ = frame

    new_blob_ = []
    for blob in blob_:
        new_blob_.append(eval_blob(blob, dert__))

    frame = I, G, Dx, Dy, xD, abs_xD, Ly, new_blob_, dert__
    return frame  # frame of 2D patterns, to be outputted to level 2

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************
# Pattern filters ----------------------------------------------------------------
# eventually updated by higher-level feedback, initialized here as constants:
from misc import get_filters
get_filters(globals())          # imports all filters at once
from misc import draw_blobs

# Main ---------------------------------------------------------------------------
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)
draw_blobs('./debug', frame[7], (frame_blobs.Y, frame_blobs.X), out_ablob=1, debug=1)
