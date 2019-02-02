from time import time
# Recursion branches -------------------------------------------------------------
from angle_blobs import blob_to_ablobs
# from inc_deriv import inc_deriv
# from comp_Py_ import comp_Py_

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

def eval_blob(blob):  # evaluate blob for comp_angle, incr_rng_comp, incr_der_comp, comp_Py_, orthogonal blob flip

    L, I, G, Dx, Dy = blob.params
    Ave = ave * L   # whole-blob reprocessing filter, fixed: no if L?
    rdn = 1  # redundant representation counter
    val_deriv, val_range = 0, 0

    if blob.sign:     # positive gblob: area of noisy or directional gradient
        if G > Ave:   # likely edge, angle comp, ablobs def:
            rdn += 1  # or greater?
            blob_of_ablobs = blob_to_ablobs(blob)
            sDa = blob_of_ablobs.params[5]
            val_deriv = G * -sDa  # -sDa represents summed proximate angle match
        val_range = G  # non-directional variation, thus likely d reversal and match among distant pixels
    val_PP_ = L + I + G + Dx + Dy  # max P match -> PP_, also abs_Dx, abs_Dy: more accurate but not needed for most blobs?

    # Three branches of recursion start with three generic function calls:
    values      = [val_range, val_deriv, val_PP_]   # projected values of three branches of recursion
    branches    = [inc_range, inc_deriv, comp_Py_]   # functions of each branch
    arguments   = [[blob], [blob], [val_PP_, 0, blob]]   # arguments of each branch
    eval_queue  = sorted(zip(values, branches, arguments), key= lambda item: item[0], reverse=True)  # sort by value
    recursion(eval_queue, Ave, rdn)

def recursion(eval_queue, Ave, rdn):
    ''' evaluation of recursion branches, result is evaluated for insertion in eval_queue, which determines next step of recursion '''

    val, branch, args = eval_queue.pop(0)
    if val > Ave * rdn:
        new_val, new_branch, new_args = branch(*args, rdn=rdn)  # insert new branch into eval_queue, ordered by value

        if new_val > 0:
            eval_queue = sorted(eval_queue.append((new_val, new_branch, new_args)), key= lambda item: item[0], reverse=True)
        if eval_queue:
            recursion(eval_queue, Ave, rdn+1)

''' non-recursive eval_queue eval: 
 
    values = val_deriv, val_range, val_PP_
    c, b, a = sorted(values)
    # three instances of evaluation for three branches of recursion:
    if a > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
        rdn += 1
        if a is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
        elif a is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
        else:
            if val_PP_ * ((x_end - x_start) / (max_y - min_y)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                flip(blob)  # vertical blob rescan -> comp_Px_
            comp_Py_(0, blob, xD, rdn)  #-> comp_P
        if b > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
            rdn += 1
            if b is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
            elif b is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
            else:
                if val_PP_ * ((x_end - x_start) / (max_y - min_y)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                    flip(blob)  # vertical blob rescan -> comp_Px_
                comp_Py_(0, blob, xD, rdn)  #-> comp_P
            if c > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
                rdn += 1
                if c is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
                elif c is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
                else:
                    if val_PP_ * ((x_end - x_start) / (max_y - min_y)) * (max(abs_Dx, abs_Dy) / min(abs_Dx, abs_Dy)) > flip_ave:
                        flip(blob)  # vertical blob rescan -> comp_Px_
                    comp_Py_(0, blob, xD, rdn)  #-> comp_P
'''
def inc_range(blob, rdn):
    return -1, inc_range, [blob]

def inc_deriv(blob, rdn):
    return -1, inc_deriv, [blob]

def comp_Py_(val_PP_, norm, blob, rdn):
    return -1, comp_Py_, [0, 0, blob]


def intra_blob(frame):   # evaluate blobs for comp_angle, inc_range comp, inc_deriv comp, comp_Py_

    for blob in frame.blob_:
        eval_blob(blob)
    return frame  # frame of 2D patterns, to be outputted to level 2

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

from misc import get_filters
get_filters(globals())          # imports all filters at once

# Main ---------------------------------------------------------------------------
import frame_blobs
Y, X = frame_blobs.Y, frame_blobs.X
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
# from DEBUG import draw_blob
# draw_blob('./debug', frame, debug_ablob=1, debug_parts=0, debug_local=0, show=0)