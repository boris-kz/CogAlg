from time import time
# Recursion branches -------------------------------------------------------------
from frame_2D_alg.frame_main.intra_blob.angle_blobs import blob_to_ablobs
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
    intra_blob rdn is reduced by full inclusion: mediated access, also by cross-derivation blob comp?
'''

def eval_blob(blob):  # evaluate blob for comp_angle, incr_rng_comp, incr_der_comp, comp_Py_, orthogonal blob flip

    L, I, G, Dx, Dy, Ly = blob.params
    Ave = ave * L   # whole-blob reprocessing filter, fixed: no if L?
    rdn = 1  # redundant representation counter
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional gradient
        if G > Ave:  # likely edge, angle comp, ablobs def:
            rdn += 1  # or branch-specific cost ratio?
            blob_of_ablobs = blob_to_ablobs(blob)
            sDa = blob_of_ablobs.params[5]

            val_deriv = (G / Ave) * -sDa  # relative G * angle Match
        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match
    val_PP_ = (L + I + G + Dx + Dy) * (L / Ly / Ly)   # last term is elongation: presumed stability (match) of P: cross-section?

    # approximated as box elongation = (x_max - x_min) / (y_max - y_min)?
    # proj P match; abs_Dx and abs_Dy: more accurate but not needed for most blobs?
    # evaluation of three branches of recursion starts with three generic function calls:

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

''' # eval instance: 
    values = val_deriv, val_range, val_PP_
    c, b, a = sorted(values)  # projected values for each branch of recursion:
    
    if a > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
        rdn += 1
        if a is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
        elif a is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
        else:  comp_Py_(0, blob, xD, rdn)  #-> comp_P
        
        if b > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
            rdn += 1
            if b is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
            elif b is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
            else:  comp_Py_(0, blob, xD, rdn)  #-> comp_P
            
            if c > Ave * rdn:  # filter adjusted for redundancy to previously formed representations
                rdn += 1
                if c is val_range: comp_inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
                elif c is val_deriv: comp_inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
                else:  comp_Py_(0, blob, xD, rdn)  #-> comp_P
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

from frame_2D_alg.misc import get_filters
get_filters(globals())          # imports all filters at once

# Main ---------------------------------------------------------------------------
from frame_2D_alg.frame_main import frame_blobs

Y, X = frame_blobs.Y, frame_blobs.X
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
# from DEBUG import draw_blob
# draw_blob('./debug', frame, debug_ablob=1, debug_parts=0, debug_local=0, show=0)
