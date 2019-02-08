from time import time
# Recursion branches -------------------------------------------------------------
from frame_2D_alg_current.intra_blob_reprocessing.angle_blobs import blob_to_ablobs
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

def eval_blob(blob, rdn):  # evaluate blob for comp_angle, inc_range comp, inc_deriv comp, comp_Py_

    L, I, G, Dx, Dy, Ly = blob.params
    Ave = ave * L   # whole-blob reprocessing filter
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional gradient
        if G > Ave:  # likely edge, angle comp, ablobs definition

            rdn += 1  # redundant representation counter or branch-specific cost ratio?
            blob_ablobs = blob_to_ablobs(blob)
            val_deriv = (G / Ave) * -blob_ablobs.params[5]  # relative_G * -sDa: angle Match

        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match
    val_PP_ = (L + I + G + Dx + Dy) * (L / Ly / Ly)

    # first term is proj P match; abs_Dx and abs_Dy: more accurate but not needed for most blobs?
    # last term is elongation: indicates above-average match of cross-section P?
    # ~ box elongation = (x_max - x_min) / (y_max - y_min)?
    # plus D_bias: Dx / Dy | Dy / Dx: indicates deviation of P match?

    return [(val_deriv, 0, blob), (val_range, 1, blob), (val_PP_, 2, blob)]  # estimated values per branch


def eval_layer(val_, rdn):  # val_: estimated values of active branches in current layer across recursion tree per blob

    sorted(val_, key= lambda item: item[0])
    new_val_ = []   # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob maps of stronger branches in val_, appended for next val evaluation

    while val_:
        val, typ, blob = val_.pop
        for map in map_:
            olp = blob.map and map   # pseudo code for counting AND between maps, if box overlap?
            rdn += 1 * (olp / blob.L)  # redundancy to previously formed representations

        if val >   ave * blob.L * rdn:
            if typ == 0: blob_sub_blobs = inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
            elif typ==1: blob_sub_blobs = inc_deriv(blob, rdn)  # recursive comp over g_ of incremental derivation
            else:        blob_sub_blobs = comp_Py_(val, 0, blob, rdn)  # -> comp_P

            map_.append( blob.map)
            new_val_ += [intra_blob_recur( new_val_, blob_sub_blobs, rdn)]  # returns estimated recursion values of the next layer

            # eval_blob(sub_blob) in intra_blob_recur() returns (val_deriv, 0, blob), (val_range, 1, blob), (val_PP_, 2, blob)
            # deep angle_blobs is called from eval_blob, from intra_blob_recur()
        else:
            break

    eval_layer(new_val_, rdn+1)  # evaluation of new_val_ for recursion


def inc_range(blob, rdn):
    return blob

def inc_deriv(blob, rdn):
    return blob

def comp_Py_(val_PP_, norm, blob, rdn):
    return blob

def intra_blob_recur(new_val_, blob_sub_blobs, rdn):

    for blob in blob_sub_blobs.sub_blob_:
        new_val_ += [eval_blob(blob, rdn)]  # rdn = 1
    return new_val_

def intra_blob(frame):  # evaluate blobs for comp_angle, inc_range comp, inc_deriv comp, comp_Py_

    for blob in frame.blob_:
        val_ = eval_blob(blob, 2)
        eval_layer(val_, 1)  # calls intra_blob_recur()

    return frame  # frame of 2D patterns, to be outputted to level 2

# ************ MAIN FUNCTIONS END ***************************************************************************************

# ************ PROGRAM BODY *********************************************************************************************

from frame_2D_alg_current.misc import get_filters
get_filters(globals())          # imports all filters at once

# Main ---------------------------------------------------------------------------
from frame_2D_alg_current import frame_blobs

Y, X = frame_blobs.Y, frame_blobs.X
start_time = time()
frame = intra_blob(frame_blobs.frame_of_blobs)
end_time = time() - start_time
print(end_time)

# Rebuild blob -------------------------------------------------------------------
# from DEBUG import draw_blob
# draw_blob('./debug', frame, debug_ablob=1, debug_parts=0, debug_local=0, show=0)