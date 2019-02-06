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

def eval_blob(blob):  # evaluate blob for comp_angle, inc_range comp, inc_deriv comp, comp_Py_

    L, I, G, Dx, Dy, Ly = blob.params
    Ave = ave * L   # whole-blob reprocessing filter
    rdn = 1  # redundant representation counter
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional gradient
        if G > Ave:  # likely edge, angle comp, ablobs definition
            rdn += 1  # or branch-specific cost ratio?
            blob_of_ablobs = blob_to_ablobs(blob)
            sDa = blob_of_ablobs.params[5]

            val_deriv = (G / Ave) * -sDa  # relative G * angle Match
        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match
    val_PP_ = (L + I + G + Dx + Dy) * (L / Ly / Ly)

    # first term is proj P match; abs_Dx and abs_Dy: more accurate but not needed for most blobs?
    # last term is elongation: indicates above-average match of cross-section P?
    # ~ box elongation = (x_max - x_min) / (y_max - y_min)?
    # plus D_bias: Dx / Dy | Dy / Dx: indicates deviation of P match?

    return Ave, ((val_deriv, blob.map), (val_range, blob.map), (val_PP_, blob.map))  # estimated value per branch


def eval_layer(val_, blob, rdn, Ave):  # evaluation of val_: one layer of recursion tree, unfinished
    new_val_ = []
    eval_ = sorted(val_, key= lambda item: item[0])

    while eval_:
        val, typ, map_ = eval_.pop
        for map in map_:
            olp = blob.map and map   # pseudo code for sum of AND between maps
            rdn += 1 * (olp / blob.L)  # redundancy to previously formed representations

        if val > Ave * rdn:
            if typ == 0:   inc_range(blob, rdn)  # recursive comp over p_ of incremental distance, also diagonal?
            elif typ == 1: inc_deriv(blob, rdn)  # recursive comp over d_ of incremental derivation
            else:          comp_Py_(val, 0, blob, rdn)  # -> comp_P

            Ave, ((val_deriv, imap), (val_range, imap), (val_PP_, imap)) = eval_blob(blob)
            map_.append(imap)
            new_val_ += [(val_deriv, 0, map_), (val_range, 1, map_), (val_PP_, 2, map_)]
            # merge three estimated branch values per blob into deeper layer of recursion tree
        else:
            break

    eval_layer(new_val_, blob, rdn, Ave)  # evaluation of new_val_ for recursion

def inc_range(blob, rdn):
    return -1, inc_range, [blob]

def inc_deriv(blob, rdn):
    return -1, inc_deriv, [blob]

def comp_Py_(val_PP_, norm, blob, rdn):
    return -1, comp_Py_, [0, 0, blob]

def intra_blob(frame):  # evaluate blobs for comp_angle, inc_range comp, inc_deriv comp, comp_Py_

    for blob in frame.blob_:
        Ave, val_ = eval_blob(blob)
        eval_layer(val_, blob, 1, Ave)  # recursive

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