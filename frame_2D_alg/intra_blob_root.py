import numpy as np
import numpy.ma as ma
from collections import deque
from frame_2D_alg import generic
from frame_2D_alg.angle_blobs import blob_to_ablobs
from frame_2D_alg.comp_inc_deriv import inc_deriv
from frame_2D_alg.comp_inc_range import inc_range
# from comp_P_ import comp_P_

from frame_2D_alg.filters import get_filters
get_filters(globals()) # imports all filters at once
'''
    - intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob
      combined with frame_blobs(), it forms a 2D version of first-level algorithm
      
    - inter_subb() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    - inter_level() will compare between blob levels, where lower composition level is integrated by inter_subb
      match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    - inter_blob() comparison will be second-level 2D algorithm, and a prototype for recursive meta-level algorithm
'''

def intra_blob(frame, rdn):  # root function

    for blob in frame.blob_:
        eval_layer( eval_blob(blob, rdn), rdn)  # eval_blob returns val_
        # for debug:
        # if blob.sign:
        #    blob_to_ablobs(blob)
        #    inc_range(blob)
        #    inc_deriv(blob)
    return frame  # frame of 2D patterns is output to level 2


def eval_blob(blob, rdn):  # evaluate blob for comp_angle, comp_inc_range, comp_inc_deriv, comp_P_

    Ly, L, Y, X, I, Dy, Dx, G = blob.params
    val_deriv, val_range = 0, 0

    if blob.sign:  # positive gblob: area of noisy or directional (edge) gradient
        if G > ave_blob:   # fixed cost of hypot_g() per blob
            hypot_g(blob)  # g is more precisely estimated as hypot(dx, dy), replaces blob with blob + sub_blob_

            if blob.G > ave_blob * 2:  # fixed cost of blob_to_ablobs() per blob
                rdn += 1  # redundant representation counter: stronger overlapping blobs, or branch-specific cost ratio?
                blob = blob_to_ablobs(blob, rdn)
                if blob.Ga > ave_blob * 4:  # fixed cost of angle comp recursion within ablobs, per blob
                    blob = intra_blob(blob, rdn)  # intra recursion per angle and ga, vs pixel and g

            val_deriv = ((G + ave*L) / ave*L) * -blob.Ga  # relative G * -Ga: angle match, likely edge
            val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

    # val_PP_ = (L + I + G) * (L / Ly / Ly) * (Dy / Dx)
    # 1st term: proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant hyp_g_P calc, comp, no indiv comp: rdn
    # 2nd term: elongation: >ave Pm? ~ box elong: (x_max - x_min) / (y_max - y_min)?
    # 3rd term: dimensional variation bias

    return [(val_deriv, 0, blob), (val_range, 1, blob)]  # + (val_PP_, 2, blob), estimated values per branch


def eval_layer(val_, rdn):  # val_: estimated values of active branches in current layer across recursion tree per blob

    val_ = sorted(val_, key=lambda val: val[0])
    sub_val_ = []  # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob boxes + maps of stronger branches in val_, appended for next val evaluation

    while val_:
        val, typ, blob = val_.pop()
        for box, map in map_:
            olp = overlap(blob, box, map)
            rdn += 1 * (olp / blob.L())  # redundancy to previously formed representations

        if val > ave * blob.params(1) * rdn:
            if typ==0: blob = inc_range(blob)  # recursive comp over p_ of incremental distance
            else:      blob = inc_deriv(blob)  # recursive comp over g_ of incremental derivation
                       # dderived, blob selected for min_g

            # else: blob_sub_blobs = comp_P_(val, 0, blob, rdn)  # -> comp_P
            # val-= sub_blob and branch switch cost: added map?  only after g,a calc: no rough g comp?

            map_.append((blob.box, blob.map))
            for sub_blob in blob.sub_blob_:
                sub_val_ += eval_blob(sub_blob, rdn)  # returns estimated recursion values of the next layer:
                # [(val_deriv, 0, blob), (val_range, 1, blob), (val_PP_, 2, blob)] per sub_blob, may include deep angle_blobs
        else:
            break

    if sub_val_:
        rdn += 1
        eval_layer(sub_val_, rdn)  # evaluation of sub_val_ for recursion

    # ---------- eval_layer() end ---------------------------------------------------------------------------------------

def hypot_g(blob):  # redefine blob and sub_blobs by reduced g and increased ave + ave_blob: var + fixed costs of angle_blobs() and eval

    global height, width
    height, width = blob.map.shape
    seg_ = deque()
    blob.new_dert__ = ma.array(blob.dert__, mask=~blob.map)
    # redefine g = hypot(dx, dy):
    blob.new_dert__[:, :, 3] = np.hypot(blob.new_dert__[:, :, 1], blob.new_dert__[:, :, 2]) - ave * 2  # incr filter = cost of angle calc

    for y in range(1, height - 1):
        P_ = generic.form_P_(y, blob)         # horizontal clustering
        P_ = generic.scan_P_(P_, seg_, blob)  # vertical clustering
        seg_ = generic.form_seg_(P_, blob)
    while seg_: generic.form_blob(seg_.popleft(), blob)

    # ---------- hypot_g() end -----------------------------------------------------------------------------------

def overlap(blob, box, map):    # returns number of overlapping pixels between blob.map and map
    y0, yn, x0, xn = blob.box
    _y0, _yn, _x0, _xn = box

    olp_y0 = max(y0, _y0)
    olp_yn = min(yn, _yn)
    if olp_yn - olp_y0 <= 0:    # no overlapping y coordinate span
        return 0
    olp_x0 = max(x0, _x0)
    olp_xn = min(xn, _xn)
    if olp_xn - olp_x0 <= 0:    # no overlapping x coordinate span
        return 0
    # master_blob coordinates olp_y0, olp_yn, olp_x0, olp_xn are converted to local coordinates before slicing:

    map1 = box.map[(olp_y0 - y0):(olp_yn - y0), (olp_x0 - x0):(olp_xn - x0)]
    map2 = map[(olp_y0 - _y0):(olp_yn - _y0), (olp_x0 - _x0):(olp_xn - _x0)]

    olp = np.logical_and(map1, map2).sum()  # compute number of overlapping pixels
    return olp

    # ---------- overlap() end ------------------------------------------------------------------------------------------

