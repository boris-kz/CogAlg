import numpy as np
import numpy.ma as ma
from comp_angle import comp_angle
from comp_deriv import comp_deriv
from comp_range import comp_range
# from comp_P_ import comp_P_
from filters import get_filters
get_filters(globals())  # imports all filters at once
from generic_branch import master_blob

'''
    - this function is under revision
    - intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob
      combined with frame_blobs(), it forms a 2D version of first-level algorithm
      
    - inter_subb() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    - inter_level() will compare between blob levels, where lower composition level is integrated by inter_subb
      match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    - inter_blob() comparison will be second-level 2D algorithm, and a prototype for recursive meta-level algorithm
'''

def intra_blob_root(frame):  # simplified initial branch + eval_layer

    for blob in frame[1]:
        if blob.sign and blob.params[-1] > ave_blob:  # g > var_cost and G > fix_cost of hypot_g: noisy or directional gradient
            master_blob(blob, hypot_g, new_params=False)  # no new params, this branch only redefines g as hypot(dx, dy)

            if blob.params[-1] > ave_blob * 2:  # G > fixed costs of comp_angle
                rdn = 1
                val_ = []
                for sub_blob in blob.sub_blob_:
                    if sub_blob.sign and sub_blob.params[-1] > ave_blob * 2:  # > variable and fixed costs of comp_angle

                        master_blob(sub_blob, comp_angle)
                        Ly, L, Y, X, Dert_tree = sub_blob.params
                        I, Dx, Dy, G, dert_ = Dert_tree[len(Dert_tree) - 1]
                        A, Dax, Day, Ga, adert_ = Dert_tree[len(Dert_tree)]
                        
                        # estimated values of next-layer recursion per sub_blob:

                        val_angle = Ga  # value of comp_ga -> gga, eval comp_angle(dax, day), next layer / aga_blob
                        val_deriv = ((G + ave * L) / ave * L) * -Ga  # relative G * -Ga: angle match, likely edge
                        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

                        val_ += [(val_angle, 0, sub_blob), (val_deriv, 1, sub_blob), (val_range, 2, sub_blob)]
                if val_:
                    eval_layer(val_, rdn)

    return frame  # frame of 2D patterns is output to level 2


def branch(blob, typ):  # compute branch, evaluate next branches: comp_angle, comp_range, comp_deriv, comp_angle_deriv
    vals = []

    if typ == 0:   master_blob(blob, comp_deriv, 1)  # comp over ga_: last 0|1 selects a_dert vs. i_dert
    elif typ == 1: master_blob(blob, comp_deriv, 0)  # recursive comp over g_ with incremental derivation
    else:          master_blob(blob, comp_range, 0)  # recursive comp over i_ with incremental distance

    # arg blob contains Dert @ last index, and return is master_blob that contains higher Dert @ last index + 1:

    if blob.params[-1] > ave_blob * 2:  # G = blob.params[-1]
        master_blob(blob, comp_angle)
        Ly, L, Y, X, Dert_tree = blob.params
        I, Dx, Dy, G, dert_ = Dert_tree[ len( Dert_tree) - 1]
        A, Dax, Day, Ga, adert_ = Dert_tree[ len( Dert_tree)]

        # Dert_tree node = root_Dert, ?ang_Dert? ?ang_der_Dert, ?der_Dert, ?rng_Dert
        # estimated values of next-layer branches per blob:

        val_angle = Ga  # value of comp_ga -> gga, eval comp_angle(dax, day), next layer / aga_blob
        val_deriv = ((G + ave * L) / ave * L) * -Ga  # relative G * -Ga: angle match, likely edge
        val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

        vals = [(val_angle, 0, blob), (val_deriv, 1, blob), (val_range, 2, blob)]  # branch values per blob
    return vals


def eval_layer(val_, rdn):  # val_: estimated values of active branches in current layer across recursion tree per blob

    val_ = sorted(val_, key=lambda val: val[0])
    sub_val_ = []  # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob boxes + maps of stronger branches in val_, appended for next (lower) val evaluation

    while val_:
        val, typ, blob = val_.pop()
        for box, map in map_:
            olp = overlap(blob, box, map)
            rdn += 1 * (olp / blob.params[1])  # rdn += 1 * (olp / G): redundancy to previous + stronger overlapping blobs, * branch cost ratio?

        if val > ave * blob.params[1] * rdn + ave_blob:  # val > ave * G * rdn + fix_cost: extend master blob syntax: += branch syntax
            for sub_blob in blob.sub_blob_:  # sub_blobs are angle blobs

                sub_vals = branch(sub_blob, typ)  # branch-specific recursion step
                if sub_vals:  # not empty []
                    sub_val_ += sub_vals

            map_.append((blob.box, blob.map))
        else:
            break

    if sub_val_:  # not empty []
        rdn += 1  # ablob redundancy to default gblob, or rdn += 2 for additional cost of angle calc?
        eval_layer(sub_val_, rdn)  # evaluation of each sub_val for recursion

    ''' 
        comp_P_(val, 0, blob, rdn) -> (val_PP_, 4, blob), (val_aPP_, 5, blob),
        val_PP_ = (L + I + G) * (L / Ly / Ly) * (Dy / Dx):
        1st term: proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant hyp_g_P calc, comp, no indiv comp: rdn
        2nd term: elongation: >ave Pm? ~ box elong: (x_max - x_min) / (y_max - y_min)?
        3rd term: dimensional variation bias
        g and ga are dderived, blob selected for min_g
        val-= sub_blob and branch switch cost: added map?  only after g,a calc: no rough g comp?
    '''
    # ---------- eval_layer() end ---------------------------------------------------------------------------------------


def hypot_g(blob):  # redefine master blob by reduced g and increased ave * 2: variable costs of comp_angle

    mask = ~blob.map[:, :, np.newaxis].repeat(4, axis=2)  # stack map 4 times to fit the shape of dert__: (width, height, number of params)
    blob.new_dert__[0] = ma.array(blob.dert__, mask=mask)  # initialize dert__ with mask for selective comp

    # redefine g = hypot(dx, dy), ave * 2 assuming that added cost of angle calc = cost of hypot_g calc
    blob.new_dert__[0][:, :, 3] = np.hypot(blob.new_dert__[0][:, :, 1], blob.new_dert__[0][:, :, 2]) - ave * 2

    return 1  # comp rng

# ---------- hypot_g() end -----------------------------------------------------------------------------------

def overlap(blob, box, map):  # returns number of overlapping pixels between blob.map and map

    y0, yn, x0, xn = blob.box
    _y0, _yn, _x0, _xn = box

    olp_y0 = max(y0, _y0)
    olp_yn = min(yn, _yn)
    if olp_yn - olp_y0 <= 0:  # no overlapping y coordinate span
        return 0
    olp_x0 = max(x0, _x0)
    olp_xn = min(xn, _xn)
    if olp_xn - olp_x0 <= 0:  # no overlapping x coordinate span
        return 0
    
    # master_blob coordinates olp_y0, olp_yn, olp_x0, olp_xn are converted to local coordinates before slicing:

    map1 = box.map[(olp_y0 - y0):(olp_yn - y0), (olp_x0 - x0):(olp_xn - x0)]
    map2 = map[(olp_y0 - _y0):(olp_yn - _y0), (olp_x0 - _x0):(olp_xn - _x0)]

    olp = np.logical_and(map1, map2).sum()  # compute number of overlapping pixels

    return olp
    # ---------- overlap() end ------------------------------------------------------------------------------------------
