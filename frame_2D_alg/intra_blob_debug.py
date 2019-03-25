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
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    frame_blobs + intra_blob forms a 2D version of 1st-level algorithm.
    
    recursive intra_blob' eval_layer' branch() calls add new dert to derts, Dert to Derts, sub_blobs ) low_layers to blob,
    where Dert params are summed params of all positive blobs per structure, with optional sub_blobs ) low_layers:
    
    blob =  
        typ,  # typ 0: primary | angle_g_blob, typ 1: angle_ga_blob, typ 2: gg_blob, typ 3: g_range_blob, formed / eval:  
              # ga_sign? 
              #    Ga? typ_1_blob = comp_deriv(a) 
              #    : G * -Ga? 
              #        typ_2_blob = comp_deriv(i) 
              #        : G - (G * -Ga)? 
              #            typ_3_blob = comp_range(i)
              
        sign, Y, X, Ly, L,  # these are common for all Derts, higher-Derts sign is always positive, else no branching 
        
        Derts = 
            [ Dert = I, Dx, Dy, G ],  # one Dert per current and higher layers (higher if root blob is a sub_blob)
        derts_ = 
            [ derts = 
                [ dert = i, dx, dy, g ] ],  # one dert per current and higher layers
        sub_blobs = 
            ( Derts = [(Ly, L, I, Dx, Dy, G)]  # one Dert per positive sub_blobs and their higher layers, selective L, Ly 
              sub_blob_),   # sub_blob structure is same as in root blob 
        low_layers =   
            ( Derts = [(Ly, L, I, Dx, Dy, G)]  # one Dert per positive sub_blobs of referred layer and their higher layers
              low_layer_)   # array of layers across derivation tree per root blob 
              
        # sub_blobs and low_layers Derts params: same as in their root_blob Derts, but summed from positive sub_blobs only. 
        # Derts of Lower layers are not represented in higher layers.
    
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D algorithm: a prototype for recursive meta-level
'''

def intra_blob_root(frame):  # simplified initial branch() and eval_layer() call

    for blob in frame.blob_:
        if blob.sign and blob.Derts[-1][-1] > ave_blob: # g > ave: variable cost, and G > fixed cost of hypot_g:
            # area of noisy or directional gradient
            master_blob(blob, hypot_g, add_dert=False)  # redefines g as hypot(dx, dy)

            if blob.Derts[-1][-1] > ave_blob * 2:  # G > fixed costs of comp_angle
                val_ = []
                for sub_blob in blob.sub_blobs[1]:  # in sub_blob_

                    if sub_blob.sign and sub_blob.Derts[-1][-1] > ave_blob * 2:  # g > ave and G > fixed costs of comp_angle
                        master_blob( sub_blob, comp_angle)  # sub_blob = master ablob, no type rdn: 1 sub_blob_, no eval_layer

                        for ablob in sub_blob.sub_blob_:  # eval / ablob: unique, def / ga sign, vs. rdn ablob_ || xblob_ if / gblob
                            I, Dx, Dy, G = ablob.Derts[-2]  # Derts include params of all higher layers
                            A, Dxa, Dya, Ga = ablob.Derts[-1]
                            L = ablob.L

                            val_angle = Ga  # value of comp_ga -> gga, eval comp_angle(dax, day), next layer / aga_blob
                            val_deriv = ((G + ave * L) / ave * L) * -Ga  # relative G * -Ga: angle match, likely edge
                            val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

                            # estimated next-layer values per ablob:
                            val_ += [(val_angle, 0, ablob), (val_deriv, 1, ablob), (val_range, 2, ablob)]
                if val_:
                    eval_layer(val_, 2)  # rdn = 2: + ablobs

    return frame  # frame of 2D patterns is output to level 2


def branch(blob, typ):  # compute branch, evaluate next-layer branches: comp_angle, comp_ga, comp_deriv, comp_range
    vals = []

    if typ == 0:
        master_blob(blob, comp_angle)
        for ablob in blob.sub_blob[1]:

            I, Dx, Dy, G = ablob.Derts[-2]   # Derts include params of all higher layers
            A, Dxa, Dya, Ga = ablob.Derts[-1]; L = ablob.L

            val_angle = Ga  # value of comp_ga -> gga, eval comp_angle(dax, day), next layer / aga_blob
            val_deriv = ((G + ave * L) / ave * L) * -Ga  # relative G * -Ga: angle match, likely edge
            val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

            # estimated next-layer values per ablob:
            vals += [(val_angle, 1, ablob), (val_deriv, 2, ablob), (val_range, 3, ablob)]
    else:
        if typ == 1:   master_blob(blob, comp_deriv, 1)  # comp over ga_: 1 selects aDert at Derts[-1]
        elif typ == 2: master_blob(blob, comp_deriv, 0)  # comp over g_ with incremental derivation
        else:          master_blob(blob, comp_range, 0)  # comp over i_ with incremental distance

        for xblob in blob.sub_blob[1]:
            vals += [(xblob.Derts[-1][-1], 0, xblob)]  # estimated value of comp.angle = G

    return vals  # also, blob is converted into branch-specific master_blob with added Dert[-1]


def eval_layer(val_, rdn):  # val_: estimated values of active branches in current layer across recursion tree per blob

    val_ = sorted(val_, key=lambda val: val[0])
    sub_val_ = []  # estimated branch values of deeper layer of recursion tree per blob
    map_ = []  # blob boxes + maps of stronger branches in val_, appended for next (lower) val evaluation

    while val_:
        val, typ, blob = val_.pop()
        for box, map in map_:
            olp = overlap(blob, box, map)
            rdn += 1 * (olp / blob.Derts[-1][-1])  # rdn += 1 * (olp / G):
            # redundancy to higher and stronger-branch overlapping blobs, * branch cost ratio?

        if val > ave * blob.Derts[-1][-1] * rdn + ave_blob:  # val > ave * G * rdn + fixed cost of master_blob per branch
            for sub_blob in blob.sub_blob_:  # sub_blobs are angle blobs

                sub_vals = branch(sub_blob, typ)  # branch-specific recursion step
                if sub_vals:  # not empty
                    sub_val_ += sub_vals

            map_.append((blob.box, blob.map))
        else:
            break

    blob.low_layers[0][:] += blob.sub_blobs[0][:]  # low_layers Derts params += sub_blobs Derts params, probably wrong
    blob.low_layers[1].append(blob.sub_blobs)  # low_layer_ += [new layer], or merge all sub_blobs in one array?

    if sub_val_:  # not empty
        rdn += 1  # ablob redundancy to default gblob, or rdn += 2 for additional cost of angle calc?
        eval_layer(sub_val_, rdn)  # evaluation of each sub_val for recursion

    ''' 
        comp_P_(val, 0, blob, rdn) -> (val_PP_, 4, blob), (val_aPP_, 5, blob),
        val_PP_ = 
        L + I + G:    proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant hyp_g_P calc, comp, no indiv comp: rdn
        * L/ Ly / Ly: elongation: >ave Pm? ~ box elong: (xn - x0) / (yn - y0)? 
        * Dy / Dx:    dimensional variation bias 
        * Ave - Ga:   angle match
        
        g and ga are dderived, blob of min_g?
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

'''
old branch(blob, typ):  # compute branch, evaluate next-layer branches: comp_angle, comp_ga, comp_deriv, comp_range

    vals = []    # master_blob: blob.sub_blobs[0][:] += Dert[:] :

    if typ == 0:   master_blob(blob, comp_deriv, 1)  # comp over ga_: 1 selects angle_Dert at Derts[-1]
    elif typ == 1: master_blob(blob, comp_deriv, 0)  # recursive comp over g_ with incremental derivation
    else:          master_blob(blob, comp_range, 0)  # recursive comp over i_ with incremental distance

    if blob.Derts[-1][-1] > ave_blob * 2:  # G > fixed costs of comp_angle
        master_blob(blob, comp_angle)  # converts blob into master ablob
        # or:
        # master_blob( ablob, comp_mixed) -> G_ -> eval_alayer: rdn olp to alt_typ branches' ablobs, then
        # master_blob( xblob, comp_angle) -> val_-> eval_xlayer: rdn olp to alt_typ branches' xblobs?
        # all vals are per sub_blob?

        for ablob in blob.sub_blob_:  # eval / ablob: unique, def / ga sign, vs. rdn ablob_ || xblob_ if / gblob
            Ly, L, I, Dx, Dy, G = ablob.Derts[-2]   # Derts include params of all higher layers
            Lya, La, A, Dxa, Dya, Ga = ablob.Derts[-1]

            val_angle = Ga  # value of comp_ga -> gga, eval comp_angle(dax, day), next layer / aga_blob
            val_deriv = ((G + ave * L) / ave * L) * -Ga  # relative G * -Ga: angle match, likely edge
            val_range = G - val_deriv  # non-directional G: likely d reversal, distant-pixels match

            # estimated next-layer values per ablob:
            vals += [(val_angle, 0, blob), (val_deriv, 1, blob), (val_range, 2, blob)]

    return vals  # blob is converted into master_blob with added Dert[-1]
'''
