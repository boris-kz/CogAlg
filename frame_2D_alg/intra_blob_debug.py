import numpy as np
import numpy.ma as ma
from comp_angle import comp_angle
from comp_gradient import comp_gradient
from comp_range_draft import comp_range
# from comp_P_ import comp_P_
from intra_comp_debug import unfold_blob, form_sub_blobs

'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    frame_blobs + intra_blob forms a 2D version of 1st-level algorithm.
    
    recursive intra_blob' eval_layer' branch() calls add new dert to derts, Dert to Derts, layer of sub_blobs to blob,
    where Dert params are summed params of all positive blobs per structure, with optional layer_:
    
    blob =  
        typ,  # typ 0: primary | angle_g_blob, typ 1: angle_ga_blob, typ 2: gg_blob, typ 3: range_g_blob, formed / eval 
        sign, Ly, L,  # these are common for all Derts, higher-Derts sign is always positive, else no branching 
        
        Derts = 
            [ Dert = I, Dx, Dy, G ],  # one Dert per current and lower layers of blob derivation tree
        seg_ = 
            [ seg_params,  # formed in frame blobs
              Py_ =  # vertical buffer of Ps per segment
                [ P_params,
                  derts = [ dert = i|a, dx, dy, g ]]],  # one dert per current and higher (if root blob is a sub blob) layers 
                  # alternating g|a layers, for 2+ odd derts: add ncomp and blob typ instead of i?
                              
        sub_blob_ # sub_blob structure is same as in root blob 
            if len( sub_blob_) > min:
                sub_Derts[:] = [(Ly, L, I, Dx, Dy, G)]  # one Dert per positive sub_blobs and their higher layers
                 
        if eval_layer branch call:  # multi-layer intra_blob, top derts_ structure is layer_
            layer_f = 1
            sub_blob_ = [(sub_Derts, sub_blob_)]  # appended by lower layers across derivation tree per root blob, flat | nested?  
            
            if len( sub_blob_) > min
                sub_Derts[:] = [(Ly, L, I, Dx, Dy, G)]  # one Dert per sub_blob_s of referred layer and its higher layers
                
        map, box, rng
              
    Derts over sub_blobs_ and layer_ are same params as in Derts of their root blob, but summed from positive sub_blobs only
    Derts of Lower layers are not represented in higher layers.
    
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg
'''

def intra_blob_hypot(frame):  # evaluates for hypot_g and recursion, initial ave per hypot_g & comp_angle, or all branches?

    for blob in frame.blob_:
        if blob.Derts[-1][-1] > ave_blob:  # G > fixed cost_form_sub_blobs: area of noisy or directional gradient
            unfold_blob( typ=0, blob, hypot_g, rdn=ave_n)  # redefines g=hypot(dx,dy), adds sub_blob_, replaces blob params

            if blob.Derts[-1][-1] > ave_blob * 2:  # G > cost_form_sub_blobs * rdn to gblob
                intra_blob(blob, rdn=ave_n)  # redundancy to higher-layers blobs += 1 + ave_n_sub_blobs per form_sub_blobs

    return frame  # frame of 2D patterns is output to level 2


def intra_blob(root_blob, rdn):  # if not comp_angle, form_sub_blobs calls recursive intra_blob: eval comp_angle ( comp_alt:

    for blob in root_blob.sub_blob_:
        if blob.Derts[-1][-1] > ave_blob:  # G > fixed cost of sub blobs: area of noisy or directional gradient
            rdn += ave_n
            unfold_blob( typ=0, blob, comp_angle, rdn)  # angle calc, immed comp (no eval by angle), extend Derts and derts

            for ablob in blob.sub_blob_:  # ablob is defined by ga sign
                G = ablob.Derts[-2][-1]   # I, Dx, Dy, G; Derts: current and higher layers' params, no lower layers yet
                Ga = ablob.Derts[-1][-1]  # I is redundant? sub-reference by dert[-1][1] = p_ref, after comp?
                Gaf = 0   # rdn increment for G + Ga blob, prior Ga: cheaper?

                if Ga > ave_blob:  # angle deviation sub_blobs
                    Gaf = 1        # typ1 p = ga: derts[-1][-1], typ2 p = g: derts[-2][-1], typ3 p = rng p: derts[-rng][1]?
                    unfold_blob( typ=1, ablob, comp_gradient, rdn + ave_n)

                elif G * -Ga > ave_blob ** 2:  # 2^crit: input deviation * angle match: likely edge blob
                    unfold_blob( typ=2, ablob, comp_gradient, rdn + ave_n)  # Ga must be negative: stable orientation?

                if G + Ga > ave_blob*2 + Gaf:  # 2*crit: input dev + angle dev: likely sign reversal and distant match
                    unfold_blob( typ=3, ablob, comp_range, rdn + ave_n + Gaf)  # rdn + 1 if overlap with ga blob

    '''
    comp_P_() if estimated val_PP_ > ave_comp_P: 
    L + I + G:   proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant hyp_g_P calc, comp, no indiv comp: rdn
    * L/ Ly/ Ly: elongation: >ave Pm? ~ box elong: (xn - x0) / (yn - y0)? 
    * Dy / Dx:   variation-per-dimension bias 
    * Ave - Ga:  if positive angle match

    ave and ave_blob are averaged between branches, else multiple blobs, adjusted for ave comp_range x comp_gradient rdn?
    g and ga are dderived, blob of min_g?  
    val -= branch switch cost?
    no Ave = ave * ablob.L, (G + Ave) / Ave: always positive  
    '''
    return root_blob


def hypot_g(blob):  # redefine master blob by reduced g and increased ave * 2: variable costs of comp_angle

    mask = ~blob.map[:, :, np.newaxis].repeat(4, axis=2)  # stack map 4 times to fit the shape of dert__: (width, height, number of params)
    blob.new_dert__[0] = ma.array(blob.dert__, mask=mask)  # initialize dert__ with mask for selective comp

    # redefine g = hypot(dx, dy), ave * 2 assuming that added cost of angle calc = cost of hypot_g calc
    blob.new_dert__[0][:, :, 3] = np.hypot(blob.new_dert__[0][:, :, 1], blob.new_dert__[0][:, :, 2]) - ave * 2

    return 1  # comp rng

    # ---------- hypot_g() end -----------------------------------------------------------------------------------

ave = 20
ave_blob = 200
ave_n = 4  # average number of sub_blobs per form_sub_blobs + 1

'''
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

for box, map in map_:  # of higher-value blobs in the layer, incrementally nested, with root_blobs per blob?

    olp = overlap(blob, box, map)  # or no lateral overlap between xblobs?
    rdn += 1 * (olp / blob.Derts[-1][-1])  # rdn += 1 * (olp / G):
    # redundancy to higher and stronger-branch overlapping blobs, * specific branch cost ratio? 
'''