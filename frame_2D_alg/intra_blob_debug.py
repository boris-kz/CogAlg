import numpy as np
import numpy.ma as ma
from math import hypot
from comp_angle_map import comp_angle
from comp_gradient_map import comp_gradient
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
            [ Dert = I, Dx, Dy, Ncomp, G ],  # one Dert per current and lower layers of blob derivation tree
        seg_ = 
            [ seg_params,  # formed in frame blobs
              Py_ =  # vertical buffer of Ps per segment
                [ P_params,
                  derts = [ dert = p|a, dx, dy, ncomp, g]]],  # one dert per current and higher (if blob is sub blob) layers 
                  # alternating g|a deep layers: p is replaced by a in odd derts and absent in even derts
                              
        sub_blob_ # sub_blob structure is same as in root blob 
            if len( sub_blob_) > min:
                sub_Derts[:] = [(Ly, L, I, Dx, Dy, Ncomp, G)]  # one Dert per positive sub_blobs and their higher layers
                 
        if eval_layer branch call:  # multi-layer intra_blob, top derts_ structure is layer_
            layer_f = 1
            sub_blob_ = [(sub_Derts, sub_blob_)]  # appended by flat|nested lower layers across derivation tree per root blob 
            
            if len( sub_blob_) > min
                sub_Derts[:] = [(Ly, L, I, Dx, Dy, Ncomp, G)]  # one Dert per sub_blob_s of referred layer & its higher layers
                
        map, box, rng
              
    Derts over sub_blobs_ and layer_ are same params as in Derts of their root blob, but summed from positive sub_blobs only
    Derts of Lower layers are not represented in higher layers.
    
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg
'''

def intra_blob_hypot(frame):  # evaluates for hypot_g and recursion, ave is per hypot_g & comp_angle, or all branches?

    for blob in frame.blob_:
        if blob.Derts[-1][-1] > ave_blob:  # G > fixed cost_form_sub_blobs: area of noisy or directional gradient
            unfold_blob(blob, hypot_g, rave_blob, 1)  # redefines g=hypot(dx,dy), adds sub_blob_, replaces blob params

            if blob.Derts[-1][-1] > ave_blob * 2:  # G > cost_form_sub_blobs * rdn to gblob
                intra_blob(blob, rave_blob, 1)  # redundancy to higher layers & prior blobs is added cost per input

    return frame  # frame of 2D patterns is output to level 2


def intra_blob(root_blob, rdn, rng):  # if not comp_angle, form_sub_blobs calls recursive intra_blob: comp_angle(comp_x:

    for blob in root_blob.sub_blob_:
        if blob.Derts[-1][-1] > ave_blob:  # G > fixed cost of sub blobs: area of noisy or directional gradient

            rdn += rave_blob  # += ave_blob / ave, then indiv eval? rdn += rave_blob per sub blob, else delete?
            unfold_blob(blob, comp_angle, rdn, rng)  # angle calc, immed comp (no angle eval), extend Derts and derts

            for ablob in blob.sub_blob_:  # ablob is defined by ga sign
                Ga_rdn = 0
                G = ablob.Derts[-2][-1]   # I, Dx, Dy, G; Derts: current + higher-layers params, no lower layers yet
                Ga = ablob.Derts[-1][-1]  # I converted per layer, not redundant to higher layers I

                if Ga > ave_blob:
                    Ga_rdn = rave_blob   # rdn increment / G+Ga blob, Ga priority: cheaper?
                    unfold_blob( ablob, comp_gradient, rdn + rave_blob, rng)  # -> angle deviation sub_blobs

                elif G * -Ga > ave_blob ** 2:  # 2^crit: input deviation * angle match: likely edge blob
                    unfold_blob( ablob, comp_gradient, rdn + rave_blob, rng)  # Ga must be negative: stable orientation

                if G + Ga > ave_blob*2 + Ga_rdn:  # 2*crit: input dev + angle dev: likely sign reversal & distant match
                    unfold_blob( ablob, comp_range, rdn + rave_blob + Ga_rdn, 1)  # rdn + ga_blob cost

    ''' typ1 p = ga: derts[-1][-1], typ2 p = g: derts[-2][-1], typ3 p = rng p: derts[-rng][1]

    ave and ave_blob are averaged between branches, else multiple blobs, adjusted for ave comp_range x comp_gradient rdn
    g and ga are dderived, blob of min_g? val -= branch switch cost?
    no Ave = ave * ablob.L, (G + Ave) / Ave: always positive?

    comp_P_() if estimated val_PP_ > ave_comp_P: 
    L + I + G:   proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant P calc, comp, no indiv comp: rdn
    * L/ Ly/ Ly: elongation: >ave Pm? ~ box elong: (xn - x0) / (yn - y0)? 
    * Dy / Dx:   variation-per-dimension bias 
    * Ave - Ga:  if positive angle match
    
    def unfold_blob(root_blob, rdn)
        for blob in root_blob.sub_blob_:
            if blob.Derts[-1][-1] > ave_blob:  # val_PP_ > ave_comp_P
                rdn *= comp_P_coef; unfold_blob( typ=0, blob, dx_g, rdn)  # ~ hypot_g, no comp
                
    # rdn += 1 + ave_n_sub_blobs per form_sub_blobs, if layered access & rep extension, else *=?
    # then corrected in form_sub_blobs: rdn += len(sub_blob_) - ave_n
    '''
    return root_blob


def hypot_g(P_, dert___):
    dert__ = []  # dert_ per P, dert__ per line, dert___ per blob

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for index, (p, dy, dx, ncomp, g) in enumerate(dert_):
            g = hypot(dx, dy)

            dert_[index] = [(p, dy, dx, ncomp, g)]  # p is replaced by a in even layers and absent in deeper odd layers
        dert__.append((x0, dert_))
    dert___.append(dert__)

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------

ave = 20
ave_blob = 200
ave_root_blob = 2000
rave_blob = ave_blob / ave_root_blob  # additive | multiplicative cost of converting blob to root_blob
