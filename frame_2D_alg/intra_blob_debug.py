from math import hypot
from comp_angle_map import comp_angle
from comp_gradient_map import comp_gradient
from comp_range_draft import comp_range
from intra_comp_debug import intra_comp

'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg

    Recursive intra_blob comp_branch() calls add a layer of sub_blobs, new dert to derts and Dert to Derts, in each blob.
    Dert params are summed params of selected sub_blobs, grouped in layers of layer_ derivation tree:
    
    blob =  
        typ,  # if typ==0: comparand is dert[0]: p|a, else comparand is dert[-1]: g, within dert in derts[-rng] 
        sign, Ly, L,  # these are common for all Derts, higher-Derts sign is always positive, else no branching 
        
        Derts = [ Dert = I, Ncomp, Dx, Dy, G ],  # Dert per current and lower layers of derivation tree per blob
        seg_ = 
            [ seg_params,  # formed in frame blobs
              Py_ = # vertical buffer of Ps per segment
                  [ P_params,
                    derts = [ dert = p|a, ncomp, dx, dy, g]]],  # one dert per current and higher layers 
                    # alternating sub g|a layers: p is replaced by angle in odd derts and is absent in even derts
                              
        sub_Derts = [(Ly, L, I, Dx, Dy, Ncomp, G)]  # per sub blob_, vs layer_ Derts, sub_blob_ ) layer, nested
        sub_blob_ # sub_blob structure is same as in root blob 
        
        if layer_f == 1: sub_blob_ = [(sub_Derts, sub_blob_)]  # converted to layer_: dim-reduced derivation tree
        map, box, rng  
        '''

ave = 20
ave_eval = 100  # fixed cost of evaluating sub_blobs and adding root_blob flag
ave_blob = 200  # fixed cost per blob, not evaluated directly

ave_root_blob = 1000  # fixed cost of intra_comp, converting blob to root_blob
rave = ave_root_blob / ave_blob  # relative cost of root blob, to form rdn
ave_n_sub_blobs = 10

# direct filter accumulation for evaluated intra_comp, vs. rdn:
# Ave += ave, Ave_blob *= rave

def intra_blob_hypot(frame):  # evaluates for hypot_g and recursion, ave is per hypot_g & comp_angle, or all branches?

    for blob in frame.blob_:
        if blob.Derts[-1][-1] > ave_root_blob:  # G > cost of forming root blob: highly noisy or directional gradient

            intra_comp(blob, hypot_g, ave, 1)  # redefines g=hypot(dx,dy), converts blob to root_blob
            Ave_blob = ave_root_blob * (len(blob.sub_blob_) / ave_n_sub_blobs)  # adjust by actual / ave n sub_blobs

            if blob.Derts[-1][-1] > ave_root_blob + ave_eval:  # root_blob G > cost of evaluating sub_blobs
                intra_blob( blob, Ave_blob, ave * 2, 1)  # Ave = ave * 2: filter for deeper-P formation

    return frame

def intra_blob(root_blob, Ave_blob, Ave, rng):  # recursive intra_comp(comp_branch) selection per branch per sub_blob

    Ave_blob *= rave  # estimated cost of redundant representations per blob
    Ave + ave         # estimated cost of redundant representations per dert

    for blob in root_blob.sub_blob_:

        if blob.Derts[-1][-1] > Ave_blob + ave_eval:  # G > cost of adding new layer of root_blob
            blob.layer_f = 1  # sub_blob_converted to layer_: [(sub_Derts, sub_blob_)]

            intra_comp(blob, comp_angle, Ave_blob, Ave, rng)  # angle calc and comp, no angle eval, add Dert and dert
            Ave_blob *= rave  # estimated cost per sub_blob, and its derts:
            Ave + ave

            for ablob in blob.sub_blob_:  # ablob is defined by ga sign
                Ga_rdn = 0
                G = ablob.Derts[-2][-1]   # I, N, Dx, Dy, G; Derts: current + higher-layers params, no lower layers yet
                Ga = ablob.Derts[-1][-1]  # I is converted per layer, not redundant to higher I

                if Ga > Ave_blob:
                    Ga_rdn = 1  # redundant to potential G + Ga blob, Ga priority: cheaper?
                    intra_comp( ablob, comp_gradient, Ave_blob, Ave, rng)  # -> angle deviation sub_blobs

                elif G * -Ga > Ave_blob ** 2: # 2^crit, -> input deviation * angle match: likely edge blob
                    intra_comp( ablob, comp_gradient, Ave_blob, Ave, rng)  # Ga must be negative: stable orientation

                if G + Ga > Ave_blob * 2 + Ga_rdn:  # 2*crit, -> i_dev + a_dev: likely sign reversal & distant match
                    intra_comp( ablob, comp_range, Ave_blob + ave_root_blob * Ga_rdn, Ave, 1)  # + rdn of ga root_blob

                    # at the end of intra_comp:
                    # Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / ave n sub_blobs
                    # if not comp_angle:
                    #   if ablob.Derts[-1][-1] > Ave_blob + ave_eval:
                    #       intra_blob( ablob, Ave_blob, Ave, rng)
    '''
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
    
    ! typ1 p = ga: derts[-1][-1], typ2 p = g: derts[-2][-1], typ3 p = rng p: derts[-rng][1]; ! dert[0] = a|g: both poss i?
    '''
    return root_blob


def hypot_g(P_, dert___):
    dert__ = []  # dert_ per P, dert__ per line, dert___ per blob

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for index, (p, ncomp, dy, dx, g) in enumerate(dert_):
            g = hypot(dx, dy)

            dert_[index] = [(p, ncomp, dy, dx, g)]  # p is replaced by a in even layers and absent in deeper odd layers
        dert__.append((x0, dert_))
    dert___.append(dert__)

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------


