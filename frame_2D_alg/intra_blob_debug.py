from math import hypot
from comp_angle import comp_angle
from comp_gradient_map import comp_gradient
from comp_range import comp_range
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
    Dert params are summed params of selected sub_blobs, grouped in layers of derivation tree.
    Blob structure:
    
        Derts = [ Dert = Ly, L, I, N, Dx, Dy, G, sub_blob_],    
        
        # Dert per current & lower layers of derivation tree for Dert-parallel comp, 
        # same-syntax cross-type param summation in Dert = Derts[>1]: meaningful combined params?  
        # sub_blob_ per Dert is nested to depth = Derts[index] for Dert-sequential blob -> sub_blob access
        
        sign, # lower Derts are sign-mixed at depth > 0, inp-mixed at depth > 1, rng-mixed at depth > 2:
        inp,  # if inp==0: i = p|a in dert[0], else i = g in dert[-1], in derts[-rng] 
        rng,  # compared i = derts[-rng][inp], 0 if hypot_g, 1 if comp_gradient, > 1 if comp_range  
        
        map,  # boolean map of blob, to compute overlap; map and box of lower Derts are similar to top Dert
        box,  # boundary box: y0, yn, x0, xn
        root_blob,  # reference, to return summed blob params
        
        seg_ =  # seg_s of lower Derts are packed in their sub_blobs
            [ seg_params,  
              Py_ = # vertical buffer of Ps per segment
                  [ P_params,        
                  
                    derts_ [ derts [ dert = (p|none | a), ncomp, dx, dy, g ]]]  
                    # one dert per current and higher layers, with alternating p|a type of dert in derts: 
                    # derts [even][0] = p if top dert, else none or ncomp 
                    # derts [odd] [0] = angle
                    '''
ave = 20
ave_eval = 100  # fixed cost of evaluating sub_blobs and adding root_blob flag
ave_blob = 200  # fixed cost per blob, not evaluated directly

ave_root_blob = 1000  # fixed cost of intra_comp, converting blob to root_blob
rave = ave_root_blob / ave_blob  # relative cost of root blob, to form rdn
ave_n_sub_blobs = 10

# direct filter accumulation for evaluated intra_comp, with rdn represented as len(derts_)
# Ave += ave: cost per next-layer dert, linear for fixed comp grain
# Ave_blob *= rave: cost per next root blob


def intra_blob_hypot(frame):  # evaluates for hypot_g and recursion, ave is per hypot_g & comp_angle, or all branches?

    for blob in frame.blob_:
        if blob.Derts[-1][-1] > ave_root_blob:  # G > root blob conversion cost
            intra_comp(blob, hypot_g, ave_root_blob, ave, i_param=None, i_dert=None, rng=None)
            # redefines g as hypot(dy, dx), calls intra_blob

    return frame

def intra_blob(root_blob, Ave_blob, Ave, rng):  # recursive intra_comp(comp_branch) selection per branch per sub_blob

    Ave_blob *= rave  # estimated cost of redundant representations per blob
    Ave + ave         # estimated cost of redundant representations per dert

    for blob in root_blob.sub_blob_:

        if blob.Derts[-1][-1] > Ave_blob + ave_eval:  # noisy or directional G: > root blob conversion cost
            Ave_blob = intra_comp(blob, comp_angle, Ave_blob, Ave, i_param=0, i_dert=-1, rng=None)

            # angle calc & comp, no eval, Ave_blob return, else intra_blob call
            Ave_blob *= rave   # estimated cost per next sub_blob
            Ave + ave   # estimated cost per next comp

            for ablob in blob.sub_blob_:  # ablobs are defined by the sign of ga: gradient of angle
                Ga_rdn = 0
                G = ablob.Derts[-2][-1]   # Derts: current + higher-layers params, no lower layers yet
                Ga = ablob.Derts[-1][-1]  # different I per layer, not redundant to higher I

                if Ga > Ave_blob:  # forms angle deviation sub_blobs
                    Ga_rdn = 1  # redundant to G + Ga rng_blob, cheaper Ga priority
                    intra_comp( ablob, comp_gradient, Ave_blob, Ave, i_param=-1, i_dert=-1, rng=1)

                elif G - Ga > Ave_blob * 2:  # 2 * crit, -> i_dev - a_dev: likely edge blob, forms gg deviation sub_blobs
                    intra_comp( ablob, comp_gradient, Ave_blob, Ave, i_param=-1, i_dert=-2, rng=1)

                    # if stable orientation: -Ga, | G - Ga: orientation is only an estimate?

                if G + Ga > Ave_blob * (2 + Ga_rdn):  # 2 * crit, -> i_dev + a_dev: likely sign reversal & distant match
                    intra_comp( ablob, comp_range, Ave_blob + ave_root_blob * Ga_rdn, Ave, i_param=0, i_dert=-(rng*2+1), rng=rng)

                    # forms extended-range-g deviation sub_blobs
    '''
    intra_comp calls intra_blob:
    Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    
    if not comp_angle:
       if blob.Derts[-1][-1] > Ave_blob + ave_eval:  # root_blob G > cost of evaluating sub_blobs
          intra_blob( ablob, Ave_blob, Ave, rng)     # Ave: filter for deeper comp
    
    ave and ave_blob are averaged between branches, else multiple blobs, adjusted for ave comp_range x comp_gradient rdn
    g and ga are dderived, blob of min_g? val -= branch switch cost?
    no Ave = ave * ablob.L, (G + Ave) / Ave: always positive?

    comp_P_() if estimated val_PP_ > ave_comp_P:
     
    L + I + Dx + Dy:  proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant P calc, comp, no indiv comp: rdn
    * L/ Ly/ Ly: elongation: >ave Pm? ~ box elong: (xn - x0) / (yn - y0)? 
    * Dy / Dx:   variation-per-dimension bias 
    * Ave - Ga:  if positive angle match
    
    def unfold_blob(root_blob, rdn)
        for blob in root_blob.sub_blob_:
            if blob.Derts[-1][-1] > ave_blob:  # val_PP_ > ave_comp_P
                rdn *= comp_P_coef; unfold_blob( input=0, blob, dx_g, rdn)  # ~ hypot_g, no comp
                
    # rdn += 1 + ave_n_sub_blobs per form_sub_blobs, if layered access & rep extension, else *=?
    # then corrected in form_sub_blobs: rdn += len(sub_blob_) - ave_n
    
    no typ1 p = ga: derts[-1][-1], typ2 p = g: derts[-2][-1], typ3 p = rng p: derts[-rng][1]
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


