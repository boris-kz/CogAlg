from math import hypot
from comp_angle import comp_angle
from comp_gradient import comp_gradient
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
    
        Derts = [ layer_Derts = [Dert = Ly, L, I, N, Dx, Dy, G, sub_blob_] ],  # len layer_Derts = rdn     
        
        # Dert per current & lower layers of derivation tree for Dert-parallel comp, 
        # same-syntax cross-type param summation in Dert = Derts[>1]: meaningful combined params?  
        # sub_blob_ per Dert is nested to depth = Derts[index] for Dert-sequential blob -> sub_blob access
        
        sign, # lower Derts are sign-mixed at depth > 0, inp-mixed at depth > 1, rng-mixed at depth > 2:
        inp,  # i_dert: -(rng*2+1) if comp_range, -1 if comp_angle, -2 if comp_gradient,
              # i_param: 0: i = p|a in dert[0], | -1: i = g in dert[-1];  for i = derts[i_dert][i_param]
        rng,  # 0 if hypot_g, 1 if comp_gradient, >1 if comp_range  
        
        map,  # boolean map of blob, to compute overlap; map and box of lower Derts are similar to top Dert
        box,  # boundary box: y0, yn, x0, xn
        root_blob,  # reference, to return summed blob params
        
        seg_ =  # seg_s of lower Derts are packed in their sub_blobs
            [ seg_params,  
              Py_ = # vertical buffer of Ps per segment
                  [ P_params,        
                  
                    derts_ [ derts [ dert = (p|None | a), ncomp, dx, dy, g ]]]  
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
        if blob.Derts[-1][-1] > ave_root_blob:  # G > root blob cost
            intra_comp(blob, hypot_g, ave_root_blob, ave)  # g = hypot(dy, dx), adds Dert & sub_blob_, calls intra_blob

    return frame

def intra_blob(root_blob, Ave_blob, Ave, rng):  # recursive intra_comp(comp_branch) selection per branch per sub_blob

    Ave_blob *= rave  # estimated cost of redundant representations per blob
    Ave + ave         # estimated cost of redundant representations per dert

    for blob in root_blob.sub_blob_:
        if blob.Derts[-1][-1] > Ave_blob + ave_eval:  # noisy or directional G: > root blob conversion cost

            blob.inp = None  # no i comp, angle calc & comp (no a eval), no intra_blob call from comp_angle
            blob.rng = 1
            Ave_blob = intra_comp(blob, comp_angle, Ave_blob, Ave)  # Ave_blob return from comp_angle only

            Ave_blob *= rave   # estimated cost per next sub_blob
            Ave + ave   # estimated cost per next comp

            for ablob in blob.sub_blob_:  # ablobs are defined by the sign of ga: gradient of angle
                rdn = 1
                G = ablob.Derts[-2][-1]   # Derts: current + higher-layers params, no lower layers yet
                Ga = ablob.Derts[-1][-1]  # different I per layer, not redundant to higher I

                val_ga = G       # value of forming gradient_of_angle deviation sub_blobs
                val_gg = G - Ga  # value of forming gradient_of_gradient deviation sub_blobs
                val_gr = G + Ga  # value of forming extended-range gradient deviation sub_blobs

                vals = sorted((
                    (val_ga, Ave_blob,   comp_gradient, (-1,-1), 1),  # [-2]: inp(i_param, i_dert), [-1]: rng
                    (val_gg, Ave_blob*2, comp_gradient, (-1,-2), 1),
                    (val_gr, Ave_blob*2, comp_range), (-1,-(rng*2+1)), rng), key=lambda val: val[0], reverse=True)

                for val, threshold, comp_branch, inp, rng in vals:

                    if val > threshold * rdn:
                        ablob.inp = inp
                        ablob.rng = rng
                        rdn += 1
                        intra_comp(ablob, comp_branch, Ave_blob + ave_root_blob * rdn, Ave + ave * rdn)
                        # root_blob.Derts += rdn Derts, derts += dert: higher-layers rep
                        # evaluates calling intra_blob
                    else:
                        break
    ''' 
    the above expanded:
    
    if Ga > Ave_blob:
       blob.inp = (-1,-1)  # i_param=-1, i_dert=-1
       blob.rng = 1
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # forms g_angle deviation sub_blobs

    if G - Ga > Ave_blob * 2:  # 2 * crit, -> i_dev - a_dev (stable-orientation G): likely edge blob
       blob.inp = (-1,-2)  # i_param=-1, i_dert=-2   # not if -Ga: orientation is only an estimate
       blob.rng = 1
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # forms gg deviation sub_blobs

    if G + Ga > Ave_blob * (2 + rdn):  # 2 * crit, -> i_dev + a_dev: likely sign reversal & distant match
       blob.inp = (-1, -(rng*2+1))  # i_param=-1, i_dert=-(rng*2+1)
       blob.rng = rng
       intra_comp( ablob, comp_range, Ave_blob, Ave)  # forms extended-range-g deviation sub_blobs
    
    end of intra_comp:
    Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    
    if not comp_angle:
       if blob.Derts[-1][-1] > Ave_blob + ave_eval:  # root_blob G > cost of evaluating sub_blobs
          intra_blob( ablob, Ave_blob, Ave, rng)     

    ave and ave_blob are averaged between branches, else multiple blobs, adjusted for ave comp_range x comp_gradient rdn
    g and ga are dderived, blob of min_g? val -= branch switch cost?
    no Ave = ave * ablob.L, (G + Ave) / Ave: always positive?

    comp_P_() if estimated val_PP_ > ave_comp_P, for blob in root_blob.sub_blob_: defined by dx_g:
     
    L + I + Dx + Dy:  proj P match Pm; Dx, Dy, abs_Dx, abs_Dy for scan-invariant P calc, comp, no indiv comp: rdn
    * L/ Ly/ Ly: elongation: >ave Pm? ~ box elong: (xn - x0) / (yn - y0)? 
    * Dy / Dx:   variation-per-dimension bias 
    * Ave - Ga:  if positive angle match
    '''

    return root_blob


def hypot_g(P_, dert___):
    dert__ = []  # dert_ per P, dert__ per line, dert___ per blob

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for i, (p, ncomp, dy, dx, g) in enumerate(dert_):
            g = hypot(dx, dy)

            dert_[i] = [(p, ncomp, dy, dx, g)]  # p is replaced by a in odd layers and absent in deep even layers
        dert__.append((x0, dert_))
    dert___.append(dert__)

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------


