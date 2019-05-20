from math import hypot
from comp_angle import comp_angle
from comp_gradient import comp_gradient
from comp_range import comp_range
from intra_comp import intra_comp

'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg

    Recursive intra_blob comp_branch() calls add a layer of sub_blobs, new dert to derts and Dert to Derts, in each blob.
    Dert params are summed params of sub_blobs per layer of derivation tree.
    Blob structure:
        
        I,  # top Dert, or in Derts for feedback?
        Derts[ fork_Derts[ (typ_Derts: G, Ga)[ Dert: (G, Dx, Dy, N, L, Ly, rng, fa, sub_blob_)]],     
        
        # Dert per current and lower layers of derivation tree for Dert-parallel comp_blob, 
        # Dert rdn = par: parallel fork index, same-syntax cross-branch summation in deeper Derts  
        # sub_blob_ per Dert is nested to depth = Derts[index] for Dert-sequential blob -> sub_blob access
        
        sign, # lower Derts are sign-mixed at depth > 0, typ-mixed at depth > 1, rng-mixed at depth > 2:
        map,  # boolean map of blob, to compute overlap; map and box of lower Derts are similar to top Dert
        box,  # boundary box: y0, yn, x0, xn
        
        root_blob,  # reference, to return summed blob params
        eval_fork_ += [(rng, fa)] of iDert: g | ga in der+, p | ga[-1] in rng+, 
        new_eval_fork_ += [select forks] after eval for rng+ only, der+ eval is local 
    
        seg_ =  # seg_s of lower Derts are packed in their sub_blobs
            [ seg_params,  
              Py_ = # vertical buffer of Ps per segment
                  [ P_params,       
                    derts_[ typ_derts: g_dert, ga_dert (g, dx, dy, ?a) per current & higher derivation layer
'''
ave = 20
ave_blob = 1000  # fixed cost of blob syntax
rave = 10        # fixed cost ratio of converting blob to root_blob: Derts+=Dert, sub_blobs, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of eval per sub_blob, as below

# These filters are accumulated for evaluated intra_comp:
# Ave += ave: cost per next-layer dert, fixed comp grain: pixel
# Ave_blob *= rave: cost per next root blob, variable len sub_blob_

def intra_blob(root_blob, comp_branch, eval_fork_, fa, Ave_blob, Ave):  # fa (flag angle) selects i_Dert and i_dert

    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, ga_root_blob = g_blob
    new_eval_fork_ = []  # next intra_blob eval branches

    for blob in root_blob.Derts[-1][-1][fa][-1]:  # [cycle][fork][fa] sub_blobs are evaluated for comp_branch
        if blob.Derts[-1][-1][fa][0] > Ave_blob:  # noisy or directional G: > root blob conversion + sub_blob eval cost

            Ave_blob = intra_comp(blob, comp_branch, Ave_blob, Ave)  # Ave_blob adjusted by actual n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.Derts[-1][-1][fa][-1]:  # sub_sub_blobs are evaluated for angle calc & comp
                if sub_blob.Derts[-1][-1][fa][0] > Ave_blob:   # G > sub -> root blob conversion cost
                    # update rng in Derts[-1 cyc][-1 fork], vs. pass as args?

                    Ave_blob = intra_comp(sub_blob, comp_angle, Ave_blob, Ave)
                    Ave_blob *= rave
                    Ave += ave
                    rdn = 1
                    G =  sub_blob.Derts[-1][-1][0][0]  # Derts: current + higher-layers params, no lower layers yet
                    Ga = sub_blob.Derts[-1][-1][1][0]  # sub_blob eval / intra_blob fork: Derts+, ablobs eval / intra_comp:

                    val_gg  = G - Ga  # value of gradient_of_gradient deviation: directional variation
                    val_gga = Ga      # value of gradient_of_angle_gradient deviation, no ga direction yet
                    val_rg  = G + Ga  # value of rng=2 gradient deviation: non-directional variation
                    val_rga = G + Ga  # value of rng=2 angle gradient deviation, with comp_angle

                    eval_fork_ += [   # sort while appending?  adjust vals by lower Gs? comp i = derts[-1] if fa
                        (val_gg,  Ave_blob * 2, comp_gradient, 0, -1, 0, 1),  # current cyc, fa=0, rng = 1
                        (val_rg,  Ave_blob * 2, comp_range, 0,    -2, 0, 2),  # higher cyc,  fa=0, rng = 2
                        (val_gga, Ave_blob,     comp_gradient, 0, -1, 1, 1),  # current cyc, fa=1, rng = 1
                        (val_rga, Ave_blob * 2, comp_range, 0,    -1, 1, 2)   # current cyc, fa=1, rng = 2
                        ]

                    sorted(eval_fork_, key=lambda val: val[0], reverse=True)
                    for val, threshold, comp_branch, prior, cyc, fa, rng in eval_fork_:

                        threshold *= rdn
                        if val > threshold:
                            rdn += 1
                            if prior: rng += 1  # for recycled forks, cyc = -rng-2 if angle?
                            Ave_blob += ave_blob * rave * rdn
                            Ave += ave * rdn
                            new_eval_fork_ += [(val, threshold, comp_range, 1, cyc, fa, rng )]
                            # copy to all select forks with rdn-incremented filter

                            intra_blob(sub_blob, comp_branch, new_eval_fork_, fa, Ave_blob, Ave)
                            # root_blob.Derts[-1] += Derts, derts += dert
                        else:
                            break
    ''' 
    if Ga > Ave_blob: 
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # forms g_angle deviation sub_blobs

    if G - Ga > Ave_blob * 2:  # 2 crit, -> i_dev - a_dev: stable estimated-orientation G (not if -Ga) 
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # likely edge blob -> gg deviation sub_blobs

    if G + Ga > Ave_blob * 2:  # 2 crit, -> i_dev + a_dev: noise, likely sign reversal & distant match
       intra_comp( ablob, comp_range, Ave_blob, Ave)  # forms extended-range-g- deviation sub_blobs
    
    end of intra_comp:
    Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    
    if not comp_angle:
       if blob.Derts[-1][0] > Ave_blob + ave_eval:  # root_blob G > cost of evaluating sub_blobs
          intra_blob( ablob, Ave_blob, Ave, rng)     

    ave and ave_blob are averaged between branches, else multiple blobs, adjusted for ave comp_range x comp_gradient rdn
    g and ga are dderived, blob of min_g? val -= branch switch cost?
    no rG =  ga_blob.Derts[-2][-3][0]  # 2-distant input G
    no rGa = ga_blob.Derts[-2][-1][0]  # 2-distant angle Ga

    def intra_blob_hypot(frame):  # evaluates for hypot_g and recursion, ave is per hypot_g & comp_angle, or all branches?

        for blob in frame.blob_:
            if blob.Derts[-1][0] > ave_root_blob:  # G > root blob cost
                intra_blob(blob, hypot_g, ave_root_blob, ave)  # g = hypot(dy, dx), adds Dert & sub_blob_, calls intra_blob
    return frame

    comp_P_() if estimated val_PP_ > ave_comp_P, for blob in root_blob.sub_blob_: defined by dx_g:
     
    L + I + |Dx| + |Dy|: P params sum is a superset of their match Pm, 
    * L/ Ly/ Ly: elongation: >ave Pm? ~ box (xn - x0) / (yn - y0), 
    * Dy / Dx:   variation-per-dimension bias 
    * Ave / Ga:  angle match rate?
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


