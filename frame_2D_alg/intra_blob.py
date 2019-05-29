from math import hypot
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
        
        layers[ I, fga_Derts, fga_forks, fga_fork_forks...]  fga=0 g_Dert: G, Dx, Dy, L, Ly, sub_blob_, fga=1 ga_Dert: +A?
        "
        blob contains summed reps of sub_blob_ layers for layer-parallel comp_blob, vs fork( sub_blob_- sequential unfold:
        forks and sub_blob_ are nested to depth = layers[index], forks = sorted [(cyc,fga)], Dert rdn = fork index + 1
        
        intra_comp initializes layers[0] and layers[1] per sub_blob, feedback adds or accumulates nested fork_layers:
        forks [(cyc,fga)] is nested to depth = layer depth-1: 
        forks within each fga_fork, sequential feedback index += [target_index] with elevation?
        "
        sign, # lower layers are mixed-sign
        map,  # boolean map of blob, to compute overlap; map and box of lower Derts are similar to top Dert
        box,  # boundary box: y0, yn, x0, xn
        
        root_blob,   # reference, to return summed blob params
        eval_fork_ = [(rng, fga)] of iDert, 
        new_eval_fork_ = [forks recycled from eval_fork_], for rng+ only, der+ is local 
    
        seg_ =  # seg_s of lower Derts are packed in their sub_blobs
            [ seg_params,  
              Py_ = # vertical buffer of Ps per segment
                  [ P_params,       
                    derts_[ fa_derts (g_dert (g, dx, dy), ga_dert (g, a, dx, dy)) per current & higher derivation layer
'''
ave = 20
ave_blob = 1000  # fixed cost of blob syntax
rave = 10        # fixed cost ratio of converting blob to root_blob: Derts+=Dert, sub_blobs, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of eval per sub_blob, as below

# These filters are accumulated for evaluated intra_comp:
# Ave += ave: cost per next-layer dert, fixed comp grain: pixel
# Ave_blob *= rave: cost per next root blob, variable len sub_blob_

def intra_blob(root_blob, rng, fga, fia, eval_fork_, Ave_blob, Ave):  # rng->cyc and fa (flag angle) select i_Dert and i_dert

    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, ga_root_blob = g_blob
    new_eval_fork_ = []  # next intra_blob eval branches

    for blob in root_blob.Derts[-1][fga][-1]:  # [cyc][fga] sub_blobs are evaluated for comp_fork, add nested fork indices?
        if blob.Derts[-1][fga][0] > Ave_blob:  # noisy or directional G: > root blob conversion + sub_blob eval cost

            Ave_blob = intra_comp(blob, rng, fga, fia, 0, Ave_blob, Ave)  #  fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.Derts[-1][fga][-1]:  # sub_sub_blobs are evaluated for angle calc & comp
                if sub_blob.Derts[-1][fga][0] > Ave_blob:   # G > sub -> root blob conversion cost

                    Ave_blob = intra_comp(sub_blob, rng, fga, fia, 1, Ave_blob, Ave)  # fa=1, Ave_blob adjust by n_sub_blobs
                    Ave_blob *= rave  # rng is passed by eval_fork_?
                    Ave += ave
                    rdn = 1
                    G =  sub_blob.Derts[-1][0][0]  # Derts: current + higher-layers params, no lower layers yet
                    Ga = sub_blob.Derts[-1][1][0]  # sub_blob eval / intra_blob fork, ga_blobs eval / intra_comp:

                    val_gg  = G - Ga  # value of gradient_of_gradient deviation: directional variation
                    val_gga = Ga      # value of gradient_of_angle_gradient deviation, no ga angle yet
                    val_rg  = G + Ga  # value of rng=2 gradient deviation:  non-directional variation
                    val_ra  = val_rg  # value of rng=2 angle gradient deviation, angle has no value?

                    eval_fork_ += [   # sort while appending?  adjust vals by lower Gs? cyc = -rng - 1 - fia
                        (val_gg,  Ave_blob * 2, 0, 0, 0),  # rng=1, fga=0, fia=0: nested i_index per fork
                        (val_rg,  Ave_blob * 2, 1, 0, 0),  # rng=2, fga=0, fia=0
                        (val_gga, Ave_blob,     0, 1, 0),  # rng=1, fga=1, fia=0
                        (val_ra,  Ave_blob * 2, 1, 1, 1)   # rng=2, fga=1, fia=1;
                        ]

                    sorted(eval_fork_, key=lambda val: val[0], reverse=True)
                    for val, threshold, rng, fga, fia in eval_fork_:

                        threshold *= rdn
                        if val > threshold:
                            rdn += 1
                            rng += 1  # for current and recycled forks
                            Ave_blob += ave_blob * rave * rdn
                            Ave += ave * rdn
                            new_eval_fork_ += [(val, threshold, rng, fga, fia)]  # for all select forks, filter*rdn

                            intra_blob(sub_blob, rng, fga, fia, new_eval_fork_, Ave_blob, Ave)
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



