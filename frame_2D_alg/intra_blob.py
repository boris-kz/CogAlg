from intra_comp import intra_comp

'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:
    
    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg

    Each intra_comp() call from intra_blob() adds a layer of sub_blobs, new dert to derts and Layer to Layers, in each blob.
    Layer params are summed params of sub_blobs per layer of derivation tree.
    Blob structure:
        
    Layers[ I, (Ga, A),  # Layers[n] += derts[n], a for rng+ comp_a 
            g_Dert, a_Dert, # (G, Dx, Dy, L, Ly, sub_blob_), +A for ga_Dert 
            forks,  # g_rng+, a_rng+, g_rng2, ga_rng2 forks, fixed or mixed order?
            fforks,.. # incrementally nested forks in deeper Layers, up to 4 per fork layer?
          ]         
    sign, # lower layers are mixed-sign
    rng,  # one active rng per fork, no need for reverse rng per g dert: ref only to immediately higher g | ga dert? 
    map,  # boolean map of blob, to compute overlap; map and box of lower Layers are similar to top Layer
    box,  # boundary box: y0, yn, x0, xn 
    root_blob,  # reference, to return summed blob params
            
    seg_ =  # seg_s of lower Layers are packed in their sub_blobs
        [ seg_params,  
          Py_ = # vertical buffer of Ps per segment
              [ P_params,       
                derts_[ (g_dert (g, (dx, dy)), ga_dert (ga, a, (dax, day)), rng): cycle per current & prior derivation layers
                
    derivation layer reps are for layer-parallel comp_blob, nesting depth = Layer index in sub_blob_, Layer index-1 in forks
    layer-sequential forks( sub_blob_ unfolding, forks: sorted [(cyc,fga)], >0 layer pairs: from alt. input_comp, angle_comp
        
    intra_comp initializes Layers[0] and Layers[1] per sub_blob, feedback adds or accumulates deeper fork layer reps:
    fork_Layer [(fork, i_cyc, i_fga)], cyc != rng, generic fork may be Dert | sub_forks | sub_sub_forks..
    forks index sequentially nested in higher blobs: feedback index += [higher-layer index]
    '''

ave = 20   # ave g reflects blob definition cost, higher for smaller positive blobs, no intra_comp for neg blobs
ave_blob = 10000       # fixed cost of intra_comp per blob
rave = 20              # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

def intra_blob(root_blob, rng, fga, fia, eval_fork_, Ave_blob, Ave):  # fga (flag ga) selects i_Dert and i_dert, no fia?

    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, root_blob fga = blob, ! fga,
    # local fork's Layers[cyc=1][fga] = Dert, initialized in prior intra_comp's feedback(), no lower Layers yet
    # sub_blob eval / intra_blob fork, ga_blobs eval / intra_comp

    for blob in root_blob.Layers[1][1][-1]:  # [cyc=1][fga=1] sub_blobs are evaluated for comp_fork, add nested fork indices?
        if blob.Layers[1][1][0] > Ave_blob:  # noisy or directional G: > intra_comp cost: rel root blob + sub_blob_

            Ave_blob = intra_comp(blob, rng, fga, fia, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.Layers[1][0][-1]:  # [cyc=1][fga=0] sub_sub_blobs evaluated for root_dert angle calc & comp
                if sub_blob.Layers[1][0][0] > Ave_blob:  # G > intra_comp cost;  no independent angle value

                    Ave_blob = intra_comp(sub_blob, rng, fga, fia, 1, Ave_blob, Ave)  # fa=1, Ave_blob adjust by n_sub_blobs
                    Ave_blob *= rave  # rng from eval_fork_ or Layers[0]?
                    Ave += ave
                    rdn = 1
                    G =  sub_blob.Layers[0][0]     # I = Layers[-1][0], same as in shorter-rng sub_blobs Layers[-1]?
                    Ga = sub_blob.Layers[0][1][0]  # Ga, A = Layers[-1][1]: rng+ input is g, its a & ga are computed together
                    Gg = sub_blob.Layers[1][0][0]    # feedback from current-intra_blob's intra_comp(g)
                    Gga = sub_blob.Layers[1][1][0]   # feedback from current-intra_blob's intra_comp(ga)

                    val_rg = G - Gg - Gga  # est. match of input gradient at rng+1: persistent magnitude and direction
                    val_ra = val_rg - Ga - Ave_blob  # est. match of input angle at rng+1, no calc_a, - added root blob cost?
                    val_gg = G - Gga  # est. directional gradient_of_gradient match at rng*2 -> ggg, Gga: current direction noise
                    val_gga = Gga     # est. gradient_of_ga match at rng*2;   der+ is always with higher-order rng+1?

                    eval_fork_ += [  # sort per append? n_crit: filter multiplier, fga: g_dert | ga_dert index:
                        (val_rg,  3, rng, 0),  # n_crit=3, rng=i_rng, fga=0
                        (val_ra,  4, rng, 1),  # n_crit=4, rng=i_rng, fga=1: works as fia here?
                        (val_gg,  2, 0,   0),  # n_crit=2, rng=0, fga=0
                        (val_gga, 1, 0,   1),  # n_crit=2, rng=0, fga=1
                    ]
                    new_eval_fork_ = []  # forks recycled for next intra_blob
                    for val, n_crit, rng, fga in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

                        if val > ave_intra_blob * n_crit * rdn:  # cost of default eval_sub_blob_ per intra_blob
                           rdn += 1  # fork rdn = fork index + 1
                           rng += 1  # for current and recycled forks, or incremented in next intra_blob,
                           Ave_blob += ave_blob * rave * rdn
                           Ave += ave * rdn
                           new_eval_fork_ += [(val, n_crit, rng, fga)]
                           intra_blob(sub_blob, rng, fga, fia, new_eval_fork_, Ave_blob, Ave)  # root_blob.Layers[-1] += [fork]

                        else:
                            break
    ''' 
    G, A, Ga: input params;  G - Gg - Gga < 0: weak blob, deconstruct for fuzzy 2x2 comp
    Gg, Gga: derived params, accumulated up to current rng in same-g Dert, no intermediate Derts 
        
    forks recycle ig, ia, add comp_gg, comp_gga, not nested, -> 8 comps?
    fforks recycle <= 4 inputs, add <= 4 der+?  *= 4 forks per Layer, access down fga, frng tree?

    intra_comp returns Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    ave and ave_blob *= fork coef, greater for coarse kernels, += input switch cost, or same for any new fork?
    no val_ra = val_rg  # primary comp_angle over a multiple of rng? different comp, no calc_a?
      (val_ra, 3, 0, 1, 1),  # n_crit=2, rng=0, fga=1, fia=1        
    
    input mag is weakly predictive, comp at rng=1, only g is worth comparing at inc_range (inc_deriv per i_dert): 
    derts[0] = 4i -> I feedback, not reinput? lower derts summation / comp?  comp_g(); eval? comp_angle (input g)

    2x2 g comp initially & across weak blobs: negative G - Gg - Ga, shift in same direction, after intra_blob, comp_blob?  
    all der+, if high Gg + Ga (0 if no comp_angle), input_g res reduction, convert to compare to 3x3 g?  
               
    3x3 g comp within strong blobs (core param Mag + Match > Ave), forming concentric-g rng+ or der+ fork sub-blobs:       
    rng+ blob def by indiv g - gg - ga, or g_blob-wide only, no sub-diff?   
    comp_P eval per angle blob, if persistent direction * elongation? 
    no deconstruction to 2x2 rng+ if weak, stop only?
    
    lateral comp val = proj match: G - Gg - Ga? contiguous inc range, no fb within P, sub_derts_ if min len only?
    vertical comp val = proj proj match: feedback-skipping inc range: maximize novel, non-redundant projections?
    
    kernel ave * ncomp (val= g_rate) * cost (ind * rdn): minor incr compared to clustering cost? 
    simplicity vs efficiency: if fixed code+syntax complexity cost < accumulated variable inefficiency cost, in delays?
    '''

    return root_blob








