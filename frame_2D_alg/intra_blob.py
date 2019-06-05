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
        
    Layers[ I, Dert_pair, forks_pair, fork_forks_pair..] Dert_pair: g_Dert (G, Dx, Dy, L, Ly, sub_blob_), ga_Dert: g_Dert + A
        
    sign, # lower layers are mixed-sign
    map,  # boolean map of blob, to compute overlap; map and box of lower Layers are similar to top Layer
    box,  # boundary box: y0, yn, x0, xn 
    root_blob,  # reference, to return summed blob params
            
    seg_ =  # seg_s of lower Derts are packed in their sub_blobs
        [ seg_params,  
          Py_ = # vertical buffer of Ps per segment
              [ P_params,       
                derts_[ fga_derts (g_dert (g, (dx, dy)), ga_dert (g, a, (dax, day))) per current and prior derivation layers
                
    derivation layer reps are for layer-parallel comp_blob, nesting depth = Layer index in sub_blob_, Layer index-1 in forks
    layer-sequential forks( sub_blob_ unfolding, forks: sorted [(cyc,fga)], >0 layer pairs: from alt. input_comp, angle_comp
        
    intra_comp initializes Layers[0] and Layers[1] per sub_blob, feedback adds or accumulates deeper fork layer reps:
    fork_Layer [(fork, i_cyc, i_fga)], generic fork may be Dert | sub_forks | sub_sub_forks...   
         
    forks index nesting in higher blobs: feedback index += [higher-layer index], alt fga, cyc+=1 if fga==0?    
    parallel | alternating rng+, sub_angle+ layers: reduced coord, angle res, vs computed g_angles in alt g, ga layers? '''

ave = 20
ave_blob = 1000  # fixed cost of blob syntax
rave = 10        # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork or revised with each access, unless tree reorder?
'''

def intra_blob(root_blob, rng, fga, fia, eval_fork_, Ave_blob, Ave):  # rng -> cyc and fga (flag ga) select i_Dert and i_dert

    new_eval_fork_ = []  # forks recycled from eval_fork_, for next intra_blob eval, rng+ only, der+ is local
    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, root_blob fga = blob !fga, local Dert = Layers[1][-1]

    for blob in root_blob.Layers[1][1][-1]:  # [cyc=1][fga=1] sub_blobs are evaluated for comp_fork, add nested fork indices?
        if blob.Layers[1][1][0] > Ave_blob:  # noisy or directional G: > root blob conversion + sub_blob eval cost

            Ave_blob = intra_comp(blob, rng, fga, fia, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.Layers[1][0][-1]:  # [cyc=1][fga=0] sub_sub_blobs evaluated for root_dert angle calc & comp
                if sub_blob.Layers[1][0][0] > Ave_blob:  # G > sub -> root blob conversion cost

                    Ave_blob = intra_comp(sub_blob, rng, fga, fia, 1, Ave_blob, Ave)  # fa=1, Ave_blob adjust by n_sub_blobs
                    Ave_blob *= rave  # rng is passed by eval_fork_?
                    Ave += ave
                    rdn = 1
                    G =  sub_blob.Layers[1][0][0]  # Derts: current + higher-layers params, no lower layers yet
                    Ga = sub_blob.Layers[1][1][0]  # sub_blob eval / intra_blob fork, ga_blobs eval / intra_comp:

                    val_gg  = G - Ga  # value of gradient_of_gradient deviation: directional variation
                    val_gga = Ga      # value of gradient_of_angle_gradient deviation, no ga angle yet
                    val_rg  = G + Ga  # value of rng=2 gradient deviation:  non-directional variation
                    val_ra  = val_rg  # value of rng=2 angle gradient deviation, + angle: no separate value?

                    eval_fork_ += [   # sort per append? nested index: rng->cyc, fga: g_dert | ga_dert, fia: g_inp | a_inp:

                        (val_gg,  2, 0, 0, 0),  # n_crit=2, rng=1, fga=0, fia=0
                        (val_rg,  2, 1, 0, 0),  # n_crit=2, rng=2, fga=0, fia=0
                        (val_gga, 1, 0, 1, 0),  # n_crit=1, rng=1, fga=1, fia=0
                        (val_ra,  2, 1, 1, 1)   # n_crit=2, rng=2, fga=1, fia=1: rng comp_angle -> ga, no compute_a?
                        ]  # n_crit: filter multiplier;  or val per combined lower-layers'G: accumulated per rng extension?

                    for val, n_crit, rng, fga, fia in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

                        if  val > ave_intra_blob * n_crit * rdn:  # cost of default eval_sub_blob_ per intra_blob
                            rdn += 1  # fork rdn = fork index + 1
                            rng += 1  # for current and recycled forks
                            Ave_blob += ave_blob * rave * rdn
                            Ave += ave * rdn
                            new_eval_fork_ += [(val, n_crit, rng, fga, fia)]  # selected forks are passed as arg:
                            intra_blob(sub_blob, rng, fga, fia, new_eval_fork_, Ave_blob, Ave)  # root_blob.Layers += [forks]
                        else:
                            break
    ''' 
    if Ga > Ave_blob: 
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # forms g_angle deviation sub_blobs

    if G - Ga > Ave_blob * 2:  # 2 crit, -> i_dev - a_dev: stable estimated-orientation G (not if -Ga) 
       intra_comp( ablob, comp_gradient, Ave_blob, Ave)  # likely edge blob -> gg deviation sub_blobs

    if G + Ga > Ave_blob * 2:  # 2 crit, -> i_dev + a_dev: noise, likely sign reversal & distant match
       intra_comp( ablob, comp_range, Ave_blob, Ave)  # forms extended-range-g- deviation sub_blobs
    
    intra_comp returns Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    ave & ave_blob *= fork coef, greater for coarse kernels, += input switch cost, or same for any new fork?  

    Simplicity vs efficiency: if fixed code+syntax complexity cost < accumulated variable inefficiency cost, in delays?
    
    Input mag is weakly predictive, comp at rng = 1/2 only, replaced by g for rng+ comp? 
    starting with rng=1 comp_g (orthogonal or 3x3?), then rng=1 comp_angle of initial g?
    
    dert: (g, (dx, dy)), derts[0] 4i -> struc_I: for feedback, not re-input? also lower derts summation in rng+ comp?  
    g, ga are dderived but angle blobs (directional patterns) are secondary: specific angle only in negative ga_blobs  
    '''

    return root_blob








