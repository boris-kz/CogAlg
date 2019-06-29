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
    intra_comp also sends feedback to fork[flags] in root_blob, then to root_root_blob, etc.
    Blob structure:    
    
    Dert: G, A, Dx, Dy, L, Ly, # current Layer, += dert: g, a, (dx, dy), A: input g angle, None for gDert, i = derts[-1][fia]
       
    sign, # current g | ga sign
    rng,  # comp range, in each Dert 
    map,  # boolean map of blob to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    
    sub_blob_,  # layer-sequential references down sub_blob derivation tree, sub_blob structure = blob structure
    segment_ =  # references down blob formation tree, in vertical (horizontal) order  
        [ seg_params,  
          Py_ # vertical buffer of Ps per segment:
              [ P_params, derts_[ (g_dert, ga_dert) ]]: pair per current and prior derivation layer, rng rep in blob
        ],
    Layers[ # summed reps of lower layers across sub_blob derivation tree, from feedback, for layer-parallel comp_blob
                
            Dert, forks,  # input g_rng+, a_rng+, derived gg_rng2, ga_rng2, fork id: f_deriv, f_angle, Dert may be None  
            Dert, fforks, ... # fork_tree depth = Layers depth-1, += <= 8 (4 rng+, 4 der+) forks per Layer 
            # Dert params are summed if min n forks, also fork_Layers if min n Layers? 
          ],
    root_blob,  # reference for feedback of all Derts params summed in sub_blobs 
    high_Derts  # higher Dert params += higher-dert params (including I), for feedback to root_blob   
    '''

ave = 20   # average g, reflects blob definition cost, higher for smaller positive blobs, no intra_comp for neg blobs
kwidth = 3   # kernel width
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

ave_blob = 10000       # fixed cost of intra_comp per blob
rave = 20              # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

def intra_blob(root_blob, rng, fia, eval_fork_, Ave_blob, Ave):  # fga (flag ga) selects i_Dert and i_dert, no fia?

    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, root_blob fga = blob, ! fga,
    # local fork's blob is initialized in prior intra_comp's feedback(), no lower Layers yet

    for blob in root_blob.sub_blob_:  # sub_blobs are evaluated for comp_fork, add nested fork indices?
        if blob.Dert[0] > Ave_blob:  # noisy or directional G | Ga: > intra_comp cost: rel root blob + sub_blob_

            Ave_blob = intra_comp(blob, rng, fia, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.sub_blob_:  # sub_sub_blobs evaluated for root_dert angle calc & comp
                if sub_blob.Dert[0] > Ave_blob:  # G > intra_comp cost;  no independent angle value

                    Ave_blob = intra_comp(sub_blob, rng, fia, 1, Ave_blob, Ave)  # fa=1: same as fia?
                    Ave_blob *= rave  # Ave_blob adjusted by n_sub_blobs
                    Ave += ave
                    rdn = 1
                    G =  sub_blob.high_Derts[-2][0]  # input
                    Gg = sub_blob.high_Derts[-1][0]  # from first intra_comp in current-intra_blob
                    Ga = sub_blob.Dert[0]            # from last intra_comp

                    eval_fork_ += [   # sort per append?
                        (G, 1, 0),    # n_crit=3, rng = input rng, fia=0;  est. match of input gradient at rng+1
                        (G- Ave_blob, 1, 1),  # - g_blob cost, no calc_a;  est. match of input angle at rng+1
                        (Gg, rng, 0), # n_crit=2, rng = kernel rng, fia=0; est. match of gradient_of_gg at rng*2
                        (Ga, rng, 1)  # n_crit=2, rng = kernel rng, fia=1; est. match of gradient_of_ga at rng*2
                    ]
                    new_eval_fork_ = []  # forks recycled for next intra_blob
                    for val, n_crit, irng, fia in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

                        if val > ave_intra_blob * n_crit * rdn:  # cost of default eval_sub_blob_ per intra_blob
                           rdn += 1  # fork rdn = fork index + 1
                           rng += irng  # incremented by input rng, for current and recycled forks, or in next intra_blob?
                           Ave_blob += ave_blob * rave * rdn
                           Ave += ave * rdn
                           new_eval_fork_ += [(val, n_crit, rng, fia)]
                           intra_blob(sub_blob, rng, fia, new_eval_fork_, Ave_blob, Ave)  # root_blob.Layers[-1] += [fork]

                        else:
                            break
    ''' 
    beyond I and A: dderived match = dev (min) g -> g_blob ( miss = dev gg -> gg_blob
    val = G| Ga, overlap for higher Gg & Ga,   separate match for rng+ eval: summed consecutive min, average, vs. deviation?
    
    no:
    val_rg = G - Gg - Ga  # est match of input gradient at rng+1, change in magnitude and direction can be +|-?
    val_ra = val_rg - Ave_blob  # est. match of input angle at rng+1, no calc_a, - added root blob cost?
    val_gg = G - Ga   # est. gradient_of_gg match at rng*2 -> ggg, - Ga: direction noise
    val_ga = Ga       # est. gradient_of_ga match at rng*2;   der+ is always at higher-order rng+?

    higher-level overlap is for the weaker regardless of derivation? 
    rng+ accumulation in same Dert, new Dert per der+ only 
            
    intra_comp returns Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    ave and ave_blob *= fork coef, greater for coarse kernels, += input switch cost, or same for any new fork?
    no val_ra = val_rg  # primary comp_angle over a multiple of rng? different comp, no calc_a?
      (val_ra, 3, 0, 1, 1),  # n_crit=2, rng=0, fga=1, fia=1        
    
    input mag is weakly predictive, comp at rng=1, only g is worth comparing at inc_range (inc_deriv per i_dert): 
    derts[0] = 4i -> I feedback, not reinput? lower derts summation / comp?  comp_g(); eval? comp_angle (input g)

    2x2 g comp initially & across weak blobs: negative G - Gg - Ga, shift in same direction, after intra_blob, comp_blob?  
    all der+, if high Gg + Ga (0 if no comp_angle), input_g res reduction, convert to compare to 3x3 g?  
    weak blob deconstruction for 2x2 rng+, not for 3x3: rng+ stop only?
               
    3x3 g comp within strong blobs (core param Mag + Match > Ave), forming concentric-g rng+ or der+ fork sub-blobs:       
    rng+ blob def by indiv g - gg - ga, or g_blob-wide only, no sub-diff?   
    comp_P eval per angle blob, if persistent direction * elongation? 
        
    lateral comp val = proj match: G - Gg - Ga? contiguous inc range, no fb within P, sub_derts_ if min len only?
    vertical comp val = proj proj match: feedback-skipping inc range: maximize novel, non-redundant projections?
    
    kernel ave * ncomp (val= g_rate) * cost (ind * rdn): minor incr compared to clustering cost? 
    simplicity vs efficiency: if fixed code+syntax complexity cost < accumulated variable inefficiency cost, in delays?
    '''

    return root_blob








