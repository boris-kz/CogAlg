'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. Gradient is estimated 
    as |dx| + |dy|, then selectively and more precisely as hypot(dx, dy), from cross-comparison among adjacent pixels.

    Complemented by intra_blob (recursive search within blobs), it will be a 2D version of first-level core algorithm.
    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:   comp_pixel (lateral and vertical comp) -> pixel + derivatives tuple: dert ) frame of derts: dert__ 
    2Le, line y-1: form_P(dert2) -> 1D pattern P
    3Le, line y-2: scan_P_(P, hP)-> hP, roots: down-connections, fork_: up-connections between Ps 
    4Le, line y-3: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    5Le, line y-4+ seg depth: form_blob(seg, blob): merge connected segments in fork_ incomplete blobs, recursively  

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, convert elements of 
    lower line into elements of new higher line, then displace elements of old higher line into higher function.

    Higher-line elements include additional variables, derived while they were lower-line elements.
    Processing is mostly sequential because blobs are irregular and very difficult to map to matrices.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array.

    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:

    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg
    Each intra_comp() call from intra_blob() adds a layer of sub_blobs, new dert to derts and Layer to Layers, in each blob.
    intra_comp also sends feedback to fork[fia][fder] in root_blob, then to root_blob.root_blob, etc.
    Blob structure:

    Dert: G, A, M, Dx, Dy, L, Ly,  # current Layer, += dert: g, a, m, (dx, dy), A: iG angle, None if gDert, M: match = min

    sign, # current g | ga sign
    rng,  # comp range, in each Dert
    map,  # boolean map of blob to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers

    segment_[  # references down blob formation tree, in vertical (horizontal) order
        seg_params,
        Py_ [(P_params, derts_)]  # vertical buffer of Ps per segment
        # derts [(g_dert, ga_dert)]: two layers per intra_blob, sum in blob.rng, i = derts[-1][fia]
        ],
    derts__,   # intra_comp inputs
    sub_blob_, # layer-sequential references down sub_blob derivation tree, sub_blob structure = blob structure
    lLayers[   # summed reps of lower layers across sub_blob derivation tree, from feedback, for layer-parallel comp_blob

        Dert, forks,  # input g_rng+, a_rng+, derived gg_rng2, ga_rng2, fork id: f_deriv, f_angle
        Dert, fforks, ... # fork_tree depth = Layers depth-1, += <= 8 (4 rng+, 4 der+) forks per Layer
        # Dert may be None: params are summed only if min nforks, also summed fork_Layers if min nLayers?
        ],
    root_blob, # reference for feedback of all Derts params summed in sub_blobs
    hLayers    # higher-Dert params += higher-dert params (including I), for feedback to root_blob, no forking
'''