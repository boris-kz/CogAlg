'''
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

import operator as op

from functools import reduce

import numpy as np
import numpy.ma as ma

from comp_i import comp_i

# -----------------------------------------------------------------------------
# Filters

ave = 20   # average g, reflects blob definition cost, higher for smaller positive blobs, no intra_comp for neg blobs
kwidth = 3   # kernel width
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

ave_blob = 10000       # fixed cost of intra_comp per blob, accumulated in deeper layers
rave = 20              # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

# Declare comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

# -----------------------------------------------------------------------------
# Functions

def root_blob_to_sub_blobs(i__, dert___, root_blob, Ave):

    # Take dert__ and i__ inside root_blob's box:
    i__ = i__[root_blob.slices]
    dert__ = dert___[-1][root_blob.slices]
    y0, yn, x0, xn = root_blob['box']

    seg_ = deque()  # buffer of running segments

    for y in range(y0, yn):  # first and last row are discarded
        P_ = form_P_(x0, i__[y], dert__[:, y].T, Ave)  # horizontal clustering
        P_ = scan_P_(P_, seg_, root_blob, rng)
        seg_ = form_seg_(y, P_, root_blob, rng)

    while seg_:  form_blob(seg_.popleft(), root_blob)  # last-line segs are merged into their blobs

    return Ave_blob


def form_P_(x0, i_, dert_, Ave):
    g = dert_[:, 0]
    groups = [(sign, next(group)[0], len([*group]) + 1)
              for sign, group in groupby(enumerate(g), lambda item: item[1][0] > Ave)
              if sign is not ma.masked]

    P_ = [dict(sign=s,
               x0=x+x0,
               I=i_[x : x+L].sum(),
               G=dert_[x : x+L, 0].sum(),
               Dy=dert_[x : x+L, 1].sum(),
               Dx=dert_[x : x+L, 2].sum(),
               L=L,
               dert_=dert_[x : x+L]) for s, x, L in groups]

    return P_

def scan_P_():
    return

def form_seg_():
    return

def form_blob()
    return


def intra_blob(root_blob, rng, eval_fork_, Ave_blob, Ave):  # fia (flag ia) selects input a | g in higher dert

    # two-level intra_comp eval per root_blob.sub_blob, deep intra_blob fork eval per input blob to last intra_comp
    # local fork's blob is initialized in prior intra_comp's feedback(), no lower Layers yet

    for blob in root_blob.sub_blob_:  # sub_blobs are evaluated for comp_fork, add nested fork indices?
        if blob.Dert[0] > Ave_blob:  # noisy or directional G | Ga: > intra_comp cost: rel root blob + sub_blob_

            Ave_blob = intra_comp(blob, rng, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

            for sub_blob in blob.sub_blob_:  # sub_sub_blobs evaluated for root_dert angle calc & comp
                if sub_blob.Dert[0] > Ave_blob:  # G > intra_comp cost;  no independent angle value

                    Ave_blob = intra_comp(sub_blob, rng, 1, Ave_blob, Ave)  # fa=1: same as fia?
                    Ave_blob *= rave  # Ave_blob adjusted by n_sub_blobs
                    Ave += ave
                    rdn = 1
                    G = sub_blob.high_Derts[-2][0]  # input gradient
                    Gg, Mg = sub_blob.high_Derts[-1][0, 2]  # from first intra_comp in current-intra_blob
                    Ga = sub_blob.Dert[0]  # from last intra_comp, no m_angle ~ no m_brightness: mag != value

                    eval_fork_ += [  # sort per append?
                        (G + Mg, 1),  # est. match of input gradient at rng+1
                        (Gg, rng + 1),  # est. match of gg at rng+rng+1
                        (Ga, rng + 1)  # est. match of ga at rng+rng+1
                    ]
                    new_eval_fork_ = []  # forks recycled for next intra_blob
                    for val, irng in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

                        if val > ave_intra_blob * rdn:  # cost of default eval_sub_blob_ per intra_blob
                            rdn += 1  # fork rdn = fork index + 1
                            rng += irng  # incremented by input rng, for current and recycled forks, or in next intra_blob?
                            Ave_blob += ave_blob * rave * rdn
                            Ave += ave * rdn
                            new_eval_fork_ += [(val, rng)]
                            intra_blob(sub_blob, rng, new_eval_fork_, Ave_blob, Ave)  # root_blob.Layers[-1] += [fork]

                        else:
                            break
    '''
    parallel forks: 
    i rng+ / v(i+m), i & m = 0 if i is p, single exposed input per fork
    g der+ / vg: initial fork, but then more coarse?
    ga der+/ vga, rng = der+'rng? 

    no (G+Mg- ave_blob, 1, 1): comp a if same-rng comp i, no indep value, replaced by ga as i is replaced by g?

    from comp_g -> mg, gg: summed over rng in same Dert, new Dert per der+ only 
    m is unsigned, weak bias: no ma, mx,my, mA: lags ga? no kernel buff: olp, no Mg -> Mgg: g is not common 

    dderived match beyond I and A: dev (min) g -> g_blob ( +match mg -> rg_blob, +miss gg -> gg_blob
    val = G| Ga, overlap for higher Gg & Ga;  higher-level overlap is per weaker (der_order - re-order cost)? 

    intra_comp returns Ave_blob *= len(blob.sub_blob_) / ave_n_sub_blobs  # adjust by actual / average n sub_blobs
    ave and ave_blob *= fork coef, greater for coarse kernels, += input switch cost, or same for any new fork?

    input mag is weakly predictive, comp at rng=1, only g is worth comparing at inc_range (inc_deriv per i_dert): 
    derts[0] = 4i -> I feedback, not reinput? lower derts summation / comp?  comp_g(); eval? comp_angle (input g)
    2x2 g comp initially & across weak blobs: negative G - Gg - Ga, shift in same direction, after intra_blob, comp_blob?  
    all der+, if high Gg + Ga (0 if no comp_angle), input_g res reduction, convert to compare to 3x3 g?  

    3x3 g comp within strong blobs (core param Mag + Match > Ave), forming concentric-g rng+ or der+ fork sub-blobs:       
    rng+ blob def by ind g + mg, or g_blob-wide only, no sub-dif? contiguous rng+, no fb in P, sub_derts_ if min len?

    comp_P eval per angle blob, if persistent direction * elongation? 
    weak blob deconstruction for 2x2 rng+, no deconstruction of 3x3: rng+ stop only?

    vertical comp val = proj proj match: feedback-skipping inc range: maximize novel, non-redundant projections?
    kernel ave * ncomp (val= g_rate) * cost (ind * rdn): minor incr compared to clustering cost? 
    simplicity vs efficiency: if fixed code+syntax complexity cost < accumulated variable inefficiency cost, in delays?
    '''

    return root_blob

def select_blobs(blob_, Ave_blob):
    """Return global dert___ and list of selected blobs."""
    # Get list of selected blob:
    selected_blob_ = [blob for blob in blob_
                      if blob['Dert']['G'] > Ave_blob] # noisy or directional G | Ga: > intra_comp cost: rel root blob + sub_blob_

    # Get dert___ of selected blob:
    dert___ = blob_[0]['dert___'] # Get dert___ of any blob.
    shape = dert___[-1][0].shape # Get previous layer's g's shape.
    mask = reduce(lambda m, blob: m[blob['slices'] & blob['mask']],
                  blob,
                  np.ones(shape, dtype=bool))
    dert___[-1][:, mask] = ma.masked # Mask irrelevant parts.

    return dert___, selected_blob_