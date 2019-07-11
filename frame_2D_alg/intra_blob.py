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

    Dert: G, A, Dx, Dy, L, Ly,  
    # core Layer of current blob, += dert: g, a, (dx, dy), if A is not None: A = root_blob Angle, fga = 1, i = derts[-1][fia]

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

import operator as op

from functools import reduce

import numpy as np
import numpy.ma as ma

from comp_i import comp_i

# -----------------------------------------------------------------------------
# Filters

ave = 20  # average g, reflects blob definition cost, higher for smaller positive blobs, no intra_comp for neg blobs
kwidth = 3  # kernel width
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

ave_blob = 10000  # fixed cost of intra_comp per blob
rave = 20  # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10  # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

# Declare comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

# -----------------------------------------------------------------------------
# Functions

def root_blob_to_sub_blobs(i__, dert___, root_blob, rng,
                           fga, fia, fa, Ave_blob, Ave):

    seg_ = deque()  # buffer of running segments
    _, height, width = derts__.shape

    for y in range(height):  # first and last row are discarded
        P_ = form_P_(i__[y], dert___[:, y].T)  # horizontal clustering
        P_ = scan_P_(P_, seg_, root_blob)
        seg_ = form_seg_(y, P_, root_blob)

    while seg_:  form_blob(seg_.popleft(), frame)  # last-line segs are merged into their blobs

    return Ave_blob


def form_P_():
    return

def scan_P_():
    return

def form_seg_():
    return

def form_blob()
    return


def intra_blob(root_blob, rng, fga, fia, eval_fork_, Ave_blob, Ave):  # fga (flag ga) selects i_Dert and i_dert, no fia?

    # two-level intra_comp eval per sub_blob, intra_blob eval per blob, root_blob fga = blob, ! fga,
    # local fork's blob is initialized in prior intra_comp's feedback(), no lower Layers yet

    dert___, blob_ = select_blobs(root_blob.sub_blob_, Ave_blob)
    a__, adert___ = comp_i(dert___, flags=F_ANGLE)

    for blob in blob_:  # (sub)blobs have been evaluated for root_dert angle calc & comp

        ablob_, Ave_blob = root_blob_to_sub_blobs(a__, adert___, blob, rng, fga, fia, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
        Ave_blob *= rave  # estimated cost of redundant representations per blob
        Ave += ave  # estimated cost per dert

    g__, gdert___ = comp_i(dert___, flags=F_DERIV)

    for blob in blob_:  # (sub)blobs have been evaluated for comp_fork, add nested fork indices?

        gblob_, Ave_blob = root_blob_to_sub_blobs(g__, gdert___, blob, rng, fga, fia, 1, Ave_blob, Ave)  # fa=1
        Ave_blob *= rave  # Ave_blob adjusted by n_sub_blobs
        Ave += ave
        rdn = 1
        G = sub_blob.high_Derts[-2][0]  # input
        Gg = sub_blob.high_Derts[-1][0]  # from first intra_comp in current-intra_blob
        Ga = sub_blob.Dert[0]  # from last intra_comp

        val_rg = G - Gg - Ga  # est. match of input gradient at rng+1: persistent magnitude and direction
        val_ra = val_rg - Ave_blob  # est. match of input angle at rng+1, no calc_a, - added root blob cost?
        val_gg = G - Ga  # est. gradient_of_gg match at rng*2 -> ggg, - Ga: direction noise
        val_ga = Ga  # est. gradient_of_ga match at rng*2;   der+ is always at higher-order rng+?

        eval_fork_ += [  # sort per append? n_crit: filter multiplier, fga: g_dert | ga_dert index:
            (val_rg, 3, 1, 0),  # n_crit=3, rng = input rng, fga=0
            (val_ra, 4, 1, 1),  # n_crit=4, rng = input rng, fga=1: works as fia here?
            (val_gg, 2, rng, 0),  # n_crit=2, rng = kernel rng, fga=0
            (val_ga, 1, rng, 1),  # n_crit=2, rng = kernel rng, fga=1
        ]
        new_eval_fork_ = []  # forks recycled for next intra_blob
        for val, n_crit, irng, fga in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

            if val > ave_intra_blob * n_crit * rdn:  # cost of default eval_sub_blob_ per intra_blob
                rdn += 1  # fork rdn = fork index + 1
                rng += irng  # incremented by input rng, for current and recycled forks, or in next intra_blob?
                Ave_blob += ave_blob * rave * rdn
                Ave += ave * rdn
                new_eval_fork_ += [(val, n_crit, rng, fga)]
                intra_blob(sub_blob, rng, fga, fia, new_eval_fork_, Ave_blob,
                           Ave)  # root_blob.Layers[-1] += [fork]

            else:
                break
    ''' 
    G, Gg, A, Ga are accumulated up to current rng in same-g Dert, no intermediate Derts, new Dert per der+ only
    G - Gg - Ga < 0: weak blob, deconstruct for fuzzy 2x2 comp,

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