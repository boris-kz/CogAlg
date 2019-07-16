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

from collections import deque, defaultdict
from functools import reduce
from itertools import groupby, starmap

import numpy as np
import numpy.ma as ma

from frame_blobs import scan_P_, form_seg_
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

def intra_comp(i__, dert___, root_blob, Ave, rng, fork_type):

    # Take dert__ and i__ inside root_blob's box:
    i__ = i__[root_blob.slices]
    dert__ = dert___[-1][root_blob.slices]
    y0, yn, x0, xn = root_blob['box']

    P__ = form_P__(x0, i__, dert__, Ave) # Horizontal clustering
    seg_ = deque()  # Buffer of running segments
    for y, P_ in enumerate(P__, start=y0):
        P_ = scan_P_(P_, seg_,
                     form_blob_func=update_sub_blob,
                     # update_sub_blob() addition arguments:
                     root_blob=root_blob, dert___=dert___,
                     rng=rng, fork_type=fork_type)
        seg_ = form_seg_(y, P_,
                         form_blob_func=update_sub_blob,
                         # update_sub_blob() addition arguments:
                         root_blob=root_blob, dert___=dert___,
                         rng=rng, fork_type=fork_type)

    # Merge last-line segments into their blobs:
    while seg_:  form_blob(seg_.popleft(), root_blob, rng)

    return Ave_blob


def form_P_(x0, i_, dert_, Ave):
    """
    Form P_ on a single line of dert__. No a_ support.
    Deprecated. use form_P__ instead.
    """
    g = dert_[:, 0]

    # Group same-sign adjacent derts: (sign, index, length)
    s_x_L_ = [(sign, next(group)[0], len([*group]) + 1)
              for sign, group in groupby(enumerate(g),
                                         lambda item: item[1] > Ave)
              if sign is not ma.masked]

    # Build list of P:
    P_ = deque(dict(sign=s,
                    x0=x+x0,
                    I=i_[..., x : x+L].sum(axis=-1),
                    G=dert_[x : x+L, 0].sum() - Ave * L,
                    Dy=dert_[x : x+L, 1].sum(),
                    Dx=dert_[x : x+L, 2].sum(),
                    L=L,
                    dert_=dert_[x : x+L])
               for s, x, L in s_x_L_)

    return P_


def form_P__(x0, i__, dert__, Ave):
    """
    Form P across the whole dert array.
    """
    # Separating i__ vs. a__ cases:
    if i__.ndim == 2: # Case of i__.
        dy_slice = 1
        dx_slice = 2
    elif i__ndim == 3: # Case of a__.
        dy_slice = slice(1, 3)
        dx_slice = slice(3, None)
    else:
        raise ValueError
    g__ = dert__[0, :, :] # Assign g__ for partitioning.

    # Partition:
    s_x_L__ = starmap(lambda y, g_:
                      (y,
                       [(sign, next(group)[0], len(list(group)) + 1)
                        for sign, group in groupby(enumerate(g_),
                                                   lambda x: x[1] > Ave)
                        if sign is not ma.masked]
                       ),
                      enumerate(g__))

    # Clustering:
    P__ = [deque(dict(sign=s,
                      x0=x+x0,
                      I=i_[..., x : x+L].sum(axis=-1),
                      G=dert_[0, x : x+L].sum() - Ave * L,
                      Dy=dert_[dy_slice, x : x+L].sum(axis=-1),
                      Dx=dert_[dx_slice, x : x+L].sum(axis=-1),
                      L=L,
                      dert_=dert_[:, x : x+L].T,
                      )
                 for s, x, L in s_x_L_)
           for i_, dert_, (y, s_x_L_) in zip(i__,
                                             dert__.swapaxes(0, 1),
                                             s_x_L__)]

    return P__


def update_sub_blob(term_seg, root_blob, dert___, rng, fork_type):
    """Terminated segment is merged into it's blob."""
    y0, Is, Gs, Dys, Dxs, Ls, Lys, Py_, blob, roots = term_seg.values()
    I, G, Dy, Dx, L, Ly = blob['Dert'].values()

    blob['Dert'].update(I=Is + I,
                        G=Gs + G,
                        Dy=Dys + Dy,
                        Dx=Dxs + Dx,
                        L=Ls + L,
                        Ly=Lys + Ly)

    blob['open_segments'] += roots - 1  # Update Number of open segments

    if blob['open_segments'] == 0:
        terminate_blob(y0+Lys, blob, root_blob, dert___, rng, fork_type)


def terminate_sub_blob(yn, blob, root_blob, dert___, rng, fork_type):
    """Sub-blob is terminated and send feedback to it's root-blob."""
    Dert, s, [y0, x0, xn], seg_, open_segs = blob.values()

    mask = np.ones((yn - y0, xn - x0), dtype=bool)  # local map of blob
    for seg in seg_:
        seg.pop('roots')
        for y, P in enumerate(seg['Py_'], start=seg['y0']):
            x_start = P['x0'] - x0
            x_stop = x_start + P['L']
            mask[y - y0, x_start:x_stop] = False

    Dert.pop('I')
    blob.pop('open_segments')
    blob.update(box=(y0, yn, x0, xn),  # boundary box
                slices=(Ellipsis, slice(y0, yn), slice(x0, xn)),
                rng=rng,
                mask=mask,
                dert___=dert___,
                hDerts=np.concatenate(
                    (
                        np.array(root_blob['Dert'].values()),
                        root_blob['hDert'],
                    ),
                    axis=0,
                ),
                root_blob=root_blob,
                fork_type=fork_type,
                Layers={},
                )

    feed_back(blob)

# -----------------------------------------------------------------------------
def feed_back(blob):

    fork_type = blob['fork_type']
    root_blob = blob['root_blob']
    sub_blob = blob

    while root_blob:  # add each Dert param to corresponding param of recursively higher root_blob

        if len(sub_blob['Layers']) == len(root_blob['Layers']):  # last blob Layer is deeper than last root_blob Layer
            root_blob['Layers'].append(defaultdict(lambda:dict(G=0,
                                                               Dy=0,
                                                               Dx=0,
                                                               L=0,
                                                               Ly=0,
                                                               sub_blob_=[])))  # new layer: a list of fork_type reps

        root_blob['hDert'][:, :] += sub_blob['hDert'][1:, :]  # pseudo for accumulation of co-located params, as below:

        G, Dy, Dx, L, Ly = sub_blob['hDert'][0, :]
        Gr, Dyr, Dxr, Lr, Lyr = root_blob['Dert']
        root_blob['Dert'].update(
            G = Gr + G,
            Dy = Dyr + Dy,
            Dx = Dxr + Dx,
            L = Lr + L,
            Ly = Lyr + Ly,
        )
        # or sub_blob accumulation next to discrete Dert: each param_layer is sub-selective summation hierarchy?

        G, Dy, Dx, L, Ly = sub_blob['Dert'].values()
        Gr, Dyr, Dxr, Lr, Lyr, sub_blob_ = root_blob['Layers'][-1][fork_type]
        root_blob['Layers'][-1][fork_type].update(
            G = Gr + G,
            Dy = Dyr + Dy,
            Dx = Dxr + Dx,
            L = Lr + L,
            Ly = Lyr + Ly,
        )
        # then root_root_blob.Layers[1] sums across min n? source layers, buffered in target Layer?

        # Update root_blob-sub_blob pair:
        sub_blob = root_blob
        root_blob = sub_blob['root_blob']
        fork_type = sub_blob['fork_type']

    blob['root_blob']['Layers'][-1][fork_type]['sub_blob_'].append(blob)


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