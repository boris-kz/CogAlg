from collections import deque, defaultdict
from functools import reduce
from itertools import groupby, starmap

import numpy as np
import numpy.ma as ma

from frame_blobs import scan_P_, form_seg_
from intra_comp import comp_i

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
    derts__,  # intra_comp inputs
    forks [ layers [ Dert, sub_blob_] ]  # layers across derivation tree consist of forks: deriv+, range+, angle
    
        # feedback is mapped to all fork_type layers, with merged forking on Layers[1:]?
        # fork_tree is nested to depth = Layers[n]-1, for layer-parallel comp_blob
        # Dert may be None: params are summed if len(sub_blob_) > min, same for fork_ and fork_layer_?

    root_blob, # reference for feedback of all Derts params summed in sub_blobs
    hDerts     # higher-Dert params += higher-dert params (including I), for layer-parallel comp_blob, no forking
    '''

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

def intra_comp(i__, dert___, root_blob, rng, fork_type, Ave, Ave_blob):

    i__ = i__[root_blob.slices]  # i__, dert__ within root_blob' box:
    dert__ = dert___[-1][root_blob.slices]
    y0, yn, x0, xn = root_blob['box']

    P__ = form_P__(x0, i__, dert__, Ave)  # horizontal clustering
    seg_ = deque()  # buffer of running segments
    for y, P_ in enumerate(P__, start=y0):
        P_ = scan_P_(P_, seg_,
                     form_blob_func=merge_segment,  # merge' arguments:
                     root_blob=root_blob, dert___=dert___, rng=rng, fork_type=fork_type)
        seg_ = form_seg_(y, P_,
                         form_blob_func=merge_segment,  # merge' arguments:
                         root_blob=root_blob, dert___=dert___, rng=rng, fork_type=fork_type)

    # merge last-line segments into their blobs:
    while seg_: form_blob(seg_.popleft(), root_blob, rng)

    return Ave_blob * len(root_blob['Layers'][-1][fork_type]['sub_blob_']) / ave_n_sub_blobs


def form_P__(x0, i__, dert__, Ave):
    """
    Forms Ps across the whole dert array.
    """
    if i__.ndim == 2:  # inputs are g_derts
        dy_slice = 1
        dx_slice = 2
    elif i__.ndim == 3:  # inputs are ga_derts
        dy_slice = slice(1, 3)
        dx_slice = slice(3, None)
    else:
        raise ValueError
    g__ = dert__[0, :, :]

    # dert clustering by g sign:
    s_x_L__ = starmap(lambda y, g_:
                      (y,
                       [(sign, next(group)[0], len(list(group)) + 1)
                        for sign, group in groupby(enumerate(g_),
                                                   lambda x: x[1] > Ave)
                        if sign is not ma.masked]
                       ),
                      enumerate(g__))

    # P accumulation:
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


def merge_segment(term_seg, root_blob, dert___, rng, fork_type):

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


def terminate_blob(yn, blob, root_blob, dert___, rng, fork_type):

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

    feedback(blob)

# -----------------------------------------------------------------------------
def feedback(blob):

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

        # next layer of blob and sub_blob:
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
                        (G + Mg, 1),  # est. match of input gradient at rng+1, G & Mg = 0 if i is p, single exposed input per fork
                        (Gg, rng + 1),  # est. match of gg at rng+rng+1, initial fork, then more coarse?
                        (Ga, rng + 1)  # est. match of ga at rng+rng+1;  a_rng+/ g_rng+, no indep value, replaced by ga as i by g?
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

    return root_blob