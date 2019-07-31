'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:

    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg

    Each intra_cluster() call from intra_blob() adds a layer of sub_blobs, new dert to derts and Layer to Layers, in each blob.
    intra_cluster also sends feedback to fork[fia][fder] in root_blob, then to root_blob.root_blob, etc.
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
    derts__,  # intra_cluster inputs
    Layers[ fork_tree [type, Dert, sub_blob_] ]  # Layers across derivation tree consist of forks: deriv+, range+, angle

        # fork_tree is nested to depth = Layers[n]-1, for layer-parallel comp_blob
        # Dert may be None: params are summed if len(sub_blob_) > min, same for fork_ and fork_layer_?

    root_blob, # reference for feedback of all Derts params summed in sub_blobs
    hDerts     # higher-Dert params += higher-dert params (including I), for layer-parallel comp_blob, no forking
'''

import operator as op
from collections import deque, defaultdict
from functools import reduce
from itertools import groupby, starmap

import numpy as np
import numpy.ma as ma

from comp_i import (
    comp_i,
    F_ANGLE, F_DERIV, F_RANGE,
)
from utils import pairwise, flatten

# -----------------------------------------------------------------------------
# Filters

ave = 20   # average g, reflects blob definition cost, higher for smaller positive blobs, no intra_cluster for neg blobs
kwidth = 3   # kernel width
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

ave_blob = 10000       # fixed cost of intra_cluster per blob, accumulated in deeper layers
rave = 20              # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_cluster
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_cluster:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

# -----------------------------------------------------------------------------
# Functions


def intra_cluster(dert___, root_blob, Ave, Ave_blob, rng=1, flags=0):

    # Take dert__ inside root_blob's bounding box:
    dert__ = dert___[-1][root_blob['slices']]
    y0, yn, x0, xn = root_blob['box']

    P__ = form_P__(x0, y0, dert__, Ave,
                   fa=flags&F_ANGLE) # Horizontal clustering
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)
    blob_ = form_blob_(seg_, root_blob, dert___, rng, fork_type=flags) # flags as fork_type

    return blob_, Ave_blob * len(blob_) / ave_n_sub_blobs


def form_P__(x0, y0, dert__, Ave, fa):
    """Form Ps across the whole dert array."""
    g__ = dert__[0, :, :]  # g sign determines clustering:

    # Clustering:
    s_x_L__ = [*map(
        lambda g_: # Each line.
            [(sign, next(group)[0], len(list(group)) + 1) # (s, x, L)
             for sign, group in groupby(enumerate(g_ > Ave),
                                        op.itemgetter(1)) # (x, s): return s.
             if sign is not ma.masked], # Ignore gaps.
        g__, # Each line.
    )]

    # Accumulation:
    P__ = [[dict(sign=s,
                 x0=x+x0,
                 G=dert_[0, x : x+L].sum() - Ave * L,
                 M=0 if fa else dert_[1, x : x+L].sum(),
                 Dy=np.array(dert_[1:3, x : x+L].sum(axis=-1)) if fa
                 else dert_[2, x : x+L].sum(),
                 Dx=np.array(dert_[3:, x : x+L].sum(axis=-1)) if fa
                 else dert_[3, x : x+L].sum(),
                 L=L,
                 dert_=dert_[:, x : x+L].T,
                 root_=[],
                 fork_=[],
                 y=y)
            for s, x, L in s_x_L_]
           for dert_, (y, s_x_L_) in zip(dert__.swapaxes(0, 1),
                                         enumerate(s_x_L__, start=y0))]

    return P__


def scan_P__(P__):
    """Detect each P's forks and roots."""

    for _P_, P_ in pairwise(P__): # Iterate through pairs of lines.
        _P_, P_ = iter(_P_), iter(P_) # Convert to iterators.
        try:
            _P, P = next(_P_), next(P_) # First pair to check.
        except StopIteration: # No more fork-root pair.
            continue # To next pair of _P_, P_.
        while True:
            left, olp = comp_edge(_P, P) # Check for 4 different cases.
            if olp and _P['sign'] == P['sign']:
                _P['root_'].append(P)
                P['fork_'].append(_P)
            try:
                _P, P = (next(_P_), P_) if left else (_P, next(P_))
            except StopIteration: # No more fork-root pair.
                break # To next pair of _P_, P_.

    return [*flatten(P__)] # Flatten P__ before return.


def comp_edge(_P, P): # Used in scan_P_().
    """
    Check for end-point relative position and overlap.
    Used in scan_P__().
    """
    _x0 = _P['x0']
    _xn = _x0 + _P['L']
    x0 = P['x0']
    xn = x0 + P['L']

    if _xn < xn: # End-point relative position
        return True, x0 < _xn
    else:
        return False, _x0 < xn


def form_segment_(P_):
    """Form segments of vertically contiguous Ps."""

    seg_pars = 'y0', 'G', 'M', 'Dy', 'Dx', 'L', 'Ly', 'Py_', 'root_', 'fork_'

    # Get a list of all segment's first P:
    P0_ = [*filter(lambda P: (len(P['fork_']) != 1
                            or len(P['fork_'][0]['root_']) != 1),
                   P_)]

    # Form segments:
    seg_ = [dict(zip(seg_pars, # segment's params as keys
                     [Py_[0].pop('y'),] # y0
                     # Accumulate params:
                     + [*map(sum,
                             zip(*map(op.itemgetter('G',
                                                    'M',
                                                    'Dy',
                                                    'Dx',
                                                    'L'),
                                      Py_)))]
                     + [len(Py_)] # Ly
                     + [Py_]
                     + [Py_[-1]['root_']] # root_
                     + [Py_[0]['fork_']])) # fork_
            # cluster_vertical(P): traverse segnent from first P:
            for Py_ in map(cluster_vertical, P0_)]

    for seg in seg_:
        seg['Py_'][0]['seg'] = seg['Py_'][-1]['seg'] = seg

    for seg in seg_:
        seg.update(root_=[*map(lambda P: P['seg'], seg['root_'])],
                   fork_=[*map(lambda P: P['seg'], seg['fork_'])])

    for seg in seg_:
        for i, ref in ((0, 'fork_'), (-1, 'root_')):
            seg['Py_'][i].pop('seg')
            seg['Py_'][i].pop(ref)

    return seg_


def cluster_vertical(P): # Used in form_segment_().
    """
    Cluster P vertically, stop at the end of segment.
    Used in form_segment_().
    """

    if len(P['root_']) == 1 and len(P['root_'][0]['fork_']) == 1:
        P['root_'][0].pop('y')
        root = P.pop('root_')[0] # Only 1 root.
        root.pop('fork_') # Only 1 fork.
        return [P] + cluster_vertical(root)
    else:
        return [P]


def form_blob_(seg_, root_blob, dert___, rng, fork_type):
    encountered = []
    blob_ = []
    for seg in seg_:
        if seg in encountered:
            continue

        q = deque([seg])
        encountered.append(seg)

        s = seg['Py_'][0]['sign']
        G, M, Dy, Dx, L, Ly, blob_seg_ = 0, 0, 0, 0, 0, 0, []
        x0, xn = 9999999, 0
        while q:
            blob_seg = q.popleft()
            for ext_seg in blob_seg['fork_'] + blob_seg['root_']:
                if ext_seg not in encountered:
                    encountered.append(ext_seg)
            G += blob_seg['G']
            M += blob_seg['M']
            Dy += blob_seg['Dy']
            Dx += blob_seg['Dx']
            L += blob_seg['L']
            Ly += blob_seg['Ly']
            blob_seg_.append(blob_seg)

            x0 = min(x0, min(map(op.itemgetter('x0'), blob_seg['Py_'])))
            xn = max(xn, max(map(lambda P: P['x0']+P['L'], blob_seg['Py_'])))

        y0 = min(map(op.itemgetter('y0'), blob_seg_))
        yn = max(map(lambda segment: segment['y0']+segment['Ly'], blob_seg_))

        mask = np.ones((yn - y0, xn - x0), dtype=bool)
        for blob_seg in blob_seg_:
            for y, P in enumerate(blob_seg['Py_'], start=blob_seg['y0']):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y - y0, x_start:x_stop] = False

        # Form blob:
        blob = dict(
            Dert=dict(G=G, M=M, Dy=Dy, Dx=Dx, L=L, Ly=Ly),
            sign=s,
            box=(y0, yn, x0, xn),  # boundary box
            slices=(Ellipsis, slice(y0, yn), slice(x0, xn)),
            seg_=blob_seg_,
            rng = rng,
            dert___ = dert___,
            mask=mask,
            root_blob = root_blob,
            hDerts = np.concatenate(
                (
                    [[*root_blob['Dert'].values()]],
                    root_blob['hDerts'],
                ),
                axis=0
            ),
            forks=defaultdict(list),
            fork_type=fork_type,
        )

        feedback(blob)

        blob_.append(blob)

    return blob_


def feedback(blob, sub_fork_type=None): # Add each Dert param to corresponding param of recursively higher root_blob.
    root_blob = blob['root_blob']
    if root_blob is None: # Stop recursion.
        return
    fork_type = blob['fork_type']

    # Last blob Layer is deeper than last root_blob Layer:
    len_sub_layers = max(0, 0, *map(len, blob['forks'].values()))
    while len(root_blob['forks'][fork_type]) <= len_sub_layers:
        root_blob['forks'][fork_type].append((0, 0, 0, 0, 0, 0, []))
    # Equivalent with:
    # if len(root_blob['forks'][fork_type]) <= len_sub_layers:
    #     root_blob['forks'][fork_type].append((0, 0, 0, 0, 0, [])))

    # First layer accumulations:
    G, M, Dy, Dx, L, Ly = blob['Dert'].values()
    Gr, Mr, Dyr, Dxr, Lr, Lyr, sub_blob_ = root_blob['forks'][fork_type][0]
    root_blob['forks'][fork_type][0] = (
        Gr + G, Mr + M, Dyr + Dy, Dxr + Dx, Lr + L, Lyr + Ly,
        sub_blob_ + [blob],
    )

    # Accumulate deeper :) layers:
    root_blob['forks'][fork_type][1:] = \
        [*starmap( # Like map() except for taking multiple arguments.
            # Function (with multiple arguments):
            lambda Dert, sDert:
                (*starmap(op.add, zip(Dert, sDert)),), # Dert and sub_blob_ accum
            # Mapped iterables:
            zip(
                root_blob['forks'][fork_type][1:],
                blob['forks'][sub_fork_type][:],
            ),
        )]
    # Dert-only numpy.ndarray equivalent: (no sub_blob_ accumulation)
    # root_blob['forks'][fork_type][1:] += blob['forks'][fork_type]

    feedback(root_blob, fork_type)


def intra_blobs(eval_fork_, Ave_blob, Ave, rdn):
    new_eval_fork_ = []

    for val, blob_, rng, flags in eval_fork_:
        if val <= ave_intra_blob * rdn:
            return # Stop recursion
        rdn += 1
        Ave_blob += ave_blob * rave * rdn
        Ave += ave * rdn
        if flags & F_RANGE:
            rng += 1

        dert___, blob_ = select_blobs(blob_, Ave_blob) # Filter blobs
        dert___ = comp_i(dert___, rng, flags) # Comparison
        fork_sub_blob_ = []
        for blob in blob_:
            sub_blob_, Ave_blob = intra_cluster(dert___, blob, Ave, Ave_blob, rng, flags)
            fork_sub_blob_.extend(sub_blob_)
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert

        Gg = sum(map(lambda blob: blob['Dert']['G'], fork_sub_blob_))
        Ga = sum(map(lambda blob: blob['Dert']['Ga'], fork_sub_blob_))
        new_eval_fork_.extend += [
            (Gg, fork_sub_blob_, rng, F_DERIV),
            (Ga, fork_sub_blob_, rng, F_ANGLE),
        ]

        if not flags & F_ANGLE:
            G = sum(map(lambda blob: blob['Dert']['G'], blob_))
            Mg = sum(map(lambda blob: blob['Dert']['M'], fork_sub_blob_))
            new_eval_fork_.append((
                G+Mg,
                fork_sub_blob_,
                rng,
                F_RANGE,
            ))

    intra_blobs(sorted(new_eval_fork_), Ave_blob, Ave, rdn)

# -----------------------------------------------------------------------------

def intra_blob(root_blob, rng, eval_fork_, Ave_blob, Ave):  # fia (flag ia) selects input a | g in higher dert

    # two-level intra_comp eval per root_blob.sub_blob, deep intra_blob fork eval per input blob to last intra_comp
    # local fork's blob is initialized in prior intra_comp's feedback(), no lower Layers yet

    G, M, Dy, Dx, L, Ly, blob_ = root_blob['forks']['fork_type']

    for blob in blob_:  # sub_blobs are evaluated for comp_fork, add nested fork indices?
        G, M, Dy, Dx, L, Ly = blob['Dert']
        if G > Ave_blob:  # noisy or directional G | Ga: > intra_comp cost: rel root blob + sub_blob_

            Ave_blob = intra_comp(blob, rng, 0, Ave_blob, Ave)  # fa=0, Ave_blob adjust by n_sub_blobs
            Ave_blob *= rave  # estimated cost of redundant representations per blob
            Ave += ave  # estimated cost per dert
            G, M, Dy, Dx, L, Ly, sub_blob_ = blob['forks'][F_DERIV]
            for sub_blob in sub_blob_:  # sub_sub_blobs evaluated for root_dert angle calc & comp
                Gs, Ms, Dys, Dxs, Ls, Lys = sub_blob['Dert']
                if Gs > Ave_blob:  # G > intra_comp cost;  no independent angle value

                    Ave_blob = intra_comp(sub_blob, rng, 1, Ave_blob, Ave)  # fa=1: same as fia?
                    Ave_blob *= rave  # Ave_blob adjusted by n_sub_blobs
                    Ave += ave
                    rdn = 1
                    G = sub_blob['hDert'][-2][0]  # input gradient
                    Gg, Mg = sub_blob['hDert'][-1][0:2]  # from first intra_comp in current-intra_blob
                    Ga = sub_blob['Dert']['Ga']  # from last intra_comp, no m_angle ~ no m_brightness: mag != value

                    eval_fork_ += [  # sort per append?
                        (G + Mg, 1), # est. match of input gradient at rng+1, 0 if i is p, single exposed input per fork
                        (Gg, rng + 1), # est. match of gg at rng+rng+1, initial fork, then more coarse?
                        (Ga, rng + 1), # est. match of ga at rng+rng+1;  a_rng+/ g_rng+, no indep value, replaced by ga
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

def select_blobs(blob_, Ave_blob):
    """Return global dert___ and list of selected blobs."""
    # Get list of selected blob:
    selected_blob_ = [blob for blob in blob_
                      if blob['Dert']['G'] > Ave_blob] # noisy or directional G | Ga: > intra_cluster cost: rel root blob + sub_blob_

    # Get dert___ of selected blob:
    dert___ = blob_[0]['dert___'] # Get dert___ of any blob.
    shape = dert___[-1][0].shape # Get previous layer's g's shape.
    mask = reduce(merge_mask,
                  blob_,
                  np.ones(shape, dtype=bool))
    dert___[-1][:, mask] = ma.masked # Mask irrelevant parts.

    return dert___, selected_blob_

def merge_mask(mask, blob):
    mask[blob['slices']] &= blob['mask']
    return mask

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------