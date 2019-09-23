'''
    intra_blob() evaluates for recursive frame_blobs() and comp_P() within each blob.
    combined frame_blobs and intra_blob form a 2D version of 1st-level algorithm.
    to be added:

    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    inter_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg

    Each intra_comp() call from intra_blob() adds a layer of sub_blobs, new dert to derts and Layer to Layers, in each blob.
    intra_comp also sends feedback to root_fork | root_blob, then to root_blob.root_blob, etc.
    Blob structure:
    
    root_fork, # = root_blob['fork_'][iG]: reference for feedback of blob' Dert params and sub_blob_, up to frame
    
    Dert = G, Gg, M, Dx, Dy, Ga, Dax, Day, L, Ly,  
    # extended per fork: iG + gDert in g_fork, + aDert in a_fork, L, Ly are defined by Gg | Ga sign
    
    iG,   # fork type: index of compared ig in dert and criterion iG in Dert: 0 if G | 1 if Gg | 5 if Ga 
    sign, # of iG
    rng,  # comp range, in each Dert
    map,  # boolean map of blob to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    dert__, # compare inputs
       
    segment_[ seg_params, Py_ [(P_params, dert_)]],  # dert = g, gg, m, dy, dx, ga, day, dax
    # references down blob formation tree, accumulating Dert, in vertical (horizontal) order
    
    fork_ # multiple derivation trees per blob: 0-1 in g_blobs, 0-3 in a_blobs, next sub_blob_ is in layer_[0]:
        [
         layer_ [(Dert, sub_blob_)]  # alternating g(even) | a(odd) layers across derivation tree, seq access
        ]
        # deeper layers are mixed-fork with nested sub_blob_, Dert params are for layer-parallel comp_blob        
    '''

import operator as op

from collections import deque, defaultdict
from functools import reduce
from itertools import groupby, starmap, repeat

import numpy as np
import numpy.ma as ma

from comp_v import comp_v
from utils import pairwise, flatten

# -----------------------------------------------------------------------------
# Filters

ave = 20   # average g, reflects blob definition cost, higher for smaller positive blobs, no intra_cluster for neg blobs

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
# Other constants

gDERT_PARAMS = "I", "G", "M", "Dy", "Dx"
aDERT_PARAMS = gDERT_PARAMS + ("Ga", "Dyay", "Dyax", "Dxay", "Dxax")

P_PARAMS = "L", "x0", "dert_", "root_", "fork_", "y", "sign"
SEG_PARAMS = "S", "Ly", "y0", "x0", "xn", "Py_", "root_", "fork_", "sign"

gP_PARAM_KEYS = gDERT_PARAMS + P_PARAMS
aP_PARAM_KEYS = aDERT_PARAMS + P_PARAMS

gSEG_PARAM_KEYS = gDERT_PARAMS + SEG_PARAMS
aSEG_PARAM_KEYS = aDERT_PARAMS + SEG_PARAMS

# -----------------------------------------------------------------------------
# Functions

def intra_fork(blob, Ave_blob, Ave, rng, nI, fig, fa):  # fig: i is ig;  fa (or nI?) selects comp_i or comp_a

    # comp_i -> dert(i, g, ?m, dy, dx) | comp_a -> dert(i, g, ?m, dy, dx, a, ga, ?ma, day, dax):
    dert__ = comp_i(blob['dert__'], rng, nI, fig, fa)

    # default clustering by new G, crit = nI: g+ fork if 1, ga+ fork if 6
    gblob_, Ave, Ave_blob = cluster(dert__, blob['fork_'][nI], Ave_blob, Ave, rng, nI, fig, fa)  # dert__ -> blob_

    if fig:  # optional parallel clustering by new M + ?I:
        if nI == 0: crit = 2  # r+ fork
        else: crit = 7  # a+ fork: nI=3, or ra+ fork: nI=5
        rblob_, Ave, Ave_blob = cluster(dert__, blob['fork_'][crit], Ave_blob, Ave, rng, crit, fig, fa)

    if not fig:  # blob formed by comp_a or comp_p is evaluated for exclusive g+ and r+ forks, same in frame_blobs:
        # afork per +G blob, sub-cluster if aAve > gAve, merge selected +-ablob dert__s if aAve < gAve?
        # a+ nI = (3,4): dy,dx, no angle yet?

        for blob in gblob_:  # sub-clustering within blobs from default clustering

            if blob['Dert']['G'] > ave_intra_blob:  # +G blob, exclusive g+ fork; G: input or default clustering
                cluster_eval(blob, Ave_blob, Ave, rng * 2 + 1, 1, fig, ~fa)

            elif -blob['Dert']['G'] > ave_intra_blob:  # -G blob, exclusive r+ fork, m & M def in comp_P
                cluster_eval(blob, Ave_blob, Ave, rng+1, 2, fig, fa)  # not ~fa: eval same-dert r+, ! a+

    else:  # fork select per input blob, but params summed from sub-blobs, +gblobs and +rblobs don't overlap?
           # iforks per ablob, prioritized:  # or generic g+ and r+ forks?

            I, G, M, Dy, Dx, Ga, Ma, Dyay, Dyax, Dxay, Dxax, Ly, L = blob['Dert'].values()
            rdn = 1
            eval_fork_ = (
                (M + fig* I, 1, 2),  # r+ / est match of i= Dert[0] at rng+1
                (G, rng + 1, 1),     # g+ / est match of g= Dert[1] at rng+rng+1, same as rng+ if rng==0
                (Ma, 1, 7),          # ra+ / est match of a=Dert[5] at rng+1 (rng was redefined in a+ fork call)
                (Ga, rng + 1, 6),    # ga+ / est match of ga=Dert[6] at rng+rng+1
            )
            for val, irng, crit in sorted(eval_fork_, key=lambda val: val[0], reverse=True):

                if val > rdn * ave_intra_blob:  # cost of sub_blob_ eval in intra_fork
                    rdn += 1  # fork rdn = fork index + 1
                    Ave_blob += ave_blob * rave * rdn
                    Ave += ave * rdn
                    cluster_eval(blob, Ave_blob, Ave, rng, crit, fig, ~fa)
                else:
                    break
    return dert__


def cluster_eval(blob, Ave_blob, Ave, rng, crit, fig, fa):  # cluster -> blob_, eval per blob

    if fa:  # or crit = 3|5: both for afork, -> exclusive ga+ | ra+ forks?
        blob['fork_'][3] = dict(  # initialize root_fork with Dyay, Dxay = Day; Dyax, Dxax = Dax for comp angle
            I=0, G=0, M=0, Dy=0, Dx=0, Ga=0, Dyay=0, Dyax=0, Dxay=0, Dxax=0, S=0, Ly=0, blob_=[]
        )
    else:
        blob['fork_'][crit] = dict(  # initialize root_fork at ['fork_'][rdn-2 | crit]:
            I=0, G=0, M=0, Dy=0, Dx=0, Ly=0, L=0, blob_=[]
        )
    blob_, Ave, Ave_blob = cluster(blob['dert__'], blob['fork_'][crit], Ave_blob, Ave, rng, crit, fig, fa)
    # dert__ -> blob_, root_fork = blob['fork_'][crit]

    for blob in blob_:
        if crit == 2:   # M
            nI = 0; val = blob['Dert']['I'+'M']  # r+ fork
        elif crit == 7: # Ma
            nI = 5; val = blob['Dert']['Ma']  # ra+ fork, nI = a | (dy,dx)?
        else:
            nI = crit; val = blob['Dert'][nI]  # g+ if 1, ga+ if 6

        if val > ave_intra_blob:  # filter maybe specific for afork and hLe blob?

            blob['fork_'][nI] = dict(  # initialize root_fork, for gforks only
                I=0, G=0, M=0, Dy=0, Dx=0, S=0, Ly=0, blob_=[]
            )
            intra_fork(blob, Ave_blob, Ave, nI, rng, fig, fa)


def cluster(dert__, root_fork, Ave_blob, Ave, rng, crit, fig, fa):  # fig = dderived, i is ig, crit: clustering criterion

    P__ = form_P__(dert__, Ave, crit)  # horizontal clustering
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)  # vertical clustering
    blob_ = form_blob_(seg_, root_fork, crit)  # with feedback to root_fork

    Ave_blob *= len(blob_) / ave_n_sub_blobs
    Ave_blob *= rave  # cost per blob, same crit G for g_fork and a_fork
    Ave += ave  # cost per dert, both Ave and Ave_blob are for next intra_comp

    return blob_, Ave, Ave_blob


def form_P__(dert__, Ave, nI, dderived, x0=0, y0=0):
    """Form Ps across the whole dert array."""

    # Determine params type:
    if len(dert__) == 9: # No M.
        param_keys = aP_PARAM_KEYS[:2] + aP_PARAM_KEYS[3:]
    else:
        param_keys = aP_PARAM_KEYS if nI != 1 else gP_PARAM_KEYS

    # Select crit__ from dert__:
    dert__[nI, :, :] -= Ave
    crit__ = dert__[nI, :, :]
    if dderived:
        crit__ += dert__[2, :, :]

    # Clustering:
    s_x_L__ = [*map(
        lambda crit_: # Each line.
            [(sign, next(group)[0], len(list(group)) + 1) # (s, x, L)
             for sign, group in groupby(enumerate(crit_ > 0),
                                        op.itemgetter(1)) # (x, s): return s.
             if sign is not ma.masked], # Ignore gaps.
        crit__,  # line, blob slice
    )]

    Pderts__ = [[dert_[:, x : x+L].T for s, x, L in s_x_L_]
                for dert_, s_x_L_ in zip(dert__.swapaxes(0, 1), s_x_L__)]

    # Accumulation:
    # if not fa: I, G, M, Dy, Dx
    # if fa: I, G, M, Dy, Dx, Ga, Dyay, Dyax, Dxay, Dxaxz
    PDerts__ = map(lambda Pderts_:
                       map(lambda Pderts: Pderts.sum(axis=0),
                           Pderts_),
                   Pderts__)

    P__ = [
        [
            dict(zip( # Key-value pairs:
                param_keys,
                [*PDerts, L, x+x0, Pderts, [], [], y, s]
            ))
            for PDerts, Pderts, (s, x, L) in zip(*Pparams_)
        ]
        for y, Pparams_ in enumerate(zip(PDerts__, Pderts__, s_x_L__), start=y0)
    ]

    return P__


def scan_P__(P__):
    """Detect forks and roots per P."""

    for _P_, P_ in pairwise(P__): # Iterate through pairs of lines.
        _itP_, itP_ = iter(_P_), iter(P_) # Convert to iterators.
        try:
            _P, P = next(_itP_), next(itP_) # First pair to check.
        except StopIteration: # No more fork-root pair.
            continue # To next pair of _P_, P_.
        while True:
            isleft, olp = comp_edge(_P, P) # Check for 4 different cases.
            if olp and _P['sign'] == P['sign']:
                _P['root_'].append(P)
                P['fork_'].append(_P)
            try: # Check for stopping:
                _P, P = (next(_itP_), P) if isleft else (_P, next(itP_))
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

    if _xn < xn: # End-point relative position.
        return True, x0 < _xn # Overlap.
    else:
        return False, _x0 < xn

def form_segment_(P_, fa):
    """Form segments of vertically contiguous Ps."""
    # Determine params type:
    if "M" not in P_[0]:
        seg_param_keys = (*aSEG_PARAM_KEYS[:2], *aSEG_PARAM_KEYS[3:])
        Dert_keys = (*aDERT_PARAMS[:2], *aDERT_PARAMS[3:], "L")
    elif fa: # segment params: I G M Dy Dx Ga Dyay Dyax Dxay Dxax S Ly y0 Py_ root_ fork_ sign
        seg_param_keys = aSEG_PARAM_KEYS
        Dert_keys = aDERT_PARAMS + ("L",)
    else: # segment params: I G M Dy Dx S Ly y0 Py_ root_ fork_ sign
        seg_param_keys = gSEG_PARAM_KEYS
        Dert_keys = gDERT_PARAMS + ("L",)

    # Get a list of every segment's top P:
    P0_ = [*filter(lambda P: (len(P['fork_']) != 1
                            or len(P['fork_'][0]['root_']) != 1),
                   P_)]

    # Form segments:
    seg_ = [dict(zip(seg_param_keys, # segment's params as keys
                     # Accumulated params:
                     [*map(sum,
                           zip(*map(op.itemgetter(*Dert_keys),
                                    Py_))),
                      len(Py_), Py_[0].pop('y'), # Ly, y0
                      min(P['x0'] for P in Py_),
                      max(P['x0']+P['L'] for P in Py_),
                      Py_, # Py_ .
                      Py_[-1].pop('root_'), Py_[0].pop('fork_'), # root_, fork_ .
                      Py_[0].pop('sign')]))
            # cluster_vertical(P): traverse segment from first P:
            for Py_ in map(cluster_vertical, P0_)]

    for seg in seg_: # Update segs' refs.
        seg['Py_'][0]['seg'] = seg['Py_'][-1]['seg'] = seg

    for seg in seg_: # Update root_ and fork_ .
        seg.update(root_=[*map(lambda P: P['seg'], seg['root_'])],
                   fork_=[*map(lambda P: P['seg'], seg['fork_'])])

    for i, seg in enumerate(seg_): # Remove segs' refs.
        del seg['Py_'][0]['seg']

    return seg_


def cluster_vertical(P): # Used in form_segment_().
    """
    Cluster P vertically, stop at the end of segment.
    Used in form_segment_().
    """

    if len(P['root_']) == 1 and len(P['root_'][0]['fork_']) == 1:
        root = P.pop('root_')[0] # Only 1 root.
        root.pop('fork_') # Only 1 fork.
        root.pop('y')
        root.pop('sign')
        return [P] + cluster_vertical(root) # Plus next P in segment

    return [P] # End of segment


def form_blob_(seg_, root_fork):
    """
    Form blobs from given list of segments.
    Each blob is formed from a number of connected segments.
    """

    # Determine params type:
    if 'M' not in seg_[0]: # No M.
        Dert_keys = (*aDERT_PARAMS[:2], *aDERT_PARAMS[3:], "S", "Ly")
    else:
        Dert_keys = (*aDERT_PARAMS, "S", "Ly") if nI != 1 \
            else (*gDERT_PARAMS, "S", "Ly")

    # Form blob:
    blob_ = []
    for blob_seg_ in cluster_segments(seg_):
        # Compute boundary box in batch:
        y0, yn, x0, xn = starmap(
            lambda func, x_: func(x_),
            zip(
                (min, max, min, max),
                zip(*[(
                    seg['y0'],  # y0_ .
                    seg['y0'] + seg['Ly'],  # yn_ .
                    seg['x0'],  # x0_ .
                    seg['xn'],  # xn_ .
                ) for seg in blob_seg_]),
            ),
        )

        # Compute mask:
        mask = np.ones((yn - y0, xn - x0), dtype=bool)
        for blob_seg in blob_seg_:
            for y, P in enumerate(blob_seg['Py_'], start=blob_seg['y0']):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y - y0, x_start:x_stop] = False

        dert__ = root_fork['dert__'][:, y0:yn, x0:xn]
        dert__.mask[:] = mask

        blob = dict(
            Dert=dict(
                zip(
                    Dert_keys,
                    [*map(sum,
                          zip(*map(op.itemgetter(*Dert_keys),
                                   blob_seg_)))],
                )
            ),
            box=(y0, yn, x0, xn),
            seg_=blob_seg_,
            sign=blob_seg_[0].pop('sign'),  # Pop the remaining segment's sign.
            dert__=dert__,
            root_fork=root_fork,
            fork_=defaultdict(list),
        )
        blob_.append(blob)

        # feedback(blob)

    return blob_

def cluster_segments(seg_):

    blob_seg__ = [] # list of blob's segment lists.
    owned_seg_ = [] # list blob-owned segments.

    # Iterate through list of all segments:
    for seg in seg_:
        # Check whether seg has already been owned:
        if seg in owned_seg_:
            continue # Skip.

        blob_seg_ = [seg] # Initialize list of blob segments.
        q_ = deque([seg]) # Initialize queue of filling segments.

        while len(q_) > 0:
            blob_seg = q_.pop()
            for ext_seg in blob_seg['fork_'] + blob_seg['root_']:
                if ext_seg not in blob_seg_:
                    blob_seg_.append(ext_seg)
                    q_.append(ext_seg)
                    ext_seg.pop("sign") # Remove all blob_seg's signs except for the first one.

        owned_seg_ += blob_seg_
        blob_seg__.append(blob_seg_)

    return blob_seg__


def feedback(blob, fork=None): # Add each Dert param to corresponding param of recursively higher root_blob.

    root_fork = blob['root_fork']

    # fork layers is deeper than root_fork Layers:
    if fork is not None:
        if len(root_fork['layer_']) <= len(fork['layer_']):
            root_fork['layer_'].append((
                dict(zip( # Dert
                    root_fork['layer_'][0][0], # keys
                    repeat(0), # values
                )),
                [], # blob_
            ))

    """
    # First layer accumulation:
    G, M, Dy, Dx, L, Ly = blob['Dert'].values()
    Gr, Mr, Dyr, Dxr, Lr, Lyr, sub_blob_ = root_blob['forks'][fork_type][0]
    root_blob['forks'][fork_type][0] = (
        Gr + G, Mr + M, Dyr + Dy, Dxr + Dx, Lr + L, Lyr + Ly,
        sub_blob_ + [blob],
    )

    # Accumulate deeper layers:
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
    """
    if root_fork['root_blob'] is not None: # Stop recursion if false.
        feedback(root_fork['root_blob'])

'''
    # initialization before accumulation, Dert only?
    if fa:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'M'=0, 'Dy'=0, 'Dx'=0, 'Ga'=0, 'Day'=0, 'Dax'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0  # (g, gg, m, dy, dx) -> (g, gg, m, dy, dx, ga, day, dax)
    else:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'Dy'=0, 'Dx'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0, 0  # g -> (g, gg, m, dy, dx)
'''