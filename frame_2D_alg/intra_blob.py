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
    dert__, # comp_i inputs
       
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
from itertools import groupby, starmap

import numpy as np
import numpy.ma as ma

from comp_i import comp_i
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
gDert_params = ["I", "G", "M", "Dy", "Dx"]
aDert_params = gDert_params + ["Ga", "Dyay", "Dyax", "Dxay", "Dxax"]

P_params = ["L", "x0", "dert_", "root_", "fork_", "y", "sign"]
seg_params = ["S", "Ly", "y0", "Py_", "root_", "fork_", "sign"]

gP_param_keys = gDert_params + P_params
aP_param_keys = aDert_params + P_params

gseg_param_keys = gDert_params + seg_params
aseg_param_keys = aDert_params + seg_params

# -----------------------------------------------------------------------------
# Functions

def intra_fork(idert__, root_fork, Ave_blob, Ave, rng, inI, dderived, fa):  # root_fork ref is for blob_ and feedback
    # inI to distinguish from nI, the same as idert__ and dert__.

    # fork fa = ~ root_fork_fa, alternating between comp_g and comp_a layers (if dderived, not from frame_blobs or p_rng+)
    # comparison:
    dert__ = comp_i(idert__, rng, inI, fa)  # comp_g -> dert(i, g, ?m, dy, dx) | comp_a -> dert(i, g, ?m, dy, dx, ga, day, dax)

    nI = 5 if fa else 1 # primary clustering is by new g: gg | ga, dert__-> blob_, with added g|a_Dert per sub_blob Dert

    sub_blob_, Ave, Ave_blob = cluster(dert__, root_fork, Ave_blob, Ave, rng, nI, dderived, fa)

    for blob in root_fork[0]['blob_']:  # blob_ in layer_[0] of root_fork, filled by feedback of form_blob

        if fa or not dderived:  # sub_blobs (a_blobs formed by comp_a) are evaluated for g_forks (rng+, der+, if dderived: ga):

            for sub_blob in blob['fork_'][nI][0]['blob_']:  # sub-blobs in layer_[0], if any from higher-layer blob eval
                I, G, M, Dy, Dx, Ga, Dyay, Dyax, Dxay, Dxax = sub_blob['Dert'].values()
                if -G > ave_intra_blob:  # no der+, exclusive (overlapping rng+ and ga+) forks eval:

                    # +G and +M are largely exclusive, I is not but I+M is?
                    # val Ga = I (val_A) + G (val_A, but not A variation?), indep ga but not ma = -ga?
                    # weaker rng+|ga+ fork' filter *= rdn to stronger alt fork, per -g_sub_blob (no 1/1 overlap per ssub_blob):
                    if Ga > ave_intra_blob and Ga > I + M: # v_ga+ > v_rng+
                        cluster_eval(dert__, root_fork, Ave_blob, Ave, rng, 5, dderived, ~fa)  # cluster by ga for ga+ eval

                        if I + M > ave_intra_blob * 2:  # redundant fork.
                            cluster_eval(dert__, root_fork, Ave_blob, Ave, rng, 0, dderived, ~fa)  # cluster by i+m for rng+ eval

                    elif I + M > ave_intra_blob:   # rng+ val > ga+ val
                        cluster_eval(dert__, root_fork, Ave_blob, Ave, rng, 0, dderived, ~fa)  # cluster by i+m for rng+ eval

                        if Ga > ave_intra_blob * 2:  # redundant fork
                            cluster_eval(dert__, root_fork, Ave_blob, Ave, rng, 5, dderived, ~fa)  # cluster by ga for ga+ eval

                elif G > ave_intra_blob:
                    # exclusive der+ fork:
                    intra_fork(sub_blob['dert__'], sub_blob['fork_'][nI], Ave_blob, Ave, rng*2, nI, dderived, ~fa)

        else:  # top blob is g_blob, evaluated for single a_fork (a_sub_blobs will overlap g_sub_blobs formed by prior fork):

            blob['fork'][0] = dict(  # initialize root_fork with Dyay, Dxay = Day; Dyax, Dxax = Dax for comp angle
                I=0, G=0, M=0, Dy=0, Dx=0, Ga=0, Dyay=0, Dyax=0, Dxay=0, Dxax=0, S=0, Ly=0, blob_=[]
            )
            intra_fork(blob['dert__'], blob['fork_'][nI], Ave_blob, Ave, rng*2 + 1, nI, dderived, ~fa)  # fa = 1

    return dert__

def cluster(dert__, root_fork, Ave_blob, Ave, rng, nI, dderived, fa):  # nI defines clustering crit

    P__ = form_P__(dert__, Ave, nI, dderived)  # horizontal clustering
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)  # vertical clustering
    blob_ = form_blob_(seg_, root_fork, nI)  # with feedback to root_fork

    Ave_blob *= len(blob_) / ave_n_sub_blobs
    Ave_blob *= rave  # cost per blob, same crit G for g_fork and a_fork
    Ave += ave  # cost per dert, both Ave and Ave_blob are for next intra_comp

    if not fa:
        Ave *= 2; Ave_blob *= 2  # a_fork_coef = 2: > cost, Aves += redundant g_sub_blob_: ga_val < gg_val

    return blob_, Ave, Ave_blob


def cluster_eval(dert__, root_fork, Ave_blob, Ave, rng, nI, dderived, fa):  # combine cluster and eval, with sub-cluster if gfork?

    blob_, Ave, Ave_blob = cluster(dert__, root_fork, Ave_blob, Ave, rng, nI, dderived, fa)

    for blob in blob_:
        if nI == 0: val = blob['Dert']['I' + 'M']
        else: val = blob['Dert'][nI]

        if val > ave_intra_blob:  # filter maybe specific for a_fork and hLe blob?

            blob['fork'][nI] = dict(  # initialize root_fork for gforks
                I=0, G=0, M=0, Dy=0, Dx=0, S=0, Ly=0, blob_=[]
            )
            if nI: rng *= 2  # nI = 1|5
            intra_fork(blob['dert__'], blob['fork_'][nI], Ave_blob, Ave, nI, rng, dderived, fa)

def form_P__(dert__, Ave, nI, dderived, x0=0, y0=0):
    """Form Ps across the whole dert array."""

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

    # Accumulated params:
    # if not fa: G, Gg, M, Dy, Dx
    # if fa: G, Gg, M, Dy, Dx, Ga, Dyay, Dyax, Dxay, Dxaxz
    PDerts__ = map(lambda Pderts_:
                       map(lambda Pderts: Pderts.sum(axis=0),
                           Pderts_),
                   Pderts__)

    param_keys = aP_param_keys if nI != 1 else gP_param_keys
    if len(dert__) == 9:
        param_keys.remove("M")

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
    """ detect forks and roots per P"""

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
            try:
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

    if _xn < xn: # End-point relative position
        return True, x0 < _xn
    else:
        return False, _x0 < xn

def form_segment_(P_, fa):
    """Form segments of vertically contiguous Ps."""
    # Get a list of every segment's first P:
    P0_ = [*filter(lambda P: (len(P['fork_']) != 1
                            or len(P['fork_'][0]['root_']) != 1),
                   P_)]

    if fa:
        seg_param_keys = aseg_param_keys
        Dert_param_keys = aDert_params
    else:
        seg_param_keys = gseg_param_keys
        Dert_param_keys = gDert_params

    if "M" not in P_[0]:
        seg_param_keys.remove("M")
        Dert_param_keys.remove('M')

    # Form segments:
    seg_ = [dict(zip(seg_param_keys, # segment's params as keys
                     # Accumulated params:
                     [*map(sum,
                           zip(*map(op.itemgetter(*Dert_param_keys, 'L'),
                                    Py_))),
                      len(Py_), Py_[0].pop('y'), Py_, # Ly, y0, Py_ .
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
        return [P] + cluster_vertical(root)

    return [P]

# ----------------------------------------------------------------------
def form_blob_(seg_, root_fork, nI):
    """
    Form blobs from given list of segments.
    Each blob is formed from a number of connected segments.
    """

    # TODO:
    #  -Add Dert, box and mask computation.
    #  -Add feedback.

    # Form blob:
    blob_ = [
        dict(
            Dert=dict(G=G, M=M, Dy=Dy, Dx=Dx, L=L, Ly=Ly),
            box=(y0, yn, x0, xn),  # boundary box
            seg_=blob_seg_,
            sign=blob_seg_[0].pop('sign'), # Pop the remaining segment's sign.
            dert__= dert__[y0:yn, x0:xn],
            root_fork=root_fork,
            root_blob=root_fork['root_blob'],
            forks=defaultdict(list),
            fork_type=nI,
        )

        for blob_seg_ in fill_blobs(seg_)
    ]

    # feedback(blob_)

    return blob_

def fill_blobs(seg_):

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

'''
UNDER REVISION:
def feedback(blob, sub_fork_type=None): # Add each Dert param to corresponding param of recursively higher root_blob.

    root_blob = blob['root_blob']
    if root_blob is None: # Stop recursion.
        return
    fork_type = blob['fork_type']

    # blob Layers is deeper than root_blob Layers:
    len_sub_layers = max(0, 0, *map(len, blob['forks'].values()))
    while len(root_blob['forks'][fork_type]) <= len_sub_layers:
        root_blob['forks'][fork_type].append((0, 0, 0, 0, 0, 0, []))

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

    feedback(root_blob, fork_type)
'''

'''
    # initialization before accumulation, Dert only?
    if fa:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'M'=0, 'Dy'=0, 'Dx'=0, 'Ga'=0, 'Day'=0, 'Dax'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0  # (g, gg, m, dy, dx) -> (g, gg, m, dy, dx, ga, day, dax)
    else:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'Dy'=0, 'Dx'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0, 0  # g -> (g, gg, m, dy, dx)
'''