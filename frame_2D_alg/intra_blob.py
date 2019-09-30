import operator as op
'''
    intra_blob() evaluates for recursive internal cross-comp and clustering: intra_fork() and comp_P(), within each blob.
    Which adds a layer of sub_blobs and sub_forks per blob, with feedback to root_fork, then root_blob, etc.
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob and comp_P.

    Input brightness is not correlated with predictive value, but both its stability and variation are: 
    stability: negative gradient deviation (-vvg), is predictive value of initial input, vs. gradient or its derivatives
    variation: positive gradient deviation (+vvg), is predictive value of input gradient, thus its cross-comparison.
    
    -vg is indirect indicator of comp_p rng+ value, lower precision than min: not for g
    vm = min | -vg - ave (lower?): double filter, complimentary to max_g | max_i; rng+ -> discrete ders: 
    value of comp_blob (as +vg), then blob-> params value distribution, not for alt_g intra_comps: exclusive?
            
    Blob structure:
    
    root_fork,  # = root_blob['fork_'][crit]: reference for feedback of blob' Dert params and sub_blob_, up to frame
    
    Dert = I, G, M, Dy, Dx, A, Ga, Ma, Day, Dax, S, Ly
    # extended per fork: nI + gDert in ifork, + aDert in afork, S: area, Ly: vert dim, defined by criterion sign
    # I: input, G: gradient, M: match, Dy, Dx: vert,lat Ds, A: angle, Ga: angle G, Ma: angle M, Day, Dax: angle Ds  
    
    crit, # index of clustering criterion in dert and Dert: g | ga for der+ fork, m | ma for rng+ fork 
    sign, # of crit
    rng,  # comp range, also per fork?
    map,  # boolean map of blob, to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    dert__, # comp_i inputs
       
    segment_[ seg_params, Py_ [(P_params, dert_)]],  # dert: i, g, ?m, dy, dx, a, ga, ?ma, day, dax
    # references down blob formation tree in vertical (horizontal) order, accumulating Dert params
    
    fork_ # 1|2 derivation trees per blob: g,m sub_blob_s per ilayer, g sub_blob_ per alayer
        [
         layer_ [(Dert, sub_blob_)]  # alternating g (even) | a (odd) layers across derivation tree
        ]
        # deeper layers are mixed-fork with nested sub_blob_, Dert params are for layer-parallel comp_blob    
    
    to be added:
    comp_P_: vectorization of critically elongated blobs, by cross-comparing vertically adjacent Ps within each,
    merge_blob_: combination of small negative-value blobs into infra-blob: for comp_blob_ but not intra_blob,
    comp_blob_:  cross-comp of sub_blobs ) blobs of same range and derivation, within root blob ) frame,
    comp_layer_: cross-comp of blob layers, after comp_blob_ within each
    (edge sub-blobs matching between levels may form composite blob, axis comp if sub_blobs within blob margin?)
    
    Combination of all these functions will form 2nd level 2D alg: a prototype for recursive meta-level alg
    '''

from collections import deque, defaultdict
from functools import reduce
from itertools import groupby, starmap

import numpy as np
import numpy.ma as ma

from comp_v import comp_v
from utils import pairwise, flatten

# -----------------------------------------------------------------------------
# Filters

ave = 20   # average g|m, reflects blob definition cost, may be different for comp_a?
ave_blob = 10000  # fixed cost of sub-clustering per blob, accumulated in deeper layers
''' 
kwidth = 3   # kernel width, if starting with 2
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

These filters are accumulated for evaluated intra_fork, rep per fork if tree reorder, else redefined at each access?
Ave += ave: next-layer cost per dert, fixed comp grain
Ave_blob += ave_blob * rave: next-layer cost per blob, retro-adjusted by len sub_blob_: '''

rave = 20              # fixed cost ratio of root_blob / blob: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted by len(sub_blob_) in cluster()
ave_clust_eval = 1000  # cost of eval in cluster_eval,    total cluster_eval cost = Ave_blob + ave_clust_eval
ave_intra_fork = 1000  # cost of comp + eval in intra_fork, total intra_fork cost = Ave_blob + ave_intra_fork

# Other constants
gDert_params = ["I", "G", "M", "Dy", "Dx"]
aDert_params = gDert_params + ["A", "Ga", "Ma", "Dyay", "Dyax", "Dxay", "Dxax"]

P_params = ["L", "x0", "dert_", "root_", "fork_", "y", "sign"]
seg_params = ["S", "Ly", "y0", "Py_", "root_", "fork_"]

gP_param_keys = gDert_params + P_params
aP_param_keys = aDert_params + P_params

gseg_param_keys = gDert_params + seg_params
aseg_param_keys = aDert_params + seg_params

# -----------------------------------------------------------------------------------------------------------------------
# Functions, ALL UNDER REVISION:


def intra_fork(blob, Ave_clust_eval, Ave_blob, Ave, rng, nI, fig, fa):  # fig: i is ig, =0 if fa, nI: comparand in dert

    dert__ = comp_v(blob['dert__'], rng, nI)  # dert: i, g, ?m, dy, dx; or dert_a: i, g, ?m, dy, dx, a, ga, ?ma, day, dax

    if nI == (3,4) or 5: # a+ | ra+ comp forks
        g_crit = 6; m_crit = 7  # Ga, Ma
    else:           # g+ | r+ | ga+ comp forks
        g_crit = 1; m_crit = 2  # G, M

    sub_blob_, Ave_blob = cluster(blob, Ave_blob, Ave, g_crit, fig, fa)  # primary clustering by g or ga

    for sub_blob in sub_blob_:  # evaluate der+ and rng+ sub-clustering forks, rng incr in cluster_eval
        I, G, M = op.itemgetter('I', 'G', 'M')(sub_blob['Dert'])

        if G > Ave_blob + Ave_clust_eval:  # +G > clustering cost (variable) + eval cost (fixed)
            if fig: # comp_g -> redundant g, m sub-clustering forks, eval: comp_a|g per g_sub_blob, comp_i per m_sub_blob

                if G > M + I:
                    cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, g_crit, fig, ~fa)  # g prior, redundant m eval
                    if M + I > Ave_blob + Ave_clust_eval:
                        cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, m_crit, fig, fa)  # parallel cluster by m+i
                else:
                    cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, m_crit, fig, fa)  # m prior, redundant g eval
                    if G > Ave_blob + Ave_clust_eval:
                        cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, g_crit, fig, ~fa)  # parallel cluster by g

            else:  # comp_a or comp_p -> exclusive g_sub_blob_, as in frame_blobs, else g_sub_blob_ is conditional
                cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, g_crit, fig, fa=0) # next comp_g; comp_a if fig only

        elif -G > Ave_blob + Ave_clust_eval:  # exclusive mfork, m and M defined in comp_P
            cluster_eval(sub_blob, Ave_clust_eval, Ave_blob, Ave, rng, m_crit, fig, fa=0)  # eval same-dert r+| ra+, no next a+

    return dert__


def cluster_eval(blob, Ave_clust_eval, Ave_blob, Ave, irng, crit, fig, fa):  # cluster -> sub_blob_, next intra_fork eval

    sub_blob_, Ave_blob = cluster(blob, Ave_blob, Ave, crit, fig, fa)  # conditional sub-clustering by g|m or ga|ma
    for sub_blob in sub_blob_:

        if fa: # comp_a evaluation per g-defined sub_blob, alternating between a and i blob layers

            if sub_blob['Dert'][0] > Ave_blob + ave_intra_fork:  # I > Ave_blob, different for a+?
                Ave_clust_eval += ave_clust_eval
                Ave_blob += ave_blob * rave
                Ave += ave
                intra_fork(sub_blob, Ave_clust_eval, Ave_blob, Ave, (3,4), irng*2+1, fig, fa)  # comp_a fork: fig=0, fa=1

        else:  # comp_i evaluation per g|m-defined sub_blob, crit-defined i, Ave, Ave_blob: rdn-adjusted in intra_fork

            if crit == 1 or 6:
                rng = irng*2 + 1  # der+ comp val = est match of i= Dert[1|6] at rng*2+1, same as rng+ if rng==0
                nI = crit  # g+ | ga+
            else:
                rng = irng + 1    # rng+ comp val = est match of i= Dert[0|5] at rng+1, irng was redefined in a+ fork?
                nI = crit-2  # r+ | ra+: i = 0 if 2 | 5 if 7

            if crit == 2 and fig: Crit = sub_blob['Dert'][0 + 2]
            else: Crit = sub_blob['Dert'][crit]

            if Crit > Ave_blob + ave_intra_fork:
                Ave_clust_eval += ave_clust_eval
                Ave_blob += ave_blob * rave
                Ave += ave
                intra_fork(sub_blob, Ave_clust_eval, Ave_blob, Ave, nI, rng, fig, fa)  # comp_i fork: fa=0


def cluster(blob, Ave_blob, Ave, crit, fig, fa):  # fig = dderived, i is ig, crit: clustering criterion

    if fa:  # comp angle of g | ga, -> ga+ | ra+ forks
        blob['fork_'][0] = dict( I=0, G=0, M=0, Dy=0, Dx=0, A=0, Ga=0, Dyay=0, Dyax=0, Dxay=0, Dxax=0, S=0, Ly=0, sub_blob_=[])
        # initialize root_fork with Dyay, Dxay = Day; Dyax, Dxax = Dax

    else:
        blob['fork_'][0 if crit == 1 or 6 else 1] = dict( I=0, G=0, M=0, Dy=0, Dx=0, S=0, Ly=0, sub_blob_=[] )
        # initialize root_fork at ['fork_'][0|1]: only two forks per blob, then possibly two per +g_sub_blob

    P__ = form_P__(blob['dert__'], Ave, crit, fig)  # horizontal clustering
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)  # vertical clustering
    sub_blob_ = form_blob_(seg_, blob['fork_'][crit], crit)  # feedback to root_fork at blob['fork_'][crit]

    Ave_blob *= len(sub_blob_) / ave_n_sub_blobs

    return sub_blob_, Ave_blob


# functions below are out of date, see Khanh's repo for more recent
#-----------------------------------------------------------------------------------------------------------

def form_P__(x0, y0, dert__, Ave, nI, fg, fa):
    """Form Ps across the whole dert array."""

    if nI == 1:
        crit__ = dert__[1, :, :] - Ave  # der+ crit is gg;  g -> crit (for clustering)
    elif nI == 0:
        crit__ = dert__[2, :, :]  # minimal rng+ crit is m (min |-vg) + I:
        if fg:
            crit__ += dert__[0, :, :]  # + nI magnitude: superset of new m
        crit__ -= Ave
    else:
        crit__ = dert__[5, :, :] - Ave  # ga_der+ crit is ga

    # Clustering:
    s_x_L__ = [*map(
        lambda g_:  # Each line.
        [(sign, next(group)[0], len(list(group)) + 1)  # (s, x, L)
         for sign, group in groupby(enumerate(g_ > 0),
                                    op.itemgetter(1))  # (x, s): return s.
         if sign is not ma.masked],  # Ignore gaps.
        crit__,  # line, blob slice
    )]

    Pderts__ = [[dert_[:, x: x + L].T for s, x, L in s_x_L_]
                for dert_, s_x_L_ in zip(dert__.swapaxes(0, 1), s_x_L__)]

    # Accumulated params:
    # if not fa: I, G, M, Dy, Dx
    # if fa: I, G, M, Dy, Dx, Ga, Dyay, Dyax, Dxay, Dxax
    PDerts__ = map(lambda Pderts_:
                   map(lambda Pderts: Pderts.sum(axis=0),
                       Pderts_),
                   Pderts__)

    param_keys = aP_param_keys if fa else gP_param_keys

    P__ = [
        [
            dict(zip(  # Key-value pairs:
                param_keys,
                [*PDerts, L, x + x0, Pderts, [], [], y, s]
            ))
            for PDerts, Pderts, (s, x, L) in zip(*Pparams_)
        ]
        for y, Pparams_ in enumerate(zip(PDerts__, Pderts__, s_x_L__), start=y0)
    ]

    return P__


def scan_P__(P__):
    """ detect forks and roots per P"""

    for _P_, P_ in pairwise(P__):  # Iterate through pairs of lines.
        _itP_, itP_ = iter(_P_), iter(P_)  # Convert to iterators.
        try:
            _P, P = next(_itP_), next(itP_)  # First pair to check.
        except StopIteration:  # No more fork-root pair.
            continue  # To next pair of _P_, P_.
        while True:
            isleft, olp = comp_edge(_P, P)  # Check for 4 different cases.
            if olp and _P['sign'] == P['sign']:
                _P['root_'].append(P)
                P['fork_'].append(_P)
            try:
                _P, P = (next(_itP_), P) if isleft else (_P, next(itP_))
            except StopIteration:  # No more fork-root pair.
                break  # To next pair of _P_, P_.

    return [*flatten(P__)]  # Flatten P__ before return.


def comp_edge(_P, P):  # Used in scan_P_().
    """
    Check for end-point relative position and overlap.
    Used in scan_P__().
    """
    _x0 = _P['x0']
    _xn = _x0 + _P['L']
    x0 = P['x0']
    xn = x0 + P['L']

    if _xn < xn:  # End-point relative position
        return True, x0 < _xn
    else:
        return False, _x0 < xn


def form_segment_(P_, fa, noM):
    """Form segments of vertically contiguous Ps."""
    # Get a list of every segment's first P:
    P0_ = [*filter(lambda P: (len(P['fork_']) != 1
                              or len(P['fork_'][0]['root_']) != 1),
                   P_)]

    param_keys = aseg_param_keys if fa else gseg_param_keys
    if noM:
        param_keys.remove("M")

    # Form segments:
    seg_ = [dict(zip(param_keys,  # segment's params as keys
                     # Accumulated params:
                     [*map(sum,
                           zip(*map(op.itemgetter(*param_keys[:-6]),
                                    Py_))),
                      len(Py_), Py_[0].pop('y'), Py_,  # Ly, y0, Py_ .
                      Py_[-1].pop('root_'), Py_[0].pop('fork_'),  # root_, fork_ .
                      Py_[0].pop('sign')]))
            # cluster_vertical(P): traverse segment from first P:
            for Py_ in map(cluster_vertical, P0_)]

    for seg in seg_:  # Update segs' refs.
        seg['Py_'][0]['seg'] = seg['Py_'][-1]['seg'] = seg

    for seg in seg_:  # Update root_ and fork_ .
        seg.update(root_=[*map(lambda P: P['seg'], seg['root_'])],
                   fork_=[*map(lambda P: P['seg'], seg['fork_'])])

    for i, seg in enumerate(seg_):  # Remove segs' refs.
        del seg['Py_'][0]['seg']

    return seg_


def cluster_vertical(P):  # Used in form_segment_().
    """
    Cluster P vertically, stop at the end of segment.
    Used in form_segment_().
    """

    if len(P['root_']) == 1 and len(P['root_'][0]['fork_']) == 1:
        root = P.pop('root_')[0]  # Only 1 root.
        root.pop('fork_')  # Only 1 fork.
        root.pop('y')
        root.pop('sign')
        return [P] + cluster_vertical(root)

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
