import operator as op
'''
    intra_blob() evaluates for recursive internal search and clustering: intra_fork() and comp_P(), within each blob.
    Which adds a layer of sub_blobs & sub_forks per blob, with feedback to root_fork, then root_blob, etc.
    2D version of 1st-level algorithm will be a combination of frame_blobs and intra_blob.
    to be added:

    inter_sub_blob() will compare sub_blobs of same range and derivation within higher-level blob, bottom-up ) top-down:
    inter_level() will compare between blob levels, where lower composition level is integrated by inter_sub_blob
    match between levels' edges may form composite blob, axis comp if sub_blobs within blob margin?
    comp_blob() will be 2nd level 2D alg: a prototype for recursive meta-level alg
    
    Input brightness is not correlated with predictive value, but both its stability and variation are: 
    stability: negative gradient deviation (-vvg), is predictive value of initial input, vs. gradient or its derivatives
    variation: positive gradient deviation (+vvg), is predictive value of input gradient, thus its cross-comparison.
    
    -vg is indirect indicator of comp_p rng+ value, lower precision than min: not for g
    vm = min | -vg - Ave: double filter, complimentary to max_g | max_inp, < ave for -vg?
    value of comp_blob (as +vg), then blob-> params value distribution, not for alt_g intra_comps: exclusive?
            
    Blob structure:
    
    root_fork, # = root_blob['fork_'][nI]: reference for feedback of blob' Dert params and sub_blob_, up to frame
    
    Dert = I, G, M, Dy, Dx, Ga, Ma, Day, Dax, S, Ly
    # extended per fork: nI + gDert in ifork, + aDert in afork, S: area, Ly: vert dim, defined by criterion sign
    # I: input, G: gradient, M: match, Dy, Dx: vert,lat Ds, Ga: G of angle, Ma: M of angle, Day, Dax: Ds of angle  
    
    crit, # fork clustering crit: 1 for index of next comp_i comparand in dert and criterion nI in Dert: 0 if I | 1 if G | 5 if Ga 
    sign, # of nI
    rng,  # comp range, per fork?
    map,  # boolean map of blob, to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    dert__, # comp_i inputs
       
    segment_[ seg_params, Py_ [(P_params, dert_)]],  
    # dert: i, g, m, dy, dx, ga, day, dax; no ma: angle_mag != val, ~brightness
    # references down blob formation tree, accumulating Dert, in vertical (horizontal) order
    
    fork_ # multiple derivation trees per blob: 0-1 in g_blobs, 0-3 in a_blobs, next sub_blob_ is in layer_[0]:
        [
         layer_ [(Dert, sub_blob_)]  # alternating g(even) | a(odd) layers across derivation tree, seq access
        ]
        # deeper layers are mixed-fork with nested sub_blob_, Dert params are for layer-parallel comp_blob        
    '''

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
kwidth = 3   # kernel width, not needed?
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

ave_blob = 10000       # fixed cost of intra_cluster per blob, accumulated in deeper layers
rave = 20              # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10   # determines rave, adjusted per intra_cluster
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

''' These filters are accumulated for evaluated intra_fork:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else redefined at each access?
'''

# Other constants
gDert_params = ["I", "G", "M", "Dy", "Dx"]
aDert_params = gDert_params + ["Ga", "Dyay", "Dyax", "Dxay", "Dxax"]

P_params = ["L", "x0", "dert_", "root_", "fork_", "y", "sign"]
seg_params = ["S", "Ly", "y0", "Py_", "root_", "fork_"]

gP_param_keys = gDert_params + P_params
aP_param_keys = aDert_params + P_params

gseg_param_keys = gDert_params + seg_params
aseg_param_keys = aDert_params + seg_params

# -----------------------------------------------------------------------------
# Functions, ALL UNDER REVISION:

def intra_fork(blob, Ave_blob, Ave, rng, nI, fig, fa):  # fig: i is ig;  fa (or nI?) selects comp_i or comp_a

    # comp_i -> dert(i, g, ?m, dy, dx) | comp_a -> dert(i, g, ?m, dy, dx, a, ga, ?ma, day, dax):
    dert__ = comp_i(blob['dert__'], rng, nI, fig, fa)

    if nI == 0 or 1: crit = 1  # r+ or g+ forks
    else: crit = 6  # nI==6: ga+ fork, nI==7: ra+ fork, nI==3: a+ fork, i is (dy,dx), no angle yet?
    gblob_, Ave, Ave_blob = cluster(dert__, blob['fork_'][crit], Ave_blob, Ave, rng, crit, fig, fa)  # cluster by g or ga

    if not fig:  # blobs formed by comp_a or comp_p evaluated for exclusive g+ and r+ forks, same in frame_blobs:
        for blob in gblob_:  # sub-clustering within sub_blobs from default clustering only

            if blob['Dert']['G'] > ave_intra_blob:  # +G blob, exclusive g+ fork; G: input or default clustering
                cluster_eval(blob, Ave_blob, Ave, rng*2+1, 1, fig, ~fa)

            elif -blob['Dert']['G'] > ave_intra_blob:  # -G blob, exclusive r+ fork, m & M def in comp_P
                cluster_eval(blob, Ave_blob, Ave, rng+1, 2, fig, fa)  # not ~fa: eval same-dert r+, ! a+

    elif nI == 0:  # overlapping g+ and r+ forks, as exclusive but for same blob (next: r+, g+, ra+, ga+ eval, a+ if a=dert[5])

        rblob_, Ave, Ave_blob = cluster(dert__, blob['fork_'][2], Ave_blob, Ave, rng, 2, fig, fa)
        # parallel cluster by m+i, rdn forks per blob (no exact gblob x rblob overlap?), eval by root_fork Dert params:

        g_prior = blob['fork_'][2]['Dert']['G'] - ave_intra_blob > 0 and blob['fork_'][2]['Dert']['I'+'M'] - ave_intra_blob

        if g_prior: # root_fork Dert' G val > 0 and > M+I val:
            for sub_gblob in gblob_:
                cluster_eval(sub_gblob, Ave_blob, Ave, rng*2+1, crit, fig, ~fa)  # gblob_ sub_clustering by g, g+ eval

            if blob['fork_'][2]['Dert']['I'+'M'] > ave_intra_blob * 2:  # rdn r+ val > 0, 2+0 in cluster_eval, not ra+
                for sub_rblob in rblob_:
                    cluster_eval(sub_rblob, Ave_blob, Ave, rng+1, 2, fig, fa)  # rblob_ sub_clustering by m+i, r+ eval

        elif blob['fork_'][2]['Dert']['I'+'M'] > ave_intra_blob:  # root_fork Dert' M+I val > 0 and > G val
            for sub_rblob in rblob_:
                cluster_eval(sub_rblob, Ave_blob, Ave, rng+1, 2, fig, fa)  # rblob_ sub_clustering by m+i, r+ eval

            if blob['fork_'][2]['Dert']['G'] > ave_intra_blob * 2:  # redundant g+ val > 0
                for sub_gblob in gblob_:
                    cluster_eval(sub_gblob, Ave_blob, Ave, rng*2+1, crit, fig, ~fa)  # gblob_ sub_clustering by g, g+ eval
        ''' 
        or only exclusive forks: minor gains for r+ in positive gblobs? 
        only two forks, no:  
            I, G, M, Dy, Dx, Ga, Ma, Dyay, Dyax, Dxay, Dxax, Ly, L = blob['Dert'].values()
            rdn = 1
            eval_fork_ = (  # iforks per ablob, prioritized:
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
        '''
    return dert__


def cluster_eval(blob, Ave_blob, Ave, rng, crit, fig, fa):  # sub-cluster -> blob_, eval per blob

    if fa:  # or no fa, same as crit = 3|5: both for comp_a -> exclusive ga+ | ra+ forks?
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


# functions below are out of date

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
