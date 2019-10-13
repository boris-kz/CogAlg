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
    Dert = I, G, Dy, Dx, if fig: += [iDy, iDx, M, if nI == 7 or (4,5): += [A, Ga, Day, Dax]], S, Ly
      
    # extended per fork: nI + gDert in ifork, + aDert in afork, S: area, Ly: vert dim, defined by criterion sign
    # I: input, G: gradient, M: match, packed in I, Dy, Dx: vert,lat Ds, A: angle, Ga: angle G, Day, Dax: angle Ds  
    
    crit, # index of clustering criterion in dert and Dert: g | ga for der+ fork, m | ma for rng+ fork 
    sign, # of crit
    rng,  # comp range, also per fork?
    map,  # boolean map of blob, to compute overlap
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    dert__, # comp_i inputs
       
    segment_[ seg_params, Py_ [(P_params, dert_)]],  # dert = i, g, dy, dx, ?(idy, idx, m, ?(a, ga, day, dax))
    # references down blob formation tree in vertical (horizontal) order, accumulating Dert params
    
    fork_ # 1|2 derivation trees per blob: g,m sub_blob_s per ilayer, g sub_blob_ per alayer
        [
         layer_ [(Dert, sub_blob_)]  # alternating g (even) | a (odd) layers across derivation tree
        ]
        # deeper layers are mixed-fork with nested sub_blob_, Dert params are for layer-parallel comp_blob    
    
    to be added:
    comp_P_: vectorization of critically elongated blobs, by cross-comparing vertically adjacent Ps within each,
    merge_blob_: merge negative summed-value (small) blobs into infra-blob: for comp_blob_ but not intra_blob,
    comp_blob_:  cross-comp of sub_blobs ) blobs of same range and derivation, within root blob ) frame,
    comp_layer_: cross-comp of blob layers, after comp_blob_ within each?
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
# Filters:
ave  = 20  # fixed cost per dert, from average g|m, reflects blob definition cost, may be different for comp_a?
aveB = 10000  # fixed cost per blob: clustering + representation?
aveN = 10  # ave_n_sub_blobs: fixed cost ratio of root_blob / blob: add sub_blobs, adjusted by actual len sub_blob_
aveC = 1000  # ave_clust_eval: cost of eval in cluster_eval,    total cluster_eval cost = Ave_blob + ave_clust_eval
aveF = 1000  # ave_intra_fork: cost of comp + eval in intra_fork, total intra_fork cost = Ave_blob + ave_intra_fork

''' 
kwidth = 3   # kernel width, if starting with 2
if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)

All filters are accumulated in cluster_eval per evaluated fork to account for redundancy: Filter += filter 
Current filters are represented in forks if tree reorder, else redefined at each access? '''

# Other constants

DERT_PARAMS = "I", "G", "Dy", "Dx"
gDERT_PARAMS = DERT_PARAMS + ("iDy", "iDx", "M")
aDERT_PARAMS = gDERT_PARAMS + ("A", "Ga", "Dyay", "Dyax", "Dxay", "Dxax")

P_PARAMS = "L", "x0", "dert_", "root_", "fork_", "y", "sign"
SEG_PARAMS = "S", "Ly", "y0", "x0", "xn", "Py_", "root_", "fork_", "sign"

P_PARAM_KEYS = DERT_PARAMS + P_PARAMS
gP_PARAM_KEYS = gDERT_PARAMS + P_PARAMS
aP_PARAM_KEYS = aDERT_PARAMS + P_PARAMS

SEG_PARAM_KEYS = DERT_PARAMS + SEG_PARAMS
gSEG_PARAM_KEYS = gDERT_PARAMS + SEG_PARAMS
aSEG_PARAM_KEYS = aDERT_PARAMS + SEG_PARAMS

# -----------------------------------------------------------------------------------------------------------------------
# Functions, ALL UNDER REVISION:


def intra_fork(blob, AveF, AveC, AveB, Ave, rng, nI, fig, fa):  # comparand nI: r+ 0, g+ 1, a+ (4,5), ra+ 7, ga+ 8

    dert__ = comp_v(blob['dert__'], rng, nI)  # dert = i, g, dy, dx, ?(idy, idx, m, ?(a, ga, day, dax)):
    # if g+ fork: g, dy, dx = 0
    # if fig: += idy, idx, m  # i is ig
    #    if nI == 7: += a, ga, day, dax
    #    elif nI==(4,5): += a, ga, day, dax = 0
    #    else base dert init?

    if nI == 0 or 1: crit = 1
    else: crit = 8  # a+| ra+
    sub_blob_, AveB = cluster(blob, AveB, Ave, crit, fig, fa)  # primary clustering by g | ga

    for sub_blob in sub_blob_:  # evaluate der+ and rng+ sub-clustering forks, rng incr in cluster_eval
        I, G, M = op.itemgetter('I', 'G', 'M')(sub_blob['Dert'])

        if G > AveB + AveC:  # +G > clustering cost (variable cluster size) + eval cost (fixed layer rep)

            if fig and nI == 0: # rdn sub-clustering by g & m, eval comp_a|g per gsub_blob, comp_i per msub_blob
                                # r+ comp_g: g_crit = 1, m_crit = 6: eval by 0+6; not ra+ | ga+: fig = 0
                if G > M + I:
                    cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, 1, fig, ~fa)  # g+ prior, redundant r+ eval:
                    if M + I > AveB + AveC:
                       cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, 0, fig, fa)  # parallel cluster by i+?m
                else:
                    cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, 0, fig, fa)   # r+ prior, redundant g+ eval:
                    if G > AveB + AveC:
                       cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, 1, fig, ~fa)  # parallel cluster by g

            else:  # comp_a or comp_p -> exclusive g_sub_blob_, as in frame_blobs, else g_sub_blob_ is conditional
                cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, crit, fig, fa)  # fa=0? eval comp_g only

        elif -G > AveB + AveC:  # exclusive mfork eval in neg_g blobs, m and M defined in comp_P
            cluster_eval(sub_blob, AveF, AveC, AveB, Ave, rng, crit, fig, fa)  # fa=0, rp+ | ra+ eval by -crit

    return dert__


def cluster_eval(blob, AveF, AveC, AveB, Ave, irng, crit, fig, fa):  # cluster -> sub_blob_, next intra_fork eval

    sub_blob_, AveB = cluster(blob, AveB, Ave, crit, fig, fa)  # conditional sub-clustering by g|m or ga|ma
    for sub_blob in sub_blob_:

        if fa:  # comp_a evaluation per g-defined sub_blob, alternating between a and i blob layers
            if sub_blob['Dert'][0] > AveB + AveF:  # I > Ave_blob, different for a+?

                AveF += aveF; AveC += aveC; AveB += aveB * aveN; Ave += ave
                intra_fork(sub_blob, AveF, AveC, AveB, Ave, (4,5), irng*2+1, fig, fa)  # fa=1: comp_a fork, fig=0

        else:  # comp_i evaluation per g|m-defined sub_blob, crit-defined i, Ave, Ave_blob: rdn-adjusted in intra_fork

            if crit == 1 or 8: rng = irng*2 + 1  # val = est match of i= g|ga at rng*2+1, same as rng+ if rng==0
            else: rng = irng + 1  # r+| ra+ comp val = est match of i= i | a at rng+1, irng was redefined in a+ fork?

            if crit == 0 and fig: Crit = sub_blob['Dert'][0 + 6]  # r+ fork
            else: Crit = sub_blob['Dert'][crit]  # g+ | ga+ fork

            if Crit > AveB + AveF:
                AveF += aveF; AveC += aveC; AveB += aveB * aveN; Ave += ave
                intra_fork(sub_blob, AveF, AveC, AveB, Ave, crit, rng, fig, fa)  # fa=0: comp_i fork / nI=crit: 0 | 1 | 8

            elif crit==8 and -Crit > AveB + AveF:
                AveF += aveF; AveC += aveC; AveB += aveB * aveN; Ave += ave
                intra_fork(sub_blob, AveF, AveC, AveB, Ave, 7, rng, fig, fa)  # fa=0, comp_i fork is ra+: nI=7?


def cluster(blob, AveB, Ave, crit, fig, fa):  # fig: i=ig, crit: clustering criterion

    # if g+: g, dy, dx = 0; if fig: += idy, idx, m; if nI==7: += a, ga, day, dax; elif nI==(4,5): += a, ga, day, dax = 0; else init dert

    if fa:  # comp angle of g | ga, then eval ga+ | ra+ forks
        blob['fork_'][0] = dict( I=0, G=0, M=0, Dy=0, Dx=0, A=0, Ga=0, Dyay=0, Dyax=0, Dxay=0, Dxax=0, S=0, Ly=0, sub_blob_=[])
        # initialize root_fork with Dyay, Dxay = Day; Dyax, Dxax = Dax  # bPARAMS?  no need for iDy, iDx?
    else:
        blob['fork_'][0 if crit == 1 or 8 else 1] = dict( I=0, G=0, M=0, Dy=0, Dx=0, S=0, Ly=0, sub_blob_=[] )
        # initialize root_fork at ['fork_'][0|1]: only two forks per blob, then possibly two per +g_sub_blob

    P__ = form_P__(blob['dert__'], Ave, crit, fig)  # horizontal clustering, if crit == 0 and fig: crit += dert[6]0+6: if ig r+?
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)  # vertical clustering
    sub_blob_ = form_blob_(seg_, blob['fork_'][crit], crit)  # with feedback to root_fork at blob['fork_'][crit]

    return sub_blob_, AveB * len(sub_blob_) / aveN


# functions below are out of date
#-----------------------------------------------------------------------------------------------------------


def form_P__(dert__, Ave, nI, fig, x0=0, y0=0):  # cluster horizontal ) vertical dert__ into P__

    if fig:
        if nI == 0 or 1: param_keys = gP_PARAM_KEYS  # nI = i-> r+, | g-> g+
        else: param_keys = aP_PARAM_KEYS  # nI = (4,5)-> a+ | 7-> ra+ | 8-> ga+
    else: param_keys = P_PARAM_KEYS

    crit__ = dert__[nI, :, :]  # extract crit__ from dert__ (2D arrays)
    if nI == 0 or 7:  # r+ | ra+ forks
        if fig:
            crit__ = ~crit__[:]  # invert -g sign, r+ only
        else:
            crit__[:] += dert__[6, :, :]
    crit__[:] -= Ave

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

    # Accumulate Dert: if fa: I, G, Dy, Dx, Ga, Dyay, Dyax, Dxay, Dxax; else: I, G, Dy, Dx

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


def form_segment_old(P_, fa, noM):
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


def cluster_vertical(P):  # Used in form_segment_().
    """
    Cluster P vertically, stop at the end of segment
    """
    if len(P['root_']) == 1 and len(P['root_'][0]['fork_']) == 1:
        root = P.pop('root_')[0]  # Only 1 root.
        root.pop('fork_')  # Only 1 fork.
        root.pop('y')
        root.pop('sign')
        return [P] + cluster_vertical(root)  # Plus next P in segment

    return [P]  # End of segment


def form_blob_old(seg_, root_blob, dert___, rng, fork_type):
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


def feedback_old(blob, sub_fork_type=None): # Add each Dert param to corresponding param of recursively higher root_blob.

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