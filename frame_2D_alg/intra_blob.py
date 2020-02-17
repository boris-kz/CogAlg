import operator as op
from collections import deque, defaultdict
from itertools import groupby, starmap, zip_longest
import numpy as np
import numpy.ma as ma
from intra_comp import extend_comp, comp_a, comp_i
from utils import pairwise, flatten
from functools import reduce
'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    
    intra_blob recursively evaluates each blob for two forks of extended internal cross-comp and divisive sub-clustering:
    der+: incremental derivation in high-variation edge areas of +vg: positive deviation of gradient, which triggers comp(g) 
    rng+: incremental comp range in low-variation flat areas of +v--vg: positive deviation of negated -vg, triggers comp(i) at rng+
    
    Each fork adds a layer of sub_blobs and sub_forks per blob, with feedback to root_fork, then root blob, etc.  
    Fork structure:
    fcr,  # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork? 
    fga,  # flag ga: clustering by gradient of angle vs input, | implicit in layer sequence?
    fig,  # flag input param is g
    rdn,  # redundancy to higher layers
    rng,  # comp range
    sub_blob_
    Blob structure, same for all layers of blob hierarchy:

    root_fork,  # reference for feedback of blob Dert params and sub_blob_, up to frame
    Dert = I, G, Dy, Dx; if fig: += [iDy, iDx, M, Ga, Day, Dax], S (area), Ly (vertical dimension)
    # I: input, G: gradient, (Dy, Dx): vert, lat Ds, (iDy, iDx): Ds of input G, M: match, Ga: angle G, Day, Dax: angle Ds  
    sign, 
    map,  # boolean map of blob, to compute overlap in comp_blob
    box,  # boundary box: y0, yn, x0, xn; selective map, box in lower Layers
    dert__, # input: comp i -> i, g, dy, dx, comp_g -> += (idy, idx), m, ga, day, dax 
    stack_[ stack_params, Py_ [(P_params, dert_)]]: refs down blob formation tree, vertical (horizontal)
    gsub_  # der+ fork: [(Dert, sub_blob_(nested)],
    rsub_  # rng+ fork; each sub_ is a list of layers across sub_blob derivation tree, lower layers are mixed-fork
'''
# constants:

DERT_PARAMS = "I", "G", "Dy", "Dx"
gDERT_PARAMS = DERT_PARAMS + ("iDy", "iDx", "M", "Ga", "Dyy", "Dxy", "Dyx", "Dxx")
P_PARAMS = "L", "x0", "dert_", "down_fork_", "up_fork_", "y", "sign"
S_PARAMS = "S", "Ly", "y0", "x0", "xn", "Py_", "down_fork_", "up_fork_", "sign"
P_PARAM_KEYS = DERT_PARAMS + P_PARAMS
gP_PARAM_KEYS = gDERT_PARAMS + P_PARAMS
S_PARAM_KEYS = DERT_PARAMS + S_PARAMS
gS_PARAM_KEYS = gDERT_PARAMS + S_PARAMS

# filters:

ave  = 50  # fixed cost per dert, from average g|m, reflects blob definition cost, may be different for comp_a?
aveB = 10000  # fixed cost per intra_blob comp and clustering
# aveN = 10   # ave_n_sub_blobs: fixed cost ratio of root_blob / blob: add sub_blobs, adjusted by actual len sub_blob_
# aveC = 1000  # ave_clust_eval: cost of eval in cluster_eval,    total cluster_eval cost = Ave_blob + ave_clust_eval
# aveF = 1000  # ave_intra_fork: cost of comp + eval in intra_fork, total intra_fork cost = Ave_blob + ave_intra_fork
''' All filters * rdn '''

# -----------------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:


def intra_blob(root_blob, blob, rdn, rng, fig, fca):  # new version with der+ selection by -ga and cluster_eval

    # dert = i, g, dy, dx, if fig: + [idx, idy, m, ga, day, dax], alternating flag_comp_angle: fca=1 and fca=0 layers:
    if fca:  # comp angle, form ga blobs, evaluate 2x2 blobs for comp_g, 3x3 blobs for comp_rng_i (flag_comp_rng: fcr=1):

        ga_dert__, ra_dert__ = comp_a(blob['dert__'], rng, fig)  # 2x2 + 3x3 comp_a, the 1st call in frame_blobs
        root_blob['gsub_'] = cluster_eval(blob, ga_dert__, 1, rdn, fig, fcr=0, crit=1)  # cluster by 2x2 ga for comp_g eval
        root_blob['rsub_'] = cluster_eval(blob, ra_dert__, rng+1, rdn, fig, fcr=1, crit=(0,6) if fig else 1)  # 3x3 -ga -> comp_ri

    else:  # comp_g | comp_rng, form gblobs, evaluate 2x2 blobs for comp_a, 3x3 blobs for comp_rng_a

        gdert__, rdert__ = comp_i(blob['dert__'], rng, fig)  # 2x2 comp_g + 3x3 comp_g if fig else comp_p
        root_blob['gsub_'] = cluster_eval(blob, gdert__, 1, rdn, fig, fcr=0, crit=7)  # cluster by 2x2 g -> comp_a eval
        root_blob['rsub_'] = cluster_eval(blob, rdert__, rng+1, rdn, fig, fcr=1, crit=7)  # 3x3 -g -> comp_ra: also low-value ga?
    '''
    select comp_rng_a by -g + ga, comp_rng_i by -ga + g: no consistent variation of a or i?  cos g is not available?  
    or no independent comp_rng_a?  
    also cluster_derts(crit=gi): abs_gg (no * cos(da)) -> abs_gblobs, no eval by Gi?
    '''
    # return sub_gblob_, sub_rblob_ or sub_ga_blob_, sub_ra_blob: no need?


def cluster_eval(root_blob, dert__, rng, rdn, fig, fcr, crit):

    deep_sub_ = []  # in cluster_eval? intra_blob recursion extends rsub_ and dsub_ hierarchies by sub_blob_ layer

    blob_ = cluster_derts(root_blob, dert__, rdn, fig, fcr, crit)  # cluster by crit: ga | g | i | i+m? forms 1st layer
    for blob in blob_:  # evaluate blob for der+ | rng+ comp

        if blob['Dert'][crit] > aveB * rdn:  # +|- G|Ga > intra_blob cost:
            lL = len(blob_)
            blob['sub_'] += [[(lL, fig, fcr, rdn, rng, blob_)]]  # 1st layer, fca=1 for odd layers?
            # deep layers compute and feedback:
            blob['sub_'] += intra_blob(root_blob, blob, rdn + 1 + 1 / lL, rng, fig, fca=crit!=7)  # or fca inversion?
            blob['LL'] = len(blob['sub_'])
        else:
            blob['sub_'] = []

        deep_sub_ = [deep_sub + gsub + rsub for deep_sub, gsub, rsub in \
                     zip_longest(deep_sub_, blob['gsub_'], blob['rsub_'], fillvalue=[])]
        # deep_rsub_ and deep_dsub_ are spliced into deep_sub_ hierarchy, fill Dert per layer if n_sub_P > min?

    return deep_sub_  # sub_blob['gsub_'] | sub_blob['rsub_']


def cluster_derts(blob, dert__, rdn, fig, fcr, crit):  # clustering crit is always g in dert[1], fder is a sign

    blob['sub_'][0][0] = dict( I=0, G=0, Dy=0, Dx=0,  # base Dert
                               iDy=0, iDx=0, M=0, Ga=0, Dyy=0, Dxy=0, Dyx=0, Dxx=0,  # extend in comp_g
                               S=0, Ly=0, sub_blob_=[] )  # initialize first sub_blob in first sub_layer

    P__ = form_P__(dert__, ave*rdn, fig, fcr, crit)  # horizontal clustering
    P_ = scan_P__(P__)
    stack_ = form_stack_(P_)  # vertical clustering
    sub_blob_ = form_blob_(stack_, blob['fork_'])  # with feedback to root_fork at blob['fork_']

    return sub_blob_

# clustering functions, out of date:
#---------------------------------------------------------------------------------------------------------------------------------------

def form_P__(dert__, Ave, fig, crit, fcr, x0=0, y0=0):  # cluster dert__ into P__, in horizontal ) vertical order

    # need to add ga if fca?
    if fig:
        param_keys = gP_PARAM_KEYS
        if crit:
            crit__ = dert__[1, :, :] - Ave  # der+ eval by g;  both crit__ and dert__ are 2D arrays
        else:
            crit__ = dert__[0 + 6, :, :] - Ave  # rng+ eval by i + m, accumulated over comp range
    else:
        param_keys = P_PARAM_KEYS
        if crit:
            crit__ = dert__[1, :, :] - Ave  # der+ eval by g
        else:
            crit__ = Ave - dert__[1, :, :]  # rng+ eval by inverted g deviation, accumulated over comp range

    # Cluster dert__ into Pdert__:
    s_x_L__ = [*map(
        lambda crit_: # Each line.
            [(sign, next(group)[0], len(list(group)) + 1)  # (s, x, L)
             for sign, group in groupby(enumerate(crit_ > 0),
                                        op.itemgetter(1)) # (x, s): return s.
             if sign is not ma.masked], # Ignore gaps.
        crit__,  # blob slices in line
    )]
    Pdert__ = [[dert_[:, x : x+L].T for s, x, L in s_x_L_]
                for dert_, s_x_L_ in zip(dert__.swapaxes(0, 1), s_x_L__)]

    # Accumulate Dert = P param_keys:
    PDert__ = map(lambda Pdert_:
                       map(lambda Pdert_: Pdert_.sum(axis=0),
                           Pdert_),
                   Pdert__)
    P__ = [
        [
            dict(zip( # Key-value pairs:
                param_keys,
                [*PDerts, L, x+x0, Pderts, [], [], y, s]  # why Pderts?
            ))
            for PDerts, Pderts, (s, x, L) in zip(*Pparams_)
        ]
        for y, Pparams_ in enumerate(zip(PDert__, Pdert__, s_x_L__), start=y0)
    ]

    return P__


def scan_P__(P__):
    """Detect up_forks and down_forks per P."""

    for _P_, P_ in pairwise(P__):  # Iterate through pairs of lines.
        _iter_P_, iter_P_ = iter(_P_), iter(P_)  # Convert to iterators.
        try:
            _P, P = next(_iter_P_), next(iter_P_)  # First pair to check.
        except StopIteration:  # No more up_fork-down_fork pair.
            continue  # To next pair of _P_, P_.
        while True:
            isleft, olp = comp_edge(_P, P)  # Check for 4 different cases.
            if olp and _P['sign'] == P['sign']:
                _P['down_fork_'].append(P)
                P['up_fork_'].append(_P)
            try:  # Check for stopping:
                _P, P = (next(_iter_P_), P) if isleft else (_P, next(iter_P_))
            except StopIteration:  # No more up_fork - down_fork pair.
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

    if _xn < xn:  # End-point relative position.
        return True, x0 < _xn  # Overlap.
    else:
        return False, _x0 < xn


def form_stack_(P_, fig):
    """Form segments of vertically contiguous Ps."""
    # Get a list of every segment's first P:
    P0_ = [*filter(lambda P: (len(P['up_fork_']) != 1
                              or len(P['up_fork_'][0]['down_fork_']) != 1),
                   P_)]

    if fig:
        param_keys = gS_PARAM_KEYS
    else:
        param_keys = S_PARAM_KEYS

    # Form segments:
    seg_ = [dict(zip(param_keys,  # segment's params as keys
                     # Accumulated params:
                     [*map(sum,
                           zip(*map(op.itemgetter(*param_keys[:-6]),
                                    Py_))),
                      len(Py_), Py_[0].pop('y'), Py_,  # Ly, y0, Py_ .
                      Py_[-1].pop('down_fork_'), Py_[0].pop('up_fork_'),  # down_fork_, up_fork_ .
                      Py_[0].pop('sign')]))
            # cluster_vertical(P): traverse segment from first P:
            for Py_ in map(cluster_vertical, P0_)]

    for seg in seg_:  # Update segs' refs.
        seg['Py_'][0]['seg'] = seg['Py_'][-1]['seg'] = seg

    for seg in seg_:  # Update down_fork_ and up_fork_ .
        seg.update(down_fork_=[*map(lambda P: P['seg'], seg['down_fork_'])],
                   up_fork_=[*map(lambda P: P['seg'], seg['up_fork_'])])

    for i, seg in enumerate(seg_):  # Remove segs' refs.
        del seg['Py_'][0]['seg']

    return seg_


def form_stack_(P_, fa):
    """Form segments of vertically contiguous Ps."""
    # Determine params type:
    if "M" not in P_[0]:
        seg_param_keys = (*aSEG_PARAM_KEYS[:2], *aSEG_PARAM_KEYS[3:])
        Dert_keys = (*aDERT_PARAMS[:2], *aDERT_PARAMS[3:], "L")
    elif fa:  # segment params: I G M Dy Dx Ga Dyay Dyax Dxay Dxax S Ly y0 Py_ down_fork_ up_fork_ sign
        seg_param_keys = aSEG_PARAM_KEYS
        Dert_keys = aDERT_PARAMS + ("L",)
    else:  # segment params: I G M Dy Dx S Ly y0 Py_ down_fork_ up_fork_ sign
        seg_param_keys = gSEG_PARAM_KEYS
        Dert_keys = gDERT_PARAMS + ("L",)

    # Get a list of every segment's top P:
    P0_ = [*filter(lambda P: (len(P['up_fork_']) != 1
                              or len(P['up_fork_'][0]['down_fork_']) != 1),
                   P_)]

    # Form segments:
    seg_ = [dict(zip(seg_param_keys,  # segment's params as keys
                     # Accumulated params:
                     [*map(sum,
                           zip(*map(op.itemgetter(*Dert_keys),
                                    Py_))),
                      len(Py_), Py_[0].pop('y'),  # Ly, y0
                      min(P['x0'] for P in Py_),
                      max(P['x0'] + P['L'] for P in Py_),
                      Py_,  # Py_ .
                      Py_[-1].pop('down_fork_'), Py_[0].pop('up_fork_'),  # down_fork_, up_fork_ .
                      Py_[0].pop('sign')]))
            # cluster_vertical(P): traverse segment from first P:
            for Py_ in map(cluster_vertical, P0_)]

    for seg in seg_:  # Update segs' refs.
        seg['Py_'][0]['seg'] = seg['Py_'][-1]['seg'] = seg

    for seg in seg_:  # Update down_fork_ and up_fork_ .
        seg.update(down_fork_=[*map(lambda P: P['seg'], seg['down_fork_'])],
                   up_fork_=[*map(lambda P: P['seg'], seg['up_fork_'])])

    for i, seg in enumerate(seg_):  # Remove segs' refs.
        del seg['Py_'][0]['seg']

    return seg_


def cluster_vertical(P):  # Used in form_segment_().
    """
    Cluster P vertically, stop at the end of segment
    """
    if len(P['down_fork_']) == 1 and len(P['down_fork_'][0]['up_fork_']) == 1:
        down_fork = P.pop('down_fork_')[0]  # Only 1 down_fork.
        down_fork.pop('up_fork_')  # Only 1 up_fork.
        down_fork.pop('y')
        down_fork.pop('sign')
        return [P] + cluster_vertical(down_fork)  # Plus next P in segment

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
            for ext_seg in blob_seg['up_fork_'] + blob_seg['down_fork_']:
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
            xn = max(xn, max(map(lambda P: P['x0'] + P['L'], blob_seg['Py_'])))

        y0 = min(map(op.itemgetter('y0'), blob_seg_))
        yn = max(map(lambda segment: segment['y0'] + segment['Ly'], blob_seg_))

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
            rng=rng,
            dert___=dert___,
            mask=mask,
            root_blob=root_blob,
            hDerts=np.concatenate(
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
    if 'M' not in seg_[0]:  # No M.
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
    blob_seg__ = []  # list of blob's segment lists.
    owned_seg_ = []  # list blob-owned segments.

    # Iterate through list of all segments:
    for seg in seg_:
        # Check whether seg has already been owned:
        if seg in owned_seg_:
            continue  # Skip.

        blob_seg_ = [seg]  # Initialize list of blob segments.
        q_ = deque([seg])  # Initialize queue of filling segments.

        while len(q_) > 0:
            blob_seg = q_.pop()
            for ext_seg in blob_seg['up_fork_'] + blob_seg['down_fork_']:
                if ext_seg not in blob_seg_:
                    blob_seg_.append(ext_seg)
                    q_.append(ext_seg)
                    ext_seg.pop("sign")  # Remove all blob_seg's signs except for the first one.

        owned_seg_ += blob_seg_
        blob_seg__.append(blob_seg_)

    return blob_seg__


def feedback_old(blob, sub_fork_type=None):  # Add each Dert param to corresponding param of recursively higher root_blob.

    root_blob = blob['root_blob']
    if root_blob is None:  # Stop recursion.
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
        [*starmap(  # Like map() except for taking multiple arguments.
            # Function (with multiple arguments):
            lambda Dert, sDert:
            (*starmap(op.add, zip(Dert, sDert)),),  # Dert and sub_blob_ accum
            # Mapped iterables:
            zip(
                root_blob['forks'][fork_type][1:],
                blob['forks'][sub_fork_type][:],
            ),
        )]
    # Dert-only numpy.ndarray equivalent: (no sub_blob_ accumulation)
    # root_blob['forks'][fork_type][1:] += blob['forks'][fork_type]

    feedback(root_blob, fork_type)


def feedback(blob, fork=None):  # Add each Dert param to corresponding param of recursively higher root_blob.

    root_fork = blob['root_fork']

    # fork layers is deeper than root_fork Layers:
    if fork is not None:
        if len(root_fork['layer_']) <= len(fork['layer_']):
            root_fork['layer_'].append((
                dict(zip(  # Dert
                    root_fork['layer_'][0][0],  # keys
                    repeat(0),  # values
                )),
                [],  # blob_
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
    if root_fork['root_blob'] is not None:  # Stop recursion if false.
        feedback(root_fork['root_blob'])

'''

def intra_full_g(blob, rdn, rng, fig, fder):  # version of frame_blobs with sub-clustering per recursive extended comp

    deep_sub_ = []  # each intra_blob recursion extends rsub_ and dsub_ hierarchies by sub_blob_ layer
     
    # dert = i, g, dy, dx, if fig: += [idx, idy, m, ga, day, dax]; comp_g select by ga, directional resolution loss?
    # comp_g: comp_a -> da -> ga, day, dax; dg = g - _g * cos(da), combine in gg
    
    gdert__, rdert__ = extend_comp(blob['dert__'], rng, fig)  # 2x2 and 3x3 comp, same dert structure

    sub_gblob_ = cluster_derts(blob, gdert__, rdn, True, fder=1)
    for sub_gblob in sub_gblob_:  # evaluate blob for der+ fork, semi-exclusive with rng+:

        if sub_gblob['Dert']['G'] > aveB * rdn:  # +G > intra_blob cost, | Ga:
            lL = len(sub_gblob_)
            blob['gsub_'] += [[(lL, True, True, rdn, rng, sub_gblob_)]]  # 1st layer: lL, fig, fder, rdn, rng
            blob['gsub_'] += intra_blob(blob, rdn + 1 + 1 / lL, rng=1, fig=1, fder=1)  # deep layers feedback
            blob['gLL'] = len(blob['gsub_'])

    sub_rblob_ = cluster_derts(blob, rdert__, rdn, fig, fder=0)
    for sub_rblob in sub_rblob_:  # evaluate blob for rng+ fork, semi-exclusive with der+:

        if -sub_rblob['Dert']['G'] > aveB * rdn:  # -G > intra_blob cost
            lL = len(sub_rblob_)
            blob['rsub_'] += [[(lL, fig, False, rdn, rng, sub_rblob_)]]  # 1st layer: lL, fig, fder, rdn, rng
            blob['rsub_'] += intra_blob(blob, rdn + 1 + 1 / lL, rng+1, fig, fder=0)  # deep layers feedback
            blob['rLL'] = len(blob['gsub_'])
            
            # evaluate overlaps among selected blobs of different forks, with additional rdn?
            
    deep_sub_ = [deep_sub + gsub + rsub for deep_sub, gsub, rsub in \
                 zip_longest(deep_sub_, blob['gsub_'], blob['rsub_'], fillvalue=[])]
    # deep_rsub_ and deep_dsub_ are spliced into deep_sub_ hierarchy, fill Dert per layer if n_sub_P > min?

    return deep_sub_


    for sub_ablob in sub_ablob_:
        if -sub_ablob['Dert']['Ga'] > aveB * rdn:  # -Ga > intra_blob cost
            sub_ablob_ = intra(sub_ablob, adert__, rng, rdn, fig, ~ga)  # comp_g
    if fig:
        sub_ablob_ = cluster_derts(blob, gdert__, rdn, True, fder=1, fga=1)  # sub-cluster by ga
        for sub_ablob in sub_ablob_:  # evaluate blob for der+ fork, semi-exclusive with rng+:
            if -sub_ablob['Dert']['Ga'] > aveB * rdn:
                form_sub_(g)  # not form_sub_(r): no ga computed
            else: 
                form_sub_(ga): from 2x2 ga, also parallel 3x3 and 2x2 compute?     

    # initialization before accumulation, Dert only?
    if fa:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'M'=0, 'Dy'=0, 'Dx'=0, 'Ga'=0, 'Day'=0, 'Dax'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0  # (g, gg, m, dy, dx) -> (g, gg, m, dy, dx, ga, day, dax)
    else:
        blob['Dert'] = 'G'=0, 'Gg'=0, 'Dy'=0, 'Dx'=0, 'L'=0, 'Ly'=0
        dert___[:] = dert___[:][:], 0, 0, 0, 0  # g -> (g, gg, m, dy, dx)

    kwidth = 3  # kernel width, if starting with 2
    if kwidth != 2:  # ave is an opportunity cost per comp:
    ave *= (kwidth ** 2 - 1) / 2  # ave *= ncomp_per_kernel / 2 (base ave is for ncomp = 2 in 2x2)
    
    no 1x1 comp_a: segment per input vs. comparand
    separate comp_a, comp_g in comp_P: known oriented, low value per kernel: not dir selective?
    ?(idy, idx, m, ?(a, ga, day, dax)); newI: index in dert: 0 if r+, 1 if g+, (4,5) if a+, 7 if ra+, 8 if ga+
'''