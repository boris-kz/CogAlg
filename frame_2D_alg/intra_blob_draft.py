import operator as op
from collections import deque, defaultdict
from itertools import groupby, starmap, zip_longest, repeat, accumulate, chain, starmap, tee
import numpy as np
import numpy.ma as ma
from intra_comp import *
from utils import pairwise, flatten
from functools import reduce
'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    
    intra_blob recursively evaluates each blob for one of three forks of extended internal cross-comparison and sub-clustering:
    angle cross-comp,
    der+: incremental derivation cross-comp in high-variation edge areas of +vg: positive deviation of gradient triggers comp_g, 
    rng+: incremental range cross-comp in low-variation flat areas of +v--vg: positive deviation of negated -vg triggers comp_r.
    Each adds a layer of sub_blobs per blob.  
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_forking_scheme.png
    
    Blob structure, for all layers of blob hierarchy:
    
    root,  # reference to root blob, for feedback of blob Dert params and sub_blob_, up to frame
    Dert = I, iDy, iDx, G, Dy, Dx, M, S (area), Ly (vertical dimension)
    # I: input, (iDy, iDx): angle of input gradient, if any, G: gradient, (Dy, Dx): vertical and lateral Ds, M: match  
    sign, 
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, idy, idx, g, dy, dx, m
    stack_[ stack_params, Py_ [(P_params, dert_)]]: refs down blob formation tree, in vertical (horizontal) order
    
    # fork structure of next layer:
    fcr, # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork? 
    fig, # flag input is gradient
    rdn, # redundancy to higher layers
    rng, # comp range
    sub_layers  # [(Dert, sub_blobs)]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
# filters, All *= rdn:

ave  = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 10000  # fixed cost per intra_blob comp and clustering

# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def intra_blob(blob, rdn, rng, fig, fcr):  # recursive input rng+ | der+ cross-comp within blob
    # fig: flag input is g, fcr: flag comp over rng+

    if fcr: dert__ = comp_r(blob['dert__'], fig, blob['root']['fcr'])  #-> m sub_blobs
    else:   dert__ = comp_g(blob['dert__'])  #-> g sub_blobs:

    cluster_derts(blob, dert__, ave*rdn, fcr, fig)
    # feedback: root['layer_'] += [[(lL, fig, fcr, rdn, rng, blob['sub_blob_'])]]  # 1st layer

    for sub_blob in blob['blob_']:  # eval intra_blob comp_g | comp_rng if low gradient
        if sub_blob['sign']:
            if sub_blob['Dert']['M'] > aveB * rdn:  # -> comp_r:
                intra_blob(sub_blob, rdn + 1, rng**2, fig=fig, fcr=1)  # rng=1 in first call

        elif sub_blob['Dert']['G'] > aveB * rdn:
            intra_blob(sub_blob, rdn + 1, rng=rng, fig=1, fcr=0)  # -> comp_g
    '''
    feedback:
    for sub_blob in blob['blob_']:
        blob['layer_'] += intra_blob(sub_blob, rdn + 1 + 1 / lA, rng, fig, fcr) 
    '''

def cluster_derts(blob, dert__, Ave, fcr, fig):  # analog of frame_to_blobs
    # clustering criterion per fork:

    if fcr:  # comp_r output
        if fig: crit__ = dert__[:, :, 0] + dert__[:, :, 4] - Ave  # eval by i + m, accumulated in rng
        else:   crit__ = Ave - dert__[:, :, 1]  # eval by -g, accumulated in rng
    else:  # comp_g output
        crit__ = dert__[:, :, 4] - Ave  # comp_g output eval by m, or clustering is always by m?

    height, width = dert__.shape[1:]
    dert__ = ma.transpose(dert__, axes=(1, 2, 0))  # transpose dert__ into shape [y,x,params]
    stack_ = deque()  # buffer of running vertical stacks of Ps

    for y in range(height):  # first and last row are discarded

        print(f'Processing line {y}...')
        P_ = form_P_(dert__[y, :], crit__[y, :])  # horizontal clustering, adds a row of Ps
        P_ = scan_P_(P_, stack_, blob['root'])    # vertical clustering, adds up_forks per P and down_fork_cnt per stack
        stack_ = form_stack_(P_, blob['root'], y)

    while stack_:  # frame ends, last-line stacks are merged into their blobs:
        form_blob(stack_.popleft(), blob['root'])

    # return sub_blob_  # not needed, feedback to root is in form_blob?


# clustering functions:
#-------------------------------------------------------------------------------------------------------------------

def form_P_(dert_, crit_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    mask_ = dert_[:,0].mask
    sign_ = crit_ > 0
    x0 = -1
    for x in range(len(dert_)):
        if ~mask_[x]:
            x0 = x  # coordinate of first unmasked dert in line
            break
    I, iDy, iDx, G, Dy, Dx, M, L = *dert_[x0], 1  # initialize P params
    _sign = sign_[x0]
    _mask = False  # mask bit per dert

    for x in range(x0+1, dert_.shape[0]):  # loop left to right in each row of derts
        sign = sign_[x]
        mask = mask_[x]
        if (~_mask and mask) or sign != _sign:
            # (P exists and input is not in blob) or sign changed, terminate and pack P:
            P = dict(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
            P_.append(P)
            # initialize P params:
            I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x

        if ~mask:  # accumulate P params:
            I += dert_[x][0]
            iDy += dert_[x][1]
            iDx += dert_[x][2]
            G += dert_[x][3]
            Dy += dert_[x][4]
            Dx += dert_[x][5]
            M += dert_[x][6]
            L += 1
            _sign = sign  # prior sign
        _mask = mask

    # terminate and pack last P in a row
    P = dict(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
    P_.append(P)

    return P_


def scan_P_(P_, stack_, blob_root):  # merge P into higher-row stack of Ps which have same sign and overlap by x_coordinate

    next_P_ = deque()  # to recycle P + up_fork_ that finished scanning _P, will be converted into next_stack_

    if P_ and stack_:  # if both input row and higher row have any Ps / _Ps left

        P = P_.popleft()          # load left-most (lowest-x) input-row P
        stack = stack_.popleft()  # higher-row stacks
        _P = stack['Py_'][-1]     # last element of each stack is higher-row P
        up_fork_ = []             # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P['x0']         # first x in P
            xn = x0 + P['L']     # first x beyond P
            _x0 = _P['x0']       # first x in _P
            _xn = _x0 + _P['L']  # first x beyond _P

            if (P['sign'] == stack['sign']
                    and _x0 < xn and x0 < _xn):  # test for sign match and x overlap between loaded P and _P
                stack['down_fork_cnt'] += 1
                up_fork_.append(stack)  # P-connected higher-row stacks are buffered into up_fork_ per P

            if xn < _xn:  # _P overlaps next P in P_
                next_P_.append((P, up_fork_))  # recycle _P for the next run of scan_P_
                up_fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack['down_fork_cnt'] != 1:  # terminate stack, merge it into up_forks' blobs
                        form_blob(stack, blob_root)
                    break
            else:  # no next-P overlap
                if stack['down_fork_cnt'] != 1:  # terminate stack, merge it into up_forks' blobs
                    form_blob(stack, blob_root)
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack['Py_'][-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_fork_))
                    break

    while P_:  # terminate Ps and stacks that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_fork
    while stack_:
        form_blob(stack_.popleft(), blob_root)  # down_fork_cnt always == 0

    return next_P_  # each element is P + up_fork_ refs


def comp_end(_P, P):  # Check for end-point relative position and overlap

    _x0 = _P['x0']
    _xn = _x0 + _P['L']
    x0 = P['x0']
    xn = x0 + P['L']

    if _xn < xn:  # End-point relative position.
        return True, x0 < _xn  # Overlap.
    else:
        return False, _x0 < xn

# constants:

iPARAMS = "I", "G", "Dy", "Dx", "M"  # formed by comp_pixel
gPARAMS = iPARAMS + ("iDy", "iDx")  # angle of input g

P_PARAMS = "L", "x0", "dert_", "down_fork_", "up_fork_", "y", "sign"
S_PARAMS = "A", "Ly", "y0", "x0", "xn", "Py_", "down_fork_", "up_fork_", "sign"

P_PARAM_KEYS = iPARAMS + P_PARAMS
gP_PARAM_KEYS = gPARAMS + P_PARAMS
S_PARAM_KEYS = iPARAMS + S_PARAMS
gS_PARAM_KEYS = gPARAMS + S_PARAMS


# old -------------------------------------------------------------------------------------------------------:

def form_P__khanh(dert__, Ave, x0=0, y0=0):  # cluster dert__ into P__, in horizontal ) vertical order

    crit__ = Ave - dert__[-1, :, :]  # eval by inverse deviation of g
    param_keys = P_PARAM_KEYS  # comp_g output params

    # Cluster dert__ into Pdert__:
    s_x_L__ = [*map(
        lambda crit_:  # Each line
            [(sign, next(group)[0], len(list(group)) + 1)  # (s, x, L)
             for sign, group in groupby(enumerate(crit_ > 0),
                                        op.itemgetter(1))  # (x, s): return s
             if sign is not ma.masked],  # Ignore gaps.
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

def form_stack_(P_, fig):
    """Form stacks of vertically contiguous Ps."""
    # list of first Ps in stacks:
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


def form_blob(seg_, root_fork):
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
    
    dert_=dert__[:, y, x0:x0 + L]
    """
    if root_fork['root_blob'] is not None:  # Stop recursion if false.
        feedback(root_fork['root_blob'])
