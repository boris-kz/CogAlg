'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    
    der+: incremental derivation cross-comp in high-variation edge areas of +vg: positive deviation of gradient triggers comp_g, 
    rng+: incremental range cross-comp in low-variation flat areas of +v--vg: positive deviation of negated -vg triggers comp_r.
    Each adds a layer of sub_blobs per blob.  
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_2_fork_scheme.png
    
    Blob structure, for all layers of blob hierarchy:
    root_dert__,  
    Dert = I, iDy, iDx, G, Dy, Dx, M, S (area), Ly (vertical dimension)
    # I: input, (iDy, iDx): angle of input gradient, G: gradient, (Dy, Dx): vertical and lateral Ds, M: match  
    sign, 
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, idy, idx, g, dy, dx, m
    stack_[ stack_params, Py_ [(P_params, dert_)]]: refs down blob formation tree, in vertical (horizontal) order
    # next fork:
    fcr,  # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork? 
    fig,  # flag input is gradient
    rdn,  # redundancy to higher layers
    rng,  # comp range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
from collections import deque, defaultdict
from class_cluster import ClusterStructure, NoneType
from class_bind import AdjBinder
from frame_blobs_yx import assign_adjacents
from intra_comp_g import comp_g, comp_r
from itertools import zip_longest
from class_stream import BlobStreamer
from utils import pairwise
import numpy as np
# from comp_P_draft import comp_P_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering

class CDeepP(ClusterStructure):
    I = int
    G = int
    Dy = int
    Dx = int
    M = int
    iDy = int
    iDx = int
    L = int
    x0 = int
    sign = NoneType

class CDeepStack(ClusterStructure):
    I = int
    G = int
    Dy = int
    Dx = int
    M = int
    iDy = int
    iDx = int
    S = int
    Ly = int
    y0 = int
    Py_ = list
    blob = object
    down_connect_cnt = int
    sign = NoneType

class CDeepBlob(ClusterStructure):
    Dert = dict
    box = list
    stack_ = list
    sign = NoneType
    open_stacks = int
    root_dert__ = object
    dert__ = object
    mask = object
    adj_blobs = list
    fopen = bool
    margin = list
    fcr = bool
    fig = bool
    rdn = float
    rng = int
    Ls = int  # for visibility and next-fork rdn
    sub_layers = list

# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def intra_blob(blob, rdn, rng, fig, fcr, **kwargs):  # recursive input rng+ | der+ cross-comp within blob
    # fig: flag input is g | p, fcr: flag comp over rng+ | der+
    if kwargs.get('render', None) is not None:  # stop rendering sub-blobs when blob is too small
        if blob.Dert['S'] < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)
    if fcr:
        dert__, mask = comp_r(ext_dert__, fig, fcr, ext_mask)  # -> m sub_blobs
    else:
        dert__, mask = comp_g(ext_dert__, ext_mask)  # -> g sub_blobs:

    if dert__[0].shape[0] > 2 and dert__[0].shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
        sub_blobs = cluster_derts(dert__, mask, ave * rdn, fcr, fig, **kwargs)
        # fork params:
        blob.fcr = fcr
        blob.fig = fig
        blob.rdn = rdn
        blob.rng = rng
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate for intra_blob comp_g | comp_r:

            G = blob.Dert['G']; adj_G = blob.adj_blobs[2]
            borrow = min(abs(G), abs(adj_G) / 2)  # or adjacent M if negative sign?

            if sub_blob.sign:
                if sub_blob.Dert['M'] - borrow > aveB * rdn:  # M - (intra_comp value lend to edge blob)
                    # comp_r fork:
                    blob.sub_layers += intra_blob(sub_blob, rdn + 1 + 1 / blob.Ls, rng * 2, fig=fig, fcr=1, **kwargs)
                # else: comp_P_
            elif sub_blob.Dert['G'] + borrow > aveB * rdn:  # G + (intra_comp value borrow from flat blob)
                # comp_g fork:
                blob.sub_layers += intra_blob(sub_blob, rdn + 1 + 1 / blob.Ls, rng=rng, fig=1, fcr=0, **kwargs)
            # else: comp_P_

        spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                          zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]
    return spliced_layers


def cluster_derts(dert__, mask, Ave, fcr, fig, render=False):  # similar to frame_to_blobs

    if fcr:  # comp_r output;  form clustering criterion:
        if fig:
            crit__ = dert__[0] + dert__[6] - Ave  # eval by i + m, accum in rng; dert__[:,:,0] if not transposed
        else:
            crit__ = Ave - dert__[3]  # eval by -g, accum in rng
    else:  # comp_g output
        crit__ = dert__[6] - Ave  # comp_g output eval by m, or clustering is always by m?

    root_dert__ = dert__  # derts after the comps operation, which is the root_dert__
    dert__ = [*zip(*dert__)]  # transpose dert__ into shape [y, params, x]

    sub_blobs = []  # from form_blob:
    stack_ = deque()  # buffer of running vertical stacks of Ps
    stack_binder = AdjBinder(CDeepStack)
    if render:
        streamer = BlobStreamer(CDeepBlob, crit__, mask)

    if render:
        streamer = BlobStreamer(CDeepBlob, crit__, mask)
    for y, dert_ in enumerate(dert__):  # in height, first and last row are discarded;  print(f'Processing intra line {y}...')
        # if False in mask[i]:  # [y,x,params], there is at least one dert in line
        P_binder = AdjBinder(CDeepP)  # binder needs data about clusters of the same level
        P_ = form_P_(zip(*dert_), crit__[y], mask[y], P_binder)  # horizontal clustering, adds a row of Ps
        if render:
            render = streamer.update_blob_conversion(y, P_)  # if return False, stop rendering
        P_ = scan_P_(P_, stack_, root_dert__, sub_blobs, P_binder)  # vertical clustering, adds up_connects per P and down_connect_cnt per stack
        stack_ = form_stack_(P_, root_dert__, sub_blobs, y)
        stack_binder.bind_from_lower(P_binder)

    while stack_:  # frame ends, last-line stacks are merged into their blobs:
        form_blob(stack_.popleft(), root_dert__, sub_blobs)

    blob_binder = AdjBinder(CDeepBlob)
    blob_binder.bind_from_lower(stack_binder)
    assign_adjacents(blob_binder)  # add adj_blobs to each blob
    # sub_blobs = find_adjacent(sub_blobs)

    if render:  # rendering mode after blob conversion
        streamer.end_blob_conversion(y)

    return sub_blobs


# clustering functions:
# -------------------------------------------------------------------------------------------------------------------

def form_P_(dert_, crit_, mask_, binder):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    sign_ = crit_ > 0
    x0 = 0
    try:
        while mask_[x0]:  # skip until not masked
            next(dert_)
            x0 += 1
    except IndexError:
        return P_  # the whole line is masked, return an empty P

    I, iDy, iDx, G, Dy, Dx, M, L = *next(dert_), 1  # initialize P params
    _sign = sign_[x0]
    _mask = mask_[x0]  # mask bit per dert

    for x, (i, idy, idx, g, dy, dx, m) in enumerate(dert_, start=x0+1):  # loop left to right in each row of derts
        mask = mask_[x]
        if ~mask:  # current dert is not masked
            sign = sign_[x]
            if ~_mask and sign != _sign:  # prior dert is not masked and sign changed
                # pack P
                P = CDeepP(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L,x0=x0, sign=_sign)
                P_.append(P)
                # initialize P params:
                I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x
            elif _mask:
                I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x
        # current dert is masked
        elif ~_mask:  # prior dert is not masked
            # pack P
            P = CDeepP(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
            P_.append(P)
            # initialize P params: (redundant)
            # I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x + 1

        if ~mask:  # accumulate P params:
            I += i
            iDy += idy
            iDx += idx
            G += g
            Dy += dy
            Dx += dx
            M += m
            L += 1
            _sign = sign  # prior sign
        _mask = mask

    if ~_mask:  # terminate and pack last P in a row if prior dert is unmasked
        P = CDeepP(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
        P_.append(P)

    for _P, P in pairwise(P_):
        if _P.x0 + _P.L == P.x0:  # check if Ps are adjacents
            binder.bind(_P, P)

    return P_


def scan_P_(P_, stack_, root_dert__, sub_blobs, binder):  # merge P into higher-row stack of Ps with same sign and x_coord overlap

    next_P_ = deque()  # to recycle P + up_connect_ that finished scanning _P, will be converted into next_stack_

    if P_ and stack_:  # if both input row and higher row have any Ps / _Ps left

        P = P_.popleft()  # load left-most (lowest-x) input-row P
        stack = stack_.popleft()  # higher-row stacks
        _P = stack.Py_[-1]  # last element of each stack is higher-row P
        up_connect_ = []  # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P.x0  # first x in P
            xn = x0 + P.L  # first x beyond P
            _x0 = _P.x0  # first x in _P
            _xn = _x0 + _P.L  # first x beyond _P

            if stack.G > 0:  # check for overlaps in 8 directions, else a blob may leak through its external blob
                if _x0 - 1 < xn and x0 < _xn + 1:  # x overlap between loaded P and _P
                    if P.sign == stack.sign:  # sign match
                        stack.down_connect_cnt += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_
                    else:
                        binder.bind(_P, P)

            else:  # -G, check for orthogonal overlaps only: 4 directions, edge blobs are more selective
                if _x0 < xn and x0 < _xn:  # x overlap between loaded P and _P
                    if P.sign == stack.sign:  # sign match
                        stack.down_connect_cnt += 1
                        up_connect_.append(stack)  # buffer P-connected higher-row stacks into P' up_connect_
                    else:
                        binder.bind(_P, P)

            if (xn < _xn or  # _P overlaps next P in P_
                    xn == _xn and stack.sign):  # sign taken accounted
                next_P_.append((P, up_connect_))  # recycle _P for the next run of scan_P_
                up_connect_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack.down_connect_cnt != 1:  # terminate stack, merge it into up_connects' blobs
                        form_blob(stack, root_dert__, sub_blobs)
                    break
            else:  # no next-P overlap
                if stack.down_connect_cnt != 1:  # terminate stack, merge it into up_connects' blobs
                    form_blob(stack, root_dert__, sub_blobs)
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack.Py_[-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_connect_))
                    break

    while P_:  # terminate Ps and stacks that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_connect
    while stack_:
        form_blob(stack_.popleft(), root_dert__, sub_blobs)  # down_connect_cnt always == 0

    return next_P_  # each element is P + up_connect_ refs


def form_stack_(P_, root_dert__, sub_blobs, y):  # Convert or merge every P into its stack of Ps, merge blobs

    next_stack_ = deque()  # converted to stack_ in the next run of scan_P_

    while P_:
        P, up_connect_ = P_.popleft()
        I, G, Dy, Dx, M, iDy, iDx, L, x0, s = P.unpack()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row:
            blob = CDeepBlob(Dert=dict(I=0, G=0, Dy=0, Dx=0, M=0, iDy=0, iDx=0, S=0, Ly=0),
                             box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = CDeepStack(I=I, G=G, Dy=0, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=L, Ly=1,
                                   y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
            new_stack.hid = blob.id
            blob.stack_.append(new_stack)
        else:
            if len(up_connect_) == 1 and up_connect_[0].down_connect_cnt == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                new_stack.accumulate(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=L, Ly=1)
                new_stack.Py_.append(P)  # Py_: vertical buffer of Ps
                new_stack.down_connect_cnt = 0  # reset down_connect_cnt
                blob = new_stack.blob

            else:  # if > 1 up_connects, or 1 up_connect that has > 1 down_connect_cnt:
                blob = up_connect_[0].blob
                # initialize new_stack with up_connect blob:
                new_stack = CDeepStack(I=I, G=G, Dy=0, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=L, Ly=1,
                                       y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
                new_stack.hid = blob.id
                blob.stack_.append(new_stack)

                if len(up_connect_) > 1:  # merge blobs of all up_connects
                    if up_connect_[0].down_connect_cnt == 1:  # up_connect is not terminated
                        form_blob(up_connect_[0], root_dert__, sub_blobs)  # merge stack of 1st up_connect into its blob

                    for up_connect in up_connect_[1:len(up_connect_)]:  # merge blobs of other up_connects into blob of 1st up_connect
                        if up_connect.down_connect_cnt == 1:
                            form_blob(up_connect, root_dert__, sub_blobs)

                        if not up_connect.blob is blob:
                            merged_blob = up_connect.blob
                            I, G, Dy, Dx, M, iDy, iDx, S, Ly = merged_blob.Dert.values()
                            accum_Dert(blob.Dert, I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=S, Ly=Ly)
                            blob.open_stacks += merged_blob.open_stacks
                            blob.box[0] = min(blob.box[0], merged_blob.box[0])  # extend box y0
                            blob.box[1] = min(blob.box[1], merged_blob.box[1])  # extend box x0
                            blob.box[2] = max(blob.box[2], merged_blob.box[2])  # extend box xn
                            for stack in merged_blob.stack_:
                                if not stack is up_connect:
                                    stack.blob = blob  # blobs in other up_connects are references to blob in the first up_connect.
                                    stack.hid = blob.id
                                    blob.stack_.append(stack)  # buffer of merged root stacks.
                            up_connect.blob = blob
                            up_connect.hid = blob.id
                            blob.stack_.append(up_connect)
                        blob.open_stacks -= 1  # overlap with merged blob.

        blob.box[1] = min(blob.box[1], x0)  # extend box x0
        blob.box[2] = max(blob.box[2], xn)  # extend box xn
        P.hid = new_stack.id  # assign higher cluster id for P
        next_stack_.append(new_stack)

    return next_stack_


def form_blob(stack, root_dert__, sub_blobs):  # increment blob with terminated stack, check for blob termination

    I, G, Dy, Dx, M, iDy, iDx, S, Ly, y0, Py_, blob, down_connect_cnt, sign = stack.unpack()
    accum_Dert(blob.Dert, I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=S, Ly=Ly)
    # terminated stack is merged into continued or initialized blob (all connected stacks):

    blob.open_stacks += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_
    if blob.open_stacks == 0:  # if number of incomplete stacks == 0
        # blob is terminated and packed in blob root:
        last_stack = stack
        y0, x0, xn = blob.box
        yn = last_stack.y0 + last_stack.Ly

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:
        for stack in blob.stack_:
            for y, P in enumerate(stack.Py_, start=stack.y0 - y0):
                x_start = P.x0 - x0
                x_stop = x_start + P.L
                mask[y, x_start:x_stop] = False

        fopen = 0  # flag: blob on frame boundary
        if x0 == 0 or xn == root_dert__[0].shape[1] or y0 == 0 or yn == root_dert__[0].shape[0]:
            fopen = 1

        blob.root_dert__ = root_dert__
        blob.box = (y0, yn, x0, xn)
        blob.dert__ = [derts[y0:yn, x0:xn] for derts in root_dert__]
        blob.mask = mask
        blob.adj_blobs = [[], 0, 0]
        blob.fopen = fopen

        sub_blobs.append(blob)


def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_dert__[0].shape  # higher dert size

    # determine pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended

    # take ext_dert__ from part of root_dert__
    ext_dert__ = [derts[y0e:yne, x0e:xne] if derts is not None else None
                  for derts in blob.root_dert__]

    # pad mask: top, btm, left, right. 1 or 0 at boundaries
    mask = np.pad(blob.mask, ((y0 - y0e, yne - yn), (x0 - x0e, xne - xn)),
                  mode='constant', constant_values=True)

    return ext_dert__, mask


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})