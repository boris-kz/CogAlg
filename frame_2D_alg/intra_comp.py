import numpy as np
from math import hypot
from collections import deque, namedtuple
from comp_range import comp_range
from comp_angle import comp_angle
from comp_gradient import comp_gradient

nt_blob = namedtuple('blob', 'I Derts sign alt rng box map root_blob seg_')

# ************ FUNCTIONS ************************************************************************************************
# -intra_comp()
# -hypot_g()
# -compute_g_()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************


def intra_comp(blob, comp_branch, Ave_blob, Ave, calc_g = (lambda dx, dy, ncomp: int(( hypot(dx, dy) / ncomp)))):

    # calculate g = hypot(dx,dy) / ncomp,
    # unfold blob into derts, perform branch-specific comparison, convert blob into root_blob with new sub_blob_

    # for testing only, else set in intra_blob:
    if comp_branch == comp_range:
        rng = blob.rng + 1      # increment rng
        alt = blob.alt          # same sub_blobs
    elif comp_branch == hypot_g:
        rng = 0
        alt = -1
    else:       # if comp gradient or or comp_angle
        rng = 1
        alt = -1 if comp_branch == comp_gradient else -2

    blob.seg_.sort(key=lambda seg: seg[0])   # sort by y0 coordinate for unfolding
    blob.Derts.append((0, 0, 0, 0, 0, 0, []))  # G, Dy, Dx, N, L, Ly, sub_blob_: for accumulating feedback per sub_blob

    seg_ = []  # buffer of segments containing line y
    buff___ = deque(maxlen=rng)
    sseg_ = deque()  # buffer of sub-segments

    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i = 0  # segment index

    while y < yn:  # unfold blob into Ps for extended comp

        while i < len(blob.seg_) and blob.seg_[i][0] == y:
            seg_.append(blob.seg_[i])
            i += 1
        P_ = []  # line y Ps
        for seg in seg_:
            if y < seg[0] + seg[1][-1]:         # y < y0 + Ly within segment, or len(Py)?
                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][-1]:     # y >= y0 + Ly (out of segment):
                seg_.remove(seg)

        P_.sort(key=lambda P: P[1])  # sort by x0 coordinate
        # core operations:

        derts__ = comp_branch(P_, buff___, alt)   # no buff___ or alt in hypot_g or future dx_g
        if derts__:     # form sub_blobs:

            compute_g_(derts__, calc_g)
            sP_ = form_P_(derts__, alt, Ave, rng)
            sP_ = scan_P_(sP_, sseg_, blob, alt, rng)
            sseg_ = form_seg_(y - rng, sP_, blob, alt, rng)

        y += 1
    y -= len(buff___)

    while buff___:   # form sub blobs with dert_s remaining in buff__

        derts__ = buff___.pop()
        compute_g_(derts__, calc_g)
        sP_ = form_P_(derts__, alt, Ave, rng)
        sP_ = scan_P_(sP_, sseg_, blob, alt, rng)
        sseg_ = form_seg_(y, sP_, blob, alt, rng)
        y += 1

    while sseg_:    # terminate last line
        form_blob(sseg_.popleft(), blob, alt, rng)

    # ---------- intra_comp() end -------------------------------------------------------------------------------------------

def hypot_g(P_, *args):     # strip g from dert, convert dert into nested derts
    derts__ = []    # line of derts

    for P in P_:       # iterate through line of root_blob's Ps
        x0 = P[1]      # coordinate of first dert in a span of horizontally contiguous derts
        dert_ = P[-1]  # span of horizontally contiguous derts
        for index, (i, g, dy, dx) in enumerate(dert_):
            dert_[index]= [(i,), (dy, dx, 4)]    # ncomp=4, specified in deeper derts only

        derts__.append((x0, dert_))
    return derts__  # return i indices and derts__

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------

def compute_g_(derts__, calc_g):

    for x0, derts_ in derts__:
        for derts in derts_:
            dy, dx, ncomp = derts[-1]

            g = calc_g(dx, dy, ncomp)   # calculate g as hypot(dx,dy) / ncomp
            derts[-1] = (g,) + derts[-1]


def form_P_(derts__, alt, Ave, rng):      # horizontally cluster and sum consecutive pixels and their derivatives into Pss

    P_ = deque()    # row of Ps
    for x_start, derts_ in derts__:     # each derts_ is a span of horizontally contiguous derts, a line might contain many of these

        dert_ = [derts[-1 - rng][0:1] + derts[-1] for derts in derts_]   # temporary branch-specific dert_: (i, g, ncomp, dy, dx)
        i, g, dy, dx, ncomp = dert_[0]

        _vg = g - Ave
        _s = _vg > 0  # sign of first dert
        x0, I, G, Dy, Dx, N, L = x_start, i, _vg, dy, dx, ncomp, 1    # initialize P params with first dert

        for x, (i, g, dy, dx, ncomp) in enumerate(dert_[1:], start=x_start + 1):
            vg = g - Ave
            s = vg > 0
            if s != _s:  # P is terminated and new P is initialized:

                P_.append([_s, x0, I, G, Dy, Dx, N, L, derts_[x0 - x_start : x0 - x_start + L]])  # derts is appended in comp_branch
                x0, I, G, Dy, Dx, N, L = x, 0, 0, 0, 0, 0, 0    # reset params

            I += i  # accumulate P params
            G += vg
            Dy += dy
            Dx += dx
            N += ncomp
            L += 1
            _s = s  # prior sign

            P_.append([_s, x0, I, G, Dy, Dx, N, L, derts_[x0 - x_start: x0 - x_start + L]])  # last P in row
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, root_blob, alt, rng):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[2][-1]  # last element of each segment is higher-line P
        stop = False
        fork_ = []

        while not stop:
            x0 = P[1]  # first x in P
            xn = x0 + P[-2]  # first x in next P
            _x0 = _P[1]  # first x in _P
            _xn = _x0 + _P[-2]  # first x in next _P

            if P[0] == _P[0] and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
                seg[3] += 1
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # if no P left: terminate loop
                    if seg[3] != 1:  # if roots != 1: terminate seg
                        form_blob(seg, root_blob, alt, rng)
                    stop = True
            else:  # no next-P overlap
                if seg[3] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, root_blob, alt, rng)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg[2][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), root_blob, alt, rng)  # roots always == 0

    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_seg_(y, P_, root_blob, alt, rng):  # convert or merge every P into segment, merge blobs
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()
        s, x0 = P[:2]           # s, x0
        params = P[2:-1]        # L and other params
        xn = x0 + params[-1]    # next-P x0 = x0 + L
        params.append(1)        # add Ly

        if not fork_:  # new_seg is initialized with initialized blob
            blob = [s, [0] * (len(params)), [], 1, [y, x0, xn]]   # s, params, seg_, open_segments, box
            new_seg = [y, params, [P], 0, fork_, blob]            # y0, params, Py_, roots, fork_, blob
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]
                params_seg = new_seg[1]     # fork segment params, P is merged into segment:
                new_seg[1] = [param + param_seg for param, param_seg in zip(params, params_seg)]
                new_seg[2].append(P)        # Py_: vertical buffer of Ps
                new_seg[3] = 0              # reset roots
                blob = new_seg[-1]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                new_seg = [y, params, [P], 0, fork_, blob]  # new_seg is initialized with fork blob
                blob[2].append(new_seg)                     # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], root_blob, alt, rng)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, root_blob, alt, rng)

                        if not fork[5] is blob:
                            params, seg_, open_segs, box = fork[5][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in
                                       zip(blob[1], params)]  # sum params of merging blobs
                            blob[3] += open_segs
                            blob[4][0] = min(blob[4][0], box[0])  # extend box y0
                            blob[4][1] = min(blob[4][1], box[1])  # extend box x0
                            blob[4][2] = max(blob[4][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is fork:
                                    seg[5] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(seg)  # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1: shared with merged blob

        blob[4][1] = min(blob[4][1], x0)  # extend box x0
        blob[4][2] = max(blob[4][2], xn)  # extend box xn
        new_seg_.append(new_seg)

    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob(term_seg, root_blob, alt, rng):  # terminated segment is merged into continued or initialized blob (all connected segments)

    y0s, params, Py_, roots, fork_, blob = term_seg
    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1  # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame

        s, [I, G, Dy, Dx, N, L, Ly], seg_, open_segs, (y0, x0, xn) = blob
        yn = y0s + params[-1]            # yn = y0 + Ly (segment's)
        map = np.zeros((yn - y0, xn - x0), dtype=bool)  # local map of blob

        for seg in seg_:
            seg.pop()  # remove references to blob
            for y, P in enumerate(seg[2], start=seg[0]):
                x0P = P[1]
                LP = P[-2]
                xnP = x0P + LP
                map[y - y0, x0P - x0:xnP - x0] = True

        # accumulate root_blob.Derts[-1] (I is not changed),
        # make recursive accum_root_blob()?

        Gr, Dyr, Dxr, Nr, Lr, Lyr, sub_blob_ = root_blob.Derts[-1]
        Dyr += Dy
        Dxr += Dx
        Gr += G
        Nr += N
        Lr += L
        Lyr += Ly
        sub_blob_.append(nt_blob(I=I,  # top Dert is I only
                                 Derts=[(G, Dy, Dx, N, L, Ly)],  #(G, Dy, Dx, N, L, Ly, []), with nested sub_blob_ of depth = Derts[index]
                                 sign=s,
                                 alt= alt,  # alt layer index: -1 for ga | -2 for g, none for hypot_g
                                 rng= rng,  # for comp_range only, i_dert = -(rng-1)*2 + alt
                                 box= (y0, yn, x0, xn),  # boundary box
                                 map= map,   # blob boolean map, to compute overlap
                                 root_blob=blob,
                                 seg_=seg_,
                                 ) )
        root_blob.Derts[-1] = Gr, Dyr, Dxr, Nr, Lr, Lyr, sub_blob_


        # initialized Derts[-1] is added at intra_comp call
        # add recursive call to higher root_blob:
        # while root_blob.root_blob:
        #   blob = root_blob...
        # add Ave return if fangle, eval intra_blob call?

    # ---------- form_blob() end ----------------------------------------------------------------------------------------
