import numpy as np
from collections import deque, namedtuple
from comp_i import comp_i

Blob = namedtuple('Blob', 'Dert, sign, rng, box, map, seg_, sub_blob_, lLayers, root_blob, hLayers')

# ************ FUNCTIONS ************************************************************************************************
# -intra_comp()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************


def intra_comp(blob, rng, fia, fa, Ave_blob, Ave):

    # unfold blob into derts, perform branch-specific comparison, convert blob into root_blob with new sub_blob_

    blob.seg_.sort(key=lambda seg: seg[0])   # sort by y0 coordinate for unfolding
    seg_ = []  # buffer of segments containing line y
    _derts___ = deque(maxlen=rng)  # buffer of template derts, accumulated over multiple comps

    sseg_ = deque()  # buffer of sub-segments
    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i_seg = 0   # segment index

    while y < yn:  # unfold blob into Ps for extended comp

        while i_seg < len(blob.seg_) and blob.seg_[i_seg][0] == y:
            seg_.append(blob.seg_[i_seg])
            i_seg += 1
        P_ = []  # line y Ps
        for seg in seg_:
            if y < seg[0] + seg[1][-1]:         # y < y0 + Ly within segment, or len(Py)?
                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][-1]:     # y >= y0 + Ly (out of segment):
                seg_.remove(seg)

        P_.sort(key=lambda P: P[1])  # sort by x0 coordinate
        # core operations:

        derts__ = comp_i(P_, _derts___, rng, fia, fa)   # no _derts___ in hypot_g or future dx_g
        if derts__:  # form sub_blobs:

            sP_ = form_P_(derts__, Ave, rng, fia)
            sP_ = scan_P_(sP_, sseg_, blob, rng, fa)
            sseg_ = form_seg_(y - rng, sP_, blob, rng, fa)

        y += 1

    while sseg_:    # terminate last line
        form_blob(sseg_.popleft(), blob, rng, fa)

    # ---------- intra_comp() end -------------------------------------------------------------------------------------------

def form_P_(derts__, Ave, rng, fia):  # horizontally cluster and sum consecutive (pixel, derts) into Ps

    P_ = deque()    # row of Ps
    cyc = -rng -1 + fia  # cyc and rng are cross-convertable

    for x_start, tderts_ in derts__:  # each derts_ is a span of horizontally contiguous derts, multiple derts_ per line

        dert_ = [derts[-1] + derts[-1] for derts in tderts_]   # temporary branch-specific dert_: (i, g, ncomp, dy, dx)
        i, g, dy, dx = dert_[0]

        _vg = g - Ave
        _s = _vg > 0  # sign of first dert
        x0, I, G, Dy, Dx, L = x_start, i, _vg, dy, dx, 1    # initialize P params with first dert

        for x, (i, g, dy, dx) in enumerate(dert_[1:], start=x_start + 1):
            vg = g - Ave
            s = vg > 0
            if s != _s:  # P is terminated and new P is initialized:

                P_.append([_s, x0, I, G, Dy, Dx, L, tderts_[x0 - x_start : x0 - x_start + L]])  # derts is appended in comp_branch
                x0, I, G, Dy, Dx, L = x, 0, 0, 0, 0, 0    # reset params

            I += i  # accumulate P params
            G += vg
            Dy += dy
            Dx += dx
            L += 1
            _s = s  # prior sign

            P_.append([_s, x0, I, G, Dy, Dx, L, tderts_[x0 - x_start: x0 - x_start + L]])  # last P in row
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, root_blob, rng, fa):  # integrate x overlaps (forks) between same-sign Ps and _Ps into blob segments

    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[-3][-1]  # last element of each segment is higher-line P
        stop = False
        fork_ = []

        while not stop:
            x0 = P[1]  # first x in P
            xn = x0 + P[-2]  # first x in next P
            _x0 = _P[1]  # first x in _P
            _xn = _x0 + _P[-2]  # first x in next _P

            if P[0] == _P[0] and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
                seg[-1] += 1        # roots += 1
                fork_.append(seg)   # P-connected segments are buffered into fork_

            if xn < _xn:  # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # if no P left: terminate loop
                    if seg[-1] != 1:  # if roots != 1: terminate seg
                        form_blob(seg, root_blob, rng, fa)
                    stop = True
            else:  # no next-P overlap
                if seg[-1] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, root_blob, rng, fa)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg[-3][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), root_blob, rng, fa)  # roots always == 0

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
            blob = [s, [0] * (len(params)), [], 1, [y, x0, xn]]  # s, params, seg_, open_segments, box
            new_seg = [y] + params + [[P]] + [blob, 0]           # y0, I, G, Dy, Dx, N, L, Ly, Py_, blob, roots
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]
                I, G, Dy, Dx, L, Ly = params
                Is, Gs, Dys, Dxs, Ls, Lys = new_seg[1:-3]  # fork segment params, P is merged into segment:

                new_seg[1:-3] = [Is + I, Gs + G, Dys + Dy, Dxs + Dx, Ls + L, Lys + Ly]
                new_seg[-3].append(P)  # Py_: vertical buffer of Ps
                new_seg[-1] = 0  # reset roots
                blob = new_seg[-2]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][-2]
                new_seg = [y] + params + [[P]] + [blob, 0]  # new_seg is initialized with fork blob
                blob[2].append(new_seg)                     # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][-1] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], root_blob, alt, rng)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[-1] == 1:
                            form_blob(fork, root_blob, alt, rng)

                        if not fork[-2] is blob:
                            params, seg_, open_segs, box = fork[-2][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(blob[1], params)]  # sum params of merging blobs
                            blob[3] += open_segs
                            blob[4][0] = min(blob[4][0], box[0])  # extend box y0
                            blob[4][1] = min(blob[4][1], box[1])  # extend box x0
                            blob[4][2] = max(blob[4][2], box[2])  # extend box xn
                            for seg in seg_:
                                if not seg is fork:
                                    seg[-2] = blob  # blobs in other forks are references to blob in the first fork
                                    blob[2].append(seg)  # buffer of merged root segments
                            fork[-2] = blob
                            blob[2].append(fork)
                        blob[3] -= 1  # open_segments -= 1: shared with merged blob

        blob[4][1] = min(blob[4][1], x0)  # extend box x0
        blob[4][2] = max(blob[4][2], xn)  # extend box xn
        new_seg_.append(new_seg)

    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------

def form_blob(term_seg, root_blob, rng, fa):  # terminated segment is merged into continued or initialized blob (all connected segments)

    y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_, blob, roots = term_seg
    I, G, Dy, Dx, L, Ly = blob[1]
    blob[1] = [I + Is, G + Gs, Dy + Dys, Dx + Dxs, L + Ls, Ly + Lys]
    blob[-2] += roots - 1  # number of open segments

    if blob[-2] == 0:  # if open_segments == 0: blob is terminated and packed in frame
        s, _, seg_, open_segs, [y0, x0, xn] = blob

        yn = y0s + Lys  # yn from last segment
        map = np.zeros((yn - y0, xn - x0), dtype=bool)  # local map of blob
        for seg in seg_:
            seg.pop()  # remove roots counter
            seg.pop()  # remove references to blob
            for y, P in enumerate(seg[-1], start=seg[0]):
                x0P = P[1]
                LP = P[-2]
                xnP = x0P + LP
                map[y - y0, x0P - x0:xnP - x0] = True

        feedback_draft(root_blob, blob, rng)

        del blob

    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def feedback_draft(root_blob, blob, rng):  # or rng per dert cyc: current, !i_cyc: cont. g | gg | ga rng expansion?

    # fga = g | ga sub_layer
    s, [I, G, A, Dy, Dx, L, Ly], seg_, open_segs, box = blob

    while root_blob:  # add all blob Layers' Dert params to corresponding params of recursively higher root_blob

        if len(blob.Layers) == len(root_blob.Layers):  # fork index is defined incrementally, with fb elevation?
            root_blob.Layers += [(0, 0, 0, 0, 0, []),[]]  # new Layer, breadth-first, fork = [fder][fga]?

        root_blob.high_Derts[:][:] += blob.high_Derts[1:][:]  # also accumulated per sub_blob, initially 0?
        root_blob.Dert[1][:] += blob.high_Derts[0][:]  # Dyr += Dy, Dxr += Dx, Gr += G, Lyr += Ly, Lr += L

        # blob.high_Derts[0] is summed next to discrete root_blob.Dert[0],
        # then root_root_blob.Layers[1] sums across min n? source layers, buffered in target Layer

        root_blob.sub_blob_.append(Blob
                                  (Dert=[G, None, Dy, Dx, L, Ly],  # core Layer of current blob, A is None for g_Dert
                                  # Layers[>1] = forks[cyc][fga], same as input derts[cyc][fga], added by feedback
                                  # sub_blob_ = [] per blob or fork, nested to depth = Levels[cyc][fga]
                                  sign = s,
                                  rng= rng,
                                  box= box,  # same boundary box
                                  map= map,  # blob boolean map, to compute overlap
                                  seg_=seg_,
                                  # derts__,
                                  sub_blob_=[],
                                  lLayers = [],  # summed reps of lower layers across sub_blob derivation tree
                                  root_blob = [blob],  # ref for feedback of all Derts params summed in sub_blobs
                                  hLayers = [I]  # higher Dert params += higher-dert params, starting with I
                                  ))
        root_blob = root_blob.root_blob

    # add:
    # return Ave_blob *= len(sub_blob_) / ave_n_sub_blobs

