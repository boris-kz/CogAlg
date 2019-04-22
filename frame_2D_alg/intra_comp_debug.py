import numpy as np
from math import hypot
from collections import deque, namedtuple

nt_blob = namedtuple('blob', 'Derts typ rng sign map box root_blob seg_')
ave = 5

# ************ FUNCTIONS ************************************************************************************************
# -intra_comp()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def intra_comp(blob, comp_branch, Ave_blob, Ave, rng=1):

    # unfold blob into derts, perform branch-specific comparison, convert blob into root_blob with new sub_blob_

    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i = 0   # segment index
    blob.seg_.sort(key=lambda seg: seg[0])  # sort by y0 coordinate
    buff___ = deque(maxlen=rng)
    seg_ = []       # buffer of segments containing line y
    sseg_ = deque() # buffer of sub-segments

    while y < yn and i < len(blob.seg_):  # unfold blob into Ps for extended comp:

        while i < len(blob.seg_) and blob.seg_[i][0] == y:
            seg_.append(blob.seg_[i])
            i += 1
        P_ = []  # line y Ps
        for seg in seg_:
            if y < seg[0] + seg[1][0]:          # y < y0 + Ly within segment, or len(Py)?
                P_.append(seg[2][y - seg[0]])   # append P at line y of seg

        for seg in seg_:
            if not y < seg[0] + seg[1][0]:      # y >= y0 + Ly (out of segment):
                seg_.remove(seg)

        P_.sort(key=lambda P: P[1])  # sort by x0 coordinate
        #  core operation:
        derts__ = comp_branch(P_, buff___)     # no buff___ in hypot_g or future dx_g

        #  form sub_structures -> sub_blobs:
        sP_ = form_P_(derts__, 0)
        sP_ = scan_P_(sP_, sseg_, blob)
        sseg_ = form_seg_(y, sP_, blob)

        y += 1

    while buff___:  # form sub blobs with dert_s remaining in dert_buff__
        sP_ = form_P_(derts__)
        sP_ = scan_P_(sP_, sseg_, blob)
        sseg_ = form_seg_(y, sP_, blob)

    while sseg_:   # terminate last line
        form_blob(sseg_.popleft(), blob)

    # ---------- intra_comp() end -------------------------------------------------------------------------------------------


def hypot_g(P_, buff___):  # here for testing only
    derts__ = []

    for P in P_:
        x0 = P[1]
        dert_ = P[-1]
        for index, (i, dy, dx, g) in enumerate(dert_):
            g = int(hypot(dx, dy) - ave * 4)

            dert_[index] = [(i, dy, dx, 4, g)]  # ncomp=4: implicit in hypot_g?
        derts__.append((x0, dert_))
    return derts__

    # ---------- hypot_g() end ----------------------------------------------------------------------------------------------


def form_P_(derts__, dert_index):  # horizontally cluster and sum consecutive pixels and their derivatives into Ps
    P_ = deque()  # row of new Ps

    for x_start, derts_ in derts__:  # derts_ per input P ) derts__ per input line

        dert_ = [derts[dert_index] for derts in derts_]  # list of specific types of dert
        i, dy, dx, n, g = dert_[0]
        x0, L, I, Dy, Dx, N, G = x_start, 1, i, dy, dx, n, g  # P params, no default // ncomp: for comp_param only?
        _s = g > 0  # sign

        for x, (i, dy, dx, n, g) in enumerate(dert_[1:-1], start=x_start + 1):
            s = g > 0
            if s != _s:  # P is terminated and new P is initialized:
                P_.append([_s, x0, L, I, Dy, Dx, N, G, derts_[x0: x0 + L]])
                x0, L, I, Dy, Dx, N, G = x, 0, 0, 0, 0, 0, 0
            # accumulate P params:
            L += 1
            I += i
            Dy += dy
            Dx += dx
            N += n  # blob-wide number of comparisons, for summed params' norm -> comp
            G += g  # g -= ave * ncomp
            _s = s  # prior sign

        P_.append([_s, x0, L, I, Dy, Dx, N, G, derts_[x0: x0 + L]])  # last P in row
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_(P_, seg_, root_blob):  # x overlaps (forks) between same-sign Ps and _Ps are integrated into blob segments
    new_P_ = deque()

    if P_ and seg_:  # if both are not empty
        P = P_.popleft()  # input-line Ps
        seg = seg_.popleft()  # higher-line segments,
        _P = seg[2][-1]  # last element of each segment is higher-line P
        stop = False
        fork_ = []

        while not stop:
            x0 = P[1]  # first x in P
            xn = x0 + P[2]  # first x in next P
            _x0 = _P[1]  # first x in _P
            _xn = _x0 + _P[2]  # first x in next _P

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
                        form_blob(seg, root_blob)
                    stop = True
            else:  # no next-P overlap
                if seg[3] != 1:  # if roots != 1: terminate seg
                    form_blob(seg, root_blob)

                if seg_:  # load next _P
                    seg = seg_.popleft()
                    _P = seg[2][-1]
                else:  # if no seg left: terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    while P_:  # terminate Ps and segs that continue at line's end
        new_P_.append((P_.popleft(), []))  # no fork
    while seg_:
        form_blob(seg_.popleft(), root_blob)  # roots always == 0

    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_seg_(y, P_, root_blob):  # convert or merge every P into segment, merge blobs
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()
        s, x0 = P[:2]  # s, x0
        params = P[2:-1]  # L, I, Dx, Dy, N, G
        xn = x0 + params[0]  # next-P x0 = x0 + L

        if not fork_:  # new_seg is initialized with initialized blob
            blob = [s, [0] * (len(params) + 1), [], 1, [y, x0, xn]]  # s, params, seg_, open_segments, box
            new_seg = [y, [1] + params, [P], 0, fork_, blob]  # y0, params, Py_, roots, fork_, blob
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]
                L, I, Dy, Dx, N, G = params
                Ly, Ls, Is, Dys, Dxs, Ns, Gs = new_seg[1]  # fork segment params, P is merged into segment:
                new_seg[1] = [Ly + 1, Ls + L, Is + I, Dys + Dy, Dxs + Dx, N + Ns, Gs + G]
                new_seg[2].append(P)  # Py_: vertical buffer of Ps
                new_seg[3] = 0  # reset roots
                blob = new_seg[-1]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                new_seg = [y, [1] + params, [P], 0, fork_, blob]  # new_seg is initialized with fork blob
                blob[2].append(new_seg)  # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], root_blob)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, root_blob)

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


def form_blob(term_seg, root_blob):  # merge terminated segment into continued or initialized blob (all connected segments)

    y0s, params, Py_, roots, fork_, blob = term_seg
    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1  # number of open segments
    root_blob.sub_Derts[:] = 0,0,0,0,0,0,0

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame

        s, [Ly, L, I, Dy, Dx, N, G], seg_, open_segs, (y0, x0, xn) = blob
        yn = y0s + params[0]  # yn = y0 + Ly
        map = np.zeros((yn - y0, xn - x0), dtype=bool)  # local map of blob

        for seg in seg_:
            seg.pop()  # remove references to blob
            for y, P in zip(range(seg[0], seg[0] + seg[1][0]), seg[2]):
                x0P, LP = P[1:3]
                xnP = x0P + LP
                map[y - y0, x0P - x0:xnP - x0] = True

        if s:  # for positive sub_blobs only
            root_blob.Derts[-1][0] += Ly
            root_blob.Derts[-1][1] += L

        root_blob.Derts[-1][2] += I
        root_blob.Derts[-1][3] += N
        root_blob.Derts[-1][4] += Dy
        root_blob.Derts[-1][5] += Dx
        root_blob.Derts[-1][6] += G

        root_blob.Derts[-1][7].append(nt_blob(
                  Derts=[(Ly, L, I, N, Dy, Dx, G, [])],  # [] is nested sub_blob_ of depth = Derts[index]
                  typ=0,  # top Dert only:
                  rng=1,  # for comp_range per blob
                  sign=s,
                  box=(y0, yn, x0, xn),  # boundary box
                  map=map,  # blob boolean map, to compute overlap
                  root_blob=[blob],
                  seg_=seg_,
                  ))
    # ---------- form_blob() end ----------------------------------------------------------------------------------------
