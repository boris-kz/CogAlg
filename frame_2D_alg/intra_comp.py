import numpy as np
from collections import deque, namedtuple
from compare_i import compare_i

Dert = namedtuple('Dert', 'G, Dy, Dx, L, Ly')
Pattern = namedtuple('Pattern', 'sign, x0, I, G, Dy, Dx, L, derts_')
Blob = namedtuple('Blob', 'I Derts sign box map root_blob seg_')

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_comp_g         = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -intra_comp()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************


def intra_comp(blob, Ave, Ave_blob, flags=0):  # flags = angle | increasing_range | hypot_g. default: 0 - do hypot_g
    # unfold blob into derts, perform branch-specific comparison, convert blob into root_blob with new sub_blob_

    fia = flags & f_angle                               # check if inputs are angles
    rng = blob.rng + 1 if (flags & f_inc_rng) else 1    # check range should be incremented
    blob.seg_.sort(key=lambda seg: seg[0])              # sort by y0 coordinate for unfolding
    seg_ = []                       # buffer of segments containing line y
    _dert___ = deque(maxlen=rng)    # buffer of higher-line derts__
    i__ = object()                  # place-holder for buffer of unfolded higher-lines inputs (p, g or a). Also buffers accumulated previous-rng dx, dy if rng > 1
    sseg_ = deque()                 # buffer of sub-segments
    y0, yn, x0, xn = blob.box
    y = y0  # current y, from seg y0 -> yn - 1
    i = 0   # segment index

    while y < yn:  # unfold blob into Ps for extended comp

        while i < len(blob.seg_) and blob.seg_[i].y == y:
            seg_.append(blob.seg_[i])
            i += 1
        P_ = []  # line y Ps
        for seg in seg_:
            if y < seg.y + seg.Ly:              # y < y0 + Ly within segment, or len(Py)?
                P_.append(seg.Py_[y - seg.y])   # append P at line y of seg

        for seg in seg_:
            if y >= seg.y + seg.Ly:     # y >= y0 + Ly (out of segment):
                seg_.remove(seg)

        P_.sort(key=lambda P: P.x0)  # sort by x0 coordinate
        # core operations:
        indices = blob.map[y - y0, :].nonzero()
        derts__, i__ = compare_i(P_, _dert___, i__, (x0, xn), indices, flags)  # no _dert___ returned: _dert___ are edited not replaced
        # if derts__:     # form sub_blobs: currently excluded, for debugging compare_derts

            # sP_ = form_P_(derts__, Ave, rng, fa)
            # sP_ = scan_P_(sP_, sseg_, blob, rng, fa)
            # sseg_ = form_seg_(y - rng, sP_, blob, rng, fa)

        y += 1

    # while sseg_:    # terminate last line
    #     form_blob(sseg_.popleft(), blob, rng, fa)

    # ---------- intra_comp() end -------------------------------------------------------------------------------------------

def form_P_(derts__, Ave, rng, fa):  # horizontally cluster and sum consecutive (pixel, derts) into Ps

    P_ = deque()    # row of Ps
    for x_start, derts_ in derts__:   # each derts_ is a span of horizontally contiguous derts, multiple derts_ per line

        dert_ = [derts[-rng-1+fa][fa] + derts[-1][:1] + derts[-1][-2:] for derts in derts_]   # temporary branch-specific dert_: (i, g, ncomp, dy, dx)
        i, g, dy, dx = dert_[0]

        _vg = g - Ave
        _s = _vg > 0  # sign of first dert
        x0, I, G, Dy, Dx, L = x_start, i, _vg, dy, dx, 1    # initialize P params with first dert

        for x, (i, g, dy, dx) in enumerate(dert_[1:], start=x_start + 1):
            vg = g - Ave
            s = vg > 0
            if s != _s:  # P is terminated and new P is initialized:
                P = Pattern(_s, x0, I, G, Dy, Dx, L, derts_[x0 - x_start : x0 - x_start + L])
                P_.append(P)
                x0, I, G, Dy, Dx, L = x, 0, 0, 0, 0, 0    # reset params

            I += i  # accumulate P params
            G += vg
            Dy += dy
            Dx += dx
            L += 1
            _s = s  # prior sign

            P = Pattern(_s, x0, I, G, Dy, Dx, L, derts_[x0 - x_start: x0 - x_start + L])
            P_.append(P)
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
            x0 = P.x0  # first x in P
            xn = x0 + P.L  # first x in next P
            _x0 = _P.x0  # first x in _P
            _xn = _x0 + _P.L  # first x in next _P

            if P.sign == _P.sign and _x0 < xn and x0 < _xn:  # test for sign match and x overlap
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
        form_blob(seg_.popleft(), root_blob, rng, fa)   # roots always == 0

    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------


def form_seg_(y, P_, root_blob, rng, fa):  # convert or merge every P into segment, merge blobs
    new_seg_ = deque()

    while P_:
        P, fork_ = P_.popleft()

        s, x0, I, G, Dy, Dx, L, dert_ = P
        xn = x0 + L  # next-P x0

        if not fork_:  # new_seg is initialized with initialized blob
            blob = [s, [0, 0, 0, 0, 0, 0], [], 1, [y, x0, xn]]      # s, [I, G, Dy, Dx, L], seg_, open_segments, box
            new_seg = [y, I, G, Dy, Dx, L, 1, [P], blob, 0]  # y0, I, G, Dy, Dx, N, L, Ly, Py_, blob, roots
            blob[2].append(new_seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                new_seg = fork_[0]

                Is, Gs, Dys, Dxs, Ls, Ly = new_seg[1:-3]    # fork segment params, P is merged into segment:
                new_seg[1:-3] = [Is + I, Gs + G, Dys + Dy, Dxs + Dx, Ls + L, Ly + 1]
                new_seg[-3].append(P)  # Py_: vertical buffer of Ps
                new_seg[-1] = 0  # reset roots
                blob = new_seg[-2]

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][-2]
                new_seg = [y, I, G, Dy, Dx, L, 1, [P], blob, 0]     # new_seg is initialized with fork blob
                blob[2].append(new_seg)                             # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][-1] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], root_blob, rng, fa)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[-1] == 1:
                            form_blob(fork, root_blob, rng, fa)

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
        new_seg_ = []
        for seg in seg_:
            y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_ = seg[:-2]                  # blob and roots are ignored
            seg = Segment(y0s, Is, Gs, Dys, Dxs, Ls, Lys, Py_)              # convert to Segment namedtuple
            new_seg_.append(seg)                                            # add segment to blob as namedtuple
            for y, P in enumerate(seg.Py_, start=seg.y):
                Pxn = P.x0 + P.L
                map[y - y0, P.x0 - x0:Pxn - x0] = True

        '''
        blob = Blob(I=I,  # 0th Dert is I only
                    Derts=[[ (cyc, alt, typ), (G, Dy, Dx, N, L, Ly, []) ]],  # 1st Dert is single-blob
                    # alt: sub_layer index: -1 ga | -3 g, default -2 a if rng==1, none for hypot_g
                    # rng: dert cycle index for comp_range only: i_dert = -(rng-1)*3 + alt
                    sign = s,
                    box= box,  # same boundary box
                    map= map,  # blob boolean map, to compute overlap
                    root_blob=blob,
                    # comp_range_input_ = new_comp_range_input_ # root_blob [(alt, rng)]:
                    # Dert @ prior intra_blob comp_branch input blob is evaluated for comp_range
                    seg_=seg_,
                    )

        feedback_draft(root_blob, blob, rng)
        '''
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

def feedback_draft(root_blob, blob, rng):

    s, [I, G, Dy, Dx, N, L, Ly], seg_, open_segs, box = blob

    while root_blob:  # accumulate same- cycle( alt( type Dert in recursively higher root_blob, typ for alt -1 only?

        cyc = blob.cyc; alt = blob.alt; typ = blob.typ
        type_Derts = root_blob.Derts[cyc][alt][typ]
        same_type = 0

        if blob.cyc > len(root_blob.Derts):  # cycle( alt( type Dert
            root_blob.Derts += []  # Dert[ alt_Derts[ typ_Dert ]]] may be initialized by any comp_branch

        for i, (_cyc, _alt, _typ) in enumerate( type_Derts[0]):  # select same-type Dert by alt & rng:
            if cyc == _cyc and alt == _alt and typ == _typ:  # same-sub_blob-type Dert
                same_type = 1
                same_Dert = type_Derts[i]
                break
        if same_type == 0:
            type_Derts += ((blob.cyc, blob.alt, blob.typ), (0, 0, 0, 0, 0, 0, []))  # initialize new type_Dert
            same_Dert = type_Derts[-1]

        Gr, Dyr, Dxr, Nr, Lr, Lyr, sub_blob_ = same_Dert[1]  # I is not changed
        Dyr += Dy
        Dxr += Dx
        Gr += G
        Nr += N
        Lr += L
        Lyr += Ly
        # + nblobs per type, for nested sub_blob_ in deeper layers?

        sub_blob_.append()
        root_blob.Derts[-1][1] = Gr, Dyr, Dxr, Nr, Lr, Lyr, sub_blob_

        root_blob = root_blob.root_blob

    # add Ave_blob return if fangle,
    # add eval intra_blob call if not fangle