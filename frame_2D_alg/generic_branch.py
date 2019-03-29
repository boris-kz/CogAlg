import numpy as np
from collections import deque, namedtuple
nt_blob = namedtuple('blob', 'sign params e_ box map dert__ new_dert__ rng ncomp sub_blob_')

# ************ FUNCTIONS ************************************************************************************************
# -master_blob()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def master_blob(blob, branch_comp, add_dert=True):  # redefine blob as branch-specific master blob: local equivalent of frame

    height, width = blob.map.shape

    if add_dert:
        blob.Derts.append(0, 0, 0, 0)  # I, Dy, Dx, G
        for i, derts in enumerate (blob.derts_):
            blob.derts_[i] = derts.append((0, 0, 0, 0))  # i, dy, dx, g

    if height < 3 or width < 3:
        return False

    rng = branch_comp(blob)  # also adds a branch-specific dert_ to blob
    rng_inc = bool(rng - 1)  # flag for comp range increase

    if blob.new_dert__[0].mask.all():
        return False
    seg_ = deque()

    for y in range(rng, height - rng):
        P_ = form_P_(y, blob)           # horizontal clustering
        P_ = scan_P_(P_, seg_, blob, rng_inc)    # vertical clustering
        seg_ = form_seg_(P_, blob, rng_inc)      # vertical clustering
    while seg_: form_blob(seg_.popleft(), blob, rng_inc) # terminate last running segments

    return True  # sub_blob_ != 0

    # ---------- branch_master_blob() end -------------------------------------------------------------------------------

def form_P_(y, master_blob):    # cluster and sum horizontally consecutive pixels and their derivatives into Ps

    rng = master_blob.rng
    dert__ = master_blob.new_dert__[0]
    P_ = deque()  # initialize output
    dert_ = dert__[y, :, :]  # row of pixels + derivatives
    P_map_ = ~dert_.mask[:, 3]  # dert_.mask?
    x_stop = len(dert_) - rng
    x = rng  # first and last rng columns are discarded

    while x < x_stop:
        while x < x_stop and not P_map_[x]:
            x += 1
        if x < x_stop and P_map_[x]:
            s = dert_[x][-1] > 0  # s = g > 0
            P = [s, [0, 0, 0] + [0] * len(dert_[0]), []]
            while x < x_stop and P_map_[x] and s == P[0]:

                dert = dert_[x, :]
                # accumulate P params:
                P[1] = [par1 + par2 for par1, par2 in zip(P[1], [1, y, x] + list(dert))]
                P[2].append((y, x) + tuple(dert))
                x += 1
                s = dert_[x][-1] > 0  # s = g > 0

            if P[1][0]:         # if L > 0
                P_.append(P)    # P is packed into P_
    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, master_blob, rng_inc = False):  # this function detects connections (forks) between Ps and _Ps, to form blob segments
    new_P_ = deque()

    if P_ and seg_:            # if both are not empty
        P = P_.popleft()       # input-line Ps
        seg = seg_.popleft()   # higher-line segments,
        _P = seg[2][-1]        # last element of each segment is higher-line P
        stop = False
        fork_ = []
        while not stop:
            x0 = P[2][0][1]     # first x in P
            xn = P[2][-1][1]    # last x in P
            _x0 = _P[2][0][1]   # first x in _P
            _xn = _P[2][-1][1]  # last x in _P

            if P[0] == _P[0] and _x0 <= xn and x0 <= _xn:  # test for sign match and x overlap
                seg[3] += 1
                fork_.append(seg)  # P-connected segments are buffered into fork_

            if xn < _xn:   # _P overlaps next P in P_
                new_P_.append((P, fork_))
                fork_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:
                    if seg[3] != 1:  # if roots != 1: terminate loop
                        form_blob(seg, master_blob, rng_inc)
                    stop = True
            else:
                if seg[3] != 1:  # if roots != 1
                    form_blob(seg, master_blob, rng_inc)
                if seg_:
                    seg = seg_.popleft()  # load next seg and _P
                    _P = seg[2][-1]
                else:
                    new_P_.append((P, fork_))
                    stop = True  # terminate loop

    while P_:  # handle Ps and segs that don't terminate at line end
        new_P_.append(( P_.popleft(), []))  # no fork
    while seg_:
        form_blob( seg_.popleft(), master_blob, rng_inc)  # roots always == 0
    return new_P_

    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_(P_, master_blob, rng_inc = False):   # Convert or merge every P into segment. Merge blobs
    new_seg_ = deque()
    while P_:
        P, fork_ = P_.popleft()
        s, params, dert_ = P

        if not fork_:  # seg is initialized with initialized blob
            blob = [s, [0] * (len(params) + 1), [], 1]    # s, params, seg_, open_segments
            seg = [s, [1] + params, [P], 0, fork_, blob] # s, params. P_, roots, fork_, blob
            blob[2].append(seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                seg = fork_[0]
                # P is merged into segment:
                seg[1] = [par1 + par2 for par1, par2 in zip([1] + params, seg[1])]  # sum all params of P into seg, in addition to +1 in Ly
                seg[2].append(P)    # Py_: vertical buffer of Ps merged into seg
                seg[3] = 0          # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]
                seg = [s, [1] + params, [P], 0, fork_, blob]  # seg is initialized with fork blob
                blob[2].append(seg) # segment is buffered into blob

                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], master_blob, rng_inc)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, master_blob, rng_inc)

                        if not fork[5] is blob:
                            params, e_, open_segments = fork[5][1:]  # merged blob, omit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]  # sum same-type params of merging blobs
                            blob[3] += open_segments
                            for e in e_:
                                if not e is fork:
                                    e[5] = blob       # blobs in other forks are references to blob in the first fork
                                    blob[2].append(e) # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1    # open_segments -= 1: shared seg is eliminated

        new_seg_.append(seg)
    return new_seg_

    # ---------- form_seg_() end --------------------------------------------------------------------------------------------

def form_blob(term_seg, master_blob, rng_inc = False): # terminated segment is merged into continued or initialized blob (all connected segments)

    params, P_, roots, fork_, blob = term_seg[1:]

    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1    # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in master_blob
        blob.pop()   # remove open_segments
        s, blob_params, e_ = blob
        y0 = 9999999
        x0 = 9999999
        yn = 0
        xn = 0
        for seg in e_:
            seg.pop()   # remove references of blob
            for P in seg[2]:
                y0 = min(y0, P[2][0][0])
                x0 = min(x0, P[2][0][1])
                yn = max(yn, P[2][0][0] + 1)
                xn = max(xn, P[2][-1][1] + 1)

        derts_ = master_blob.new_dert__[0][y0:yn, x0:xn, :]
        map = np.zeros((yn-y0, xn-x0), dtype=bool)
        for seg in e_:
            for P in seg[2]:
                for y, x, i, dy, dx, g in P[2]:
                    map[y-y0, x-x0] = True
        map = map[y0:yn, x0:xn]

        master_blob.params[-4:] = [par1 + par2 for par1, par2 in zip(master_blob.params[-4:], blob_params[-4:])]
        
        master_blob.sub_blob_[-1].append(nt_blob(typ=0, sign=s, Ly=Ly, L=L,
                                                 Derts=[(I, Dy, Dx, G)],  # will remain after derts_ replacement: not selective to +sub_blobs
                                                 derts_=derts_,  # intra_blob will convert each dert of selected blobs into [dert]
                                                 subf=0,   # flag: derts_[:]= sub_blob_ convert in intra_blob, blob derts_-> sub_blob derts_
                                                 layerf=0,  # flag: derts_ = [(sub_Derts, derts_)], appended per intra_blob eval_layer
                                                 sub_Derts=[],  # sub_blob_ Derts += [(Ly, L, I, Dy, Dx, G)] if len(sub_blob_) > min
                                                 lay_Derts=[],  # layer_ Derts += [(Ly, L, I, Dy, Dx, G)] if len(layer_) > min
                                                 box=(y0, yn, x0, xn),
                                                 map=map,
                                                 add_dert=None,
                                                 rng=(master_blob.rng + 1) if rng_inc else 1,  # increase or reset rng
                                                 ncomp=1(master_blob.ncomp + master_blob.rng + 1) if rng_inc else 1,
                                                 ))
        '''
        replace with:
        
        master_blob.sub_blobs[0][:] += Derts[:]  # accumulate each Derts param
        
        master_blob.sub_blobs[1][-1].append(nt_blob(
                                typ=typ,
                                sign=s,
                                Y=Y,
                                X=X,
                                Derts = [(Ly, L, I, Dy, Dx, G)],
                                derts_ = [dert__],  # extend all elements
                                sub_blobs = [],   # replace derts_, replaced by sub_layers
                                sub_layers = [],  # type ignored if == 1
                                box=(y0, yn, x0, xn),
                                map=map,
                                add_dert=None,
                                rng=1, ncomp=1
                                ))'''
    # ---------- form_blob() end -------------------------------------------------------------------------------------