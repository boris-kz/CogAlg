from collections import deque

# ************ FUNCTIONS ************************************************************************************************
# -form_P_()
# -scan_P()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

def form_P_(y, dert__, rng = 1):
    ''' cluster horizontally consecutive inputs into Ps, buffered in P_ '''

    P_ = deque()  # initialize the output of this function

    dert_ = dert__[y, :, :] # row y

    x_stop = len(dert_) - rng
    x = rng                 # first and last rng columns are discarded

    while x < x_stop:

        s = dert_[x][-1] > 0  # s = (sdert > 0)

        P = [s, [0, 0, 0] + [0] * len(dert_[0]), []]

        while x < x_stop and s == P[0]:

            dert = dert_[x, :]

            # accumulate P's params:
            P[1] = [par1 + par2 for par1, par2 in zip(P[1], [1, y, x] + list(dert))]

            P[2].append((y, x) + tuple(dert))

            x += 1

            s = dert_[x][-1] > 0  # s = (sdert > 0)

        if P[1][0]:         # if L > 0
            P_.append(P)    # P is packed into P_

    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

def scan_P_(P_, seg_, frame):
    ''' each running segment in seg_ has 1 _P, or P of higher-line. P_ contain current line P.
        This function detects all connections between every P and _P in the form of fork_ '''

    new_P_ = deque()
    if P_ and seg_:             # if both are not empty
        P = P_.popleft()
        seg = seg_.popleft()
        _P = seg[2][-1]         # higher-line P is last element in seg's P_

        stop = False
        fork_ = []

        while not stop:

            x0 = P[2][0][1]     # P's first dert's x: y, x, i, dy, dx, sg = P[2][0]
            xn = P[2][-1][1]    # P's last dert's x: y, x, i, dy, dx, sg = P[2][-1]

            _x0 = _P[2][0][1]   # P's first dert's x: y, x, i, dy, dx, sg = P[2][0]
            _xn = _P[2][-1][1]  # P's last dert's x: y, x, i, dy, dx, sg = P[2][-1]

            if P[0] == _P[0] and _x0 <= xn and x0 <= _xn:  # check sign and olp
                seg[3] += 1
                fork_.append(seg)  # P-connected segments buffered into fork_

            if xn < _xn:    # P is on the left of _P: next P
                new_P_.append((P, fork_))
                fork_ = []
                if P_:      # switch to next P
                    P = P_.popleft()
                else:       # terminate loop
                    if seg[3] != 1: # if roots != 1
                        form_blob(seg, frame)
                    stop = True
            else:           # _P is on the left of P_: next _P
                if seg[3] != 1: # if roots != 1
                    form_blob(seg, frame)

                if seg_:    # switch to new _P
                    seg = seg_.popleft()
                    _P = seg[2][-1]
                else:       # terminate loop
                    new_P_.append((P, fork_))
                    stop = True

    # handle the remainders:
    while P_:
        new_P_.append((P_.popleft(), []))     # no fork
    while seg_:
        form_blob(seg_.popleft(), frame)  # roots always == 0

    return new_P_
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_(P_, frame):
    " Convert or merge every P into segment. Merge blobs "

    new_seg_ = deque()

    while P_:

        P, fork_ = P_.popleft()
        s, params, dert_ = P

        if not fork_:  # seg is initialized with initialized blob (params, coordinates, incomplete_segments, root_, xD)
            blob = [s, [0] * (len(params) + 1), [], 1]    # s, params, seg_, open_segments
            seg = [s, [1] + params, [P], 0, fork_, blob] # s, params. P_, roots, fork_, blob
            blob[2].append(seg)
        else:
            if len(fork_) == 1 and fork_[0][3] == 1:  # P has one fork and that fork has one root
                # P is merged into segment fork_[0] (seg):
                seg = fork_[0]

                seg[1] = [par1 + par2 for par1, par2 in zip([1] + params, seg[1])]  # sum all params of P into seg, in addition to +1 in Ly

                seg[2].append(P)    # P_: vertical buffer of Ps merged into seg
                seg[3] = 0          # reset roots

            else:  # if > 1 forks, or 1 fork that has > 1 roots:
                blob = fork_[0][5]                       # fork's blob
                seg = [s, [1] + params, [P], 0, fork_, blob] # seg is initialized with fork's blob
                blob[2].append(seg) # segment is buffered into blob
                if len(fork_) > 1:  # merge blobs of all forks
                    if fork_[0][3] == 1:  # if roots == 1: fork hasn't been terminated
                        form_blob(fork_[0], frame)  # merge seg of 1st fork into its blob

                    for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                        if fork[3] == 1:
                            form_blob(fork, frame)

                        if not fork[5] is blob:
                            params, e_, open_segments = fork[5][1:]  # merged blob, ommit sign
                            blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]  # sum every params per type of 2 merging blobs
                            blob[3] += open_segments
                            for e in e_:
                                if not e is fork:
                                    e[5] = blob       # blobs in other forks are references to blob in the first fork
                                    blob[2].append(e) # buffer of merged root segments
                            fork[5] = blob
                            blob[2].append(fork)
                        blob[3] -= 1    # open_segments -= 1 due to merged blob shared seg

        new_seg_.append(seg)
    return new_seg_
    # ---------- form_seg_() end --------------------------------------------------------------------------------------------

def form_blob(term_seg, frame):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "
    params, P_, roots, fork_, blob = term_seg[1:]

    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1    # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame

        blob.pop()
        s, blob_params, e_ = blob

        for seg in e_:
            seg.pop()   # remove references of blob

        # frame P are to compute averages, redundant for same-scope alt_frames

        frame[0] = [par1 + par2 for par1, par2 in zip(frame[0], blob_params[4:])]
        frame[1].append(nt_blob(sign=s, params=blob_params, e_=e_, sub_blob=[]))
    # ---------- form_blob() end ----------------------------------------------------------------------------------------
