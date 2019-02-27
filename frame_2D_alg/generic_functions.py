from collections import deque

def form_blob(term_seg, frame):
    " Terminated segment is merged into continued or initialized blob (all connected segments) "
    params, P_, roots, fork_, blob = term_seg[1:]

    blob[1] = [par1 + par2 for par1, par2 in zip(params, blob[1])]
    blob[3] += roots - 1    # number of open segments

    if not blob[3]:  # if open_segments == 0: blob is terminated and packed in frame

        blob.pop()
        blob_params = blob[1]

        # frame P are to compute averages, redundant for same-scope alt_frames

        frame[0] = [par1 + par2 for par1, par2 in zip(frame[0], blob_params[4:])]
        frame[1].append(blob)
    # ---------- form_blob() end ----------------------------------------------------------------------------------------

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

                seg[1] = [par1 + par2 for par1, par2 in zip([1] + params, seg[1])]  # sum all params of P into seg, in addition to +1 in H

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
            P[1] = [par1 + par2 for par1, par2 in zip(P[1], [1, y, x] + list(dert)]

            P[2].append(dert)

            x += 1

            s = dert_[x][-1] > 0  # s = (sdert > 0)

        if P[0][0]:         # if L > 0
            P_.append(P)    # P is packed into P_

    return P_

    # ---------- form_P_() end ------------------------------------------------------------------------------------------

