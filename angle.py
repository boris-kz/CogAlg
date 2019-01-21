import math
from collections import deque
# Constrains ---------------------------------------------------------------------
from constrains import ave
from constrains import angle_coeff
# --------------------------------------------------------------------------------
'''
    angle is a component of intra_blob
'''

def comp_angle(blob, dert__):  # compute and compare angle, define ablobs, accumulate a, da, sda in all reps within gblob
    ''' - Sort list of segments (root_) based on their top line P's coordinate (segment's min_y)    <---------------------------------------|
        - Iterate through each line in the blob (from blob's min_y to blob's max_y):                                                        |
            + Have every segment that contains current-line P in a list (seg_). This action is simplified by sorting step above  -----------|
            + Extract current-line slice of the blob - or the list of every P of this line (P_)
            + Have every out-of-bound segment removed from list (seg_)
            + Perform angle computing, comparison and clustering in every dert in P_ '''
    params, root_ = blob[2:4]
    # sorting based on min_y values
    blob[3] = sorted(root_, key=lambda segment: segment[1][2])  # sorted by min_y
    # init:
    params += [0, 0]# A, sDa
    blob[4] = []    # ablob_
    global y
    y = blob[1][2]  # start from top-line of the blob
    seg_ = []       # for buffering of segments that contain current line
    haP_ = []       # for buffering of higher-line aPs
    i = 0           # iterator in root_
    while y <= blob[1][3]:  # while y <= blob's max_y
        P_ = []             # buffering of current line aPs
        # preparing (discontinuous) line y of derts in blob
        while i < len(root_) and root_[i][1][2] == y:
            seg_.append([root_[i], 0])      # runningSegment consists of segments that contains y-line P and that P's index
            i += 1

        ii = 0
        while ii < len(seg_):       # for every segment that contains y-line P
            seg, iP = seg_[ii]      # P = Py_[ii][0] = seg[3][iP][0]
            P_.append(seg[3][iP][0])
            if y == seg[1][3]:      # if y has reached segment's bottom
                seg_.pop(ii)        # remove from list
                ii -= 1
            else:
                seg_[ii][1] += 1    # index point to next-line P
            ii += 1

        # actual comp_angle:
        aP_ = deque()
        for P in P_:
            [min_x, max_x], L, dert_ = P[1], P[2][0], P[3]
            aP = [-1, [min_x, -1], [0, 0, 0], []]
            buff_ = deque()
            # init previous horizontal pixel's angle:
            if min_x == 1:  # no previous horizontal pixel's angle
                _a = 0      # this may not be the best value, needs further consideration
            else:
                _dert = dert__[y][min_x - 1]
                if len(_dert) < 5:             # angle hasn't been computed for this pixe
                    dx, dy = _dert[-2:]
                    _a = math.atan2(dy, dx) * angle_coeff + 128    # angle label: 0 to 255 <--> -pi to pi in radian
                else:
                    _a = _dert[4]
            # init previous vertical pixel's angle
            if y == 1:  # no previous vertical pixel's angle
                _dert_ = [(0, 0, 0, 0, 0)] * L            # create a zero _dert_
            else:
                _dert_ = dert__[y - 1][min_x:max_x+1]   # get corresponding higher-line dert_
            x = min_x
            for dert, _dert in zip(dert_, _dert_):
                dx, dy = dert[2:]
                a = math.atan2(dy, dx) * angle_coeff + 128
                if len(_dert) < 5:
                    _dx, _dy = _dert[2:]
                    __a = math.atan2(_dy, _dx) * angle_coeff + 128
                else:
                    __a = _dert[4]
                sda = abs(a - _a) + abs(a - __a) - 2 * ave
                dert += a, sda
                aP = form_aP(dert, x, max_x, aP, aP_, buff_, haP_, blob)
                _a = a
                x += 1
                # ...to next dert/pixel in line...
            # ...to next P in line...
        # buffers for next line
        haP_ = aP_
        y += 1
        # ...to next line...
    # ---------- comp_angle() end ---------------------------------------------------------------------------------------

def form_aP(dert, x, x_stop, aP, aP_, buff_, haP_, blob):
    a, sda = dert[-2:]
    s = 1 if sda > 0 else 0
    pri_s = aP[0]
    if s != pri_s and pri_s != -1:  # aP is terminated:
        aP[1][1] = x - 1  # aP's max_x
        scan_aP_(aP, aP_, buff_, haP_, blob)  # aP scans haP_
        aP = [s, [x, -1], [0, 0, 0], []]  # new aP initialization

    [min_x, max_x], [L, A, sDa], dert_ = aP[1:]  # continued or initialized input and derivatives are accumulated:
    L += 1  # length of a pattern
    A += a  # summed angle
    sDa += sda  # summed sda
    dert_.append(dert)  # der2s are buffered for oriented rescan and incremental range | derivation comp
    aP = [s, [min_x, max_x], [L, A, sDa], dert_]

    if x == x_stop:  # aP is terminated:
        aP[1][1] = x  # aP's max_x
        scan_aP_(aP, aP_, buff_, haP_, blob)
    return aP  # accumulated within line, P_ is a buffer for conversion to _P_
    # ---------- form_aP() end ------------------------------------------------------------------------------------------

def scan_aP_(aP, aP_, _buff_, haP_, blob):
    fork_ = []  # refs to haPs connected to input aP
    _min_x = 0  # to start while loop, next ini_x = _x + 1
    min_x, max_x = aP[1]
    while _min_x <= max_x:  # while x values overlap between aP and _aP
        if _buff_:
            haP = _buff_.popleft()  # haP was extended to segment and buffered in prior scan_aP_
        elif haP_:
            haP = form_asegment(haP_.popleft(), blob)
        else:
            break  # higher line ends, all haPs are converted to segments
        roots = haP[4]
        _aP = haP[3][-1][0]
        _min_x, _max_x = _aP[1]  # first_x, last_x
        if aP[0] == _aP[0] and min_x <= _max_x and _min_x <= max_x:
            roots += 1
            haP[4] = roots
            fork_.append(haP)  # aP-connected haPs will be converted to segments at each _fork
        if _max_x > max_x:  # x overlap between haP and next aP: haP is buffered for next scan_aP_, else haP included in a blob segment
            _buff_.append(haP)
        elif roots != 1:
            form_ablob(haP, blob)  # segment is terminated and packed into its blob
        _min_x = _max_x + 1  # = first x of next _aP
    aP_.append((aP, fork_))  # aP with no overlap to next _aP is extended to haP and buffered for next-line scan_aP_
    # ---------- scan_aP_() end -----------------------------------------------------------------------------------------

def form_asegment(haP, blob):
    _aP, fork_ = haP
    s, [min_x, max_x], params = _aP[:-1]
    ave_x = (params[0] - 1) // 2  # extra-x L = L-1 (1x in L)
    if not fork_:  # seg is initialized with initialized blob (params, coordinates, incomplete_segments, root_, xD)
        ablob = [s, [min_x, max_x, y - 1, -1, 0, 0, 0], [0, 0, 0], [], 1]  # s, coords, params, root_, incomplete_segments
        haP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_aP, 0)], 0, fork_, ablob]
        ablob[3].append(haP)
    else:
        if len(fork_) == 1 and fork_[0][4] == 1:  # haP has one fork: haP[2][0], and that fork has one root: haP
            # haP is merged into higher-line blob segment (Pars, roots, _fork_, ave_x, xD, aPy_, blob) at haP[2][0]:
            fork = fork_[0]
            fork[1][0] = min(fork[1][0], min_x)
            fork[1][1] = max(fork[1][1], max_x)
            xd = ave_x - fork[1][5]
            fork[1][4] += xd
            fork[1][5] += abs(xd)
            fork[1][6] = ave_x
            L, A, sDa = params
            Ls, As, sDas = fork[2]  # seg params
            fork[2] = [Ls + L, As + A, sDas + sDa]
            fork[3].append((_aP, xd))  # aPy_: vertical buffer of aPs merged into seg
            fork[4] = 0  # reset roots
            haP = fork  # replace segment with including fork's segment
            ablob = haP[6]
        else:  # if >1 forks, or 1 fork that has >1 roots:
            haP = [s, [min_x, max_x, y - 1, -1, 0, 0, ave_x], params, [(_aP, 0)], 0, fork_, fork_[0][6]]  # seg is initialized with fork's blob
            ablob = haP[6]
            ablob[3].append(haP)  # segment is buffered into root_
            if len(fork_) > 1:  # merge blobs of all forks
                if fork_[0][4] == 1:  # if roots == 1
                    form_ablob(fork_[0], blob, 1)  # merge seg of 1st fork into its blob
                for fork in fork_[1:len(fork_)]:  # merge blobs of other forks into blob of 1st fork
                    if fork[4] == 1:
                        form_ablob(fork, blob, 1)
                    if not fork[6] is ablob:
                        [min_x, max_x, min_y, max_y, xD, abs_xD, Ly], [L, A, sDa], root_, incomplete_segments = fork[6][1:]  # ommit sign
                        ablob[1][0] = min(min_x, ablob[1][0])
                        ablob[1][1] = max(max_x, ablob[1][1])
                        ablob[1][2] = min(min_y, ablob[1][2])
                        ablob[1][4] += xD
                        ablob[1][5] += abs_xD
                        ablob[1][6] += Ly
                        ablob[2][0] += L
                        ablob[2][1] += A
                        ablob[2][2] += sDa
                        ablob[4] += incomplete_segments
                        for seg in root_:
                            if not seg is fork:
                                seg[6] = ablob  # blobs in other forks are references to blob in the first fork
                                ablob[3].append(seg)  # buffer of merged root segments
                        fork[6] = ablob
                        ablob[3].append(fork)
                    ablob[4] -= 1
        ablob[1][0] = min(min_x, ablob[1][0])
        ablob[1][1] = max(max_x, ablob[1][1])
    return haP
    # ---------- form_asegment() end ------------------------------------------------------------------------------------

def form_ablob(term_seg, blob, y_carry=0):
    [min_x, max_x, min_y, max_y, xD, abs_xD, ave_x], [L, A, sDa], Py_, roots, fork_, ablob = term_seg[1:]  # ignore sign
    ablob[1][4] += xD  # ave_x angle, to evaluate blob for re-orientation
    ablob[1][5] += len(Py_)  # Ly = number of slices in segment
    ablob[2][0] += L
    ablob[2][1] += A
    ablob[2][2] += sDa
    ablob[4] += roots - 1  # reference to term_seg is already in blob[9]
    term_seg[1][3] = y - 1 - y_carry  # y_carry: min elevation of term_seg over current hP
    if not ablob[4]:
        ablob[1][3] = term_seg[1][3]
        blob[2][-2] += A    # params: A
        blob[2][-1] += sDa  # params: sDa
        blob[4].append(ablob)
    # ---------- form_ablob() end ---------------------------------------------------------------------------------------