'''
comp_slice_ sub_recursion + utilities
'''
import cv2
from itertools import zip_longest
from copy import copy, deepcopy
import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice import *

ave_M = -500  # high negative ave M for high G blobs


def sub_recursion_eval(root):  # for PP or dir_blob

    if isinstance(root, CPP): root_PPm_, root_PPd_ = root.rlayers[0], root.dlayers[0]
    else:                     root_PPm_, root_PPd_ = root.PPm_, root.PPd_

    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:
            if fd:  # add root to derP for der+:
                for P_ in PP.P__[1:-1]:  # skip 1st and last row
                    for P in P_:
                        for derP in P.uplink_layers[-1][fd]:
                            derP.roott[fd] = PP
                comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:
                comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]

            val = PP.players.valt[fd]; alt_val = PP.players.valt[1-fd]  # for fork rdn:
            ave = PP_aves[fd] * (PP.rdn + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # 1+PP.rdn incr
                # splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

            # segs agg_recursion:
            agg_recursion_eval(PP, [copy(PP.mseg_levels[-1]), copy(PP.dseg_levels[-1])])
            # include empty comb_layers:
            if fd: root.dlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]
            else:  root.rlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]

            # or higher der val?
            if isinstance(root, CPP):  # root is CPP
                root.players.valt[fd] += PP.players.valt[fd]
            else:  # root is CBlob
                if fd: root.G += PP.players.valt[1]
                else:  root.M += PP.players.valt[0]

def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    P__ = comp_P_der(P__) if PP.players.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdn += 2  # two-fork rdn, priority is not known?  rotate?

    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.players.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.players.fds)  # returns bottom-up
    # sub_PPm_, sub_PPd_:
    PP.rlayers[0], PP.dlayers[0] = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)
    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs


def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
            P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3][0]:  # fd always = 0 here
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    return P__

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        # not revised:
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1][1]:  # fd=1
                for derP in P.downlink_layers[-1][1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP.players,
                    # or comp derP.players[1] only: it's already diffs of all lower players?
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs_ += [dderP]  # actually it could be dderPs_ ++ [derPP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    return dderPs__


def rotate_P_(P__, dert__, mask__):  # rotate each P to align it with direction of P gradient

    yn, xn = dert__[0].shape[:2]
    for P_ in P__:
        for P in P_:
            daxis = P.ptuple.angle[0] / P.ptuple.L  # dy: deviation from horizontal axis
            while P.ptuple.G * abs(daxis) > ave_rotate:
                P.ptuple.axis = P.ptuple.angle
                rotate_P(P, dert__, mask__, yn, xn)  # recursive reform P along new axis in blob.dert__
                _, daxis = comp_angle("axis", P.ptuple.axis, P.ptuple.angle)
            # store as P.daxis to adjust params?

def rotate_P(P, dert__t, mask__, yn, xn):

    L = len(P.dert_)
    rdert_ = [P.dert_[int(L/2)]]  # init rotated dert_ with old central dert

    ycenter = int(P.y0 + P.ptuple.axis[0]/2)  # can be negative
    xcenter = int(P.x0 + abs(P.ptuple.axis[1]/2))  # always positive
    Dy, Dx = P.ptuple.angle
    dy = Dy/L; dx = abs(Dx/L)  # hypot(dy,dx)=1: each dx,dy adds one rotated dert|pixel to rdert_
    # scan left:
    rx=xcenter-dx; ry=ycenter-dy; rdert=1  # to start while:
    while rdert and rx>=0 and ry>=0 and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert:
            rdert_.insert(0, rdert)
        rx += dx; ry += dy  # next rx, ry
    P.x0 = rx+dx; P.y0 = ry+dy  # revert to leftmost
    # scan right:
    rx=xcenter+dx; ry=ycenter+dy; rdert=1  # to start while:
    while rdert and ry>=0 and np.ceil(rx)<xn and np.ceil(ry)<yn:
        rdert = form_rdert(rx,ry, dert__t, mask__)
        if rdert:
            rdert_ += [rdert]
            rx += dx; ry += dy  # next rx,ry
    # form rP:
    # initialization:
    rdert = rdert_[0]; _, G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; ndert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        _, g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        ndert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1)
    ptuple = Cptuple(I=I, M=M, G=G, Ma=Ma, Ga=Ga, angle=(Dy,Dx), aangle=(Sin_da0, Cos_da0, Sin_da1, Cos_da1))
    # add n,val,L,x,axis?
    # replace P:
    P.ptuple=ptuple; P.dert_=ndert_


def form_rdert(rx,ry, dert__t, mask__):

    # coord, distance of four int-coord derts, overlaid by float-coord rdert in dert__, int for indexing
    # always in dert__ for intermediate float rx,ry:
    x1 = int(np.floor(rx)); dx1 = abs(rx - x1)
    x2 = int(np.ceil(rx));  dx2 = abs(rx - x2)
    y1 = int(np.floor(ry)); dy1 = abs(ry - y1)
    y2 = int(np.ceil(ry));  dy2 = abs(ry - y2)

    # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1?
    # approximation, square of rpixel is rotated, won't fully match not-rotated derts
    mask = mask__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
         + mask__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
         + mask__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
         + mask__[y2, x2] * (1 - np.hypot(dx2, dy2))
    mask = int(mask)  # summed mask is fractional, round to 1|0
    if not mask:
        ptuple = []
        for dert__ in dert__t:  # 10 params in dert: i, g, ga, ri, dy, dx, day0, dax0, day1, dax1
            param = dert__[y1, x1] * (1 - np.hypot(dx1, dy1)) \
                  + dert__[y2, x1] * (1 - np.hypot(dx1, dy2)) \
                  + dert__[y1, x2] * (1 - np.hypot(dx2, dy1)) \
                  + dert__[y2, x2] * (1 - np.hypot(dx2, dy2))
            ptuple += [param]
        return ptuple


def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y0 for P_ in P__]  # list of current-layer seg rows
    if P.y0 in current_ys:
        P__[current_ys.index(P.y0)].append(P)  # append P row
    elif P.y0 > current_ys[0]:  # P.y0 > largest y in ys
        P__.insert(0, [P])
    elif P.y0 < current_ys[-1]:  # P.y0 < smallest y in ys
        P__.append([P])
    elif P.y0 < current_ys[0] and P.y0 > current_ys[-1]:  # P.y0 in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y0 > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])


def copy_P(P, iPtype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not iPtype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):
            Ptype = 2
        elif isinstance(P, CderP):
            Ptype = 1
        elif isinstance(P, CP):
            Ptype = 0
    else:
        Ptype = iPtype

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        rlayers, dlayers = P.rlayers, P.dlayers
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.rlayers, P.dlayers, P.P__ = [], [], [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        rlayers, dlayers = P.rlayers, P.dlayers
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P.rlayers, P.dlayers, P.mlevels, P.dlevels = [], [], [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        P.rlayers, P.dlayers = rlayers, dlayers
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.rlayers, P.dlayers = rlayers, dlayers
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P


def CBlob2graph(blob, fseg, Cgraph):

    PPm_ = blob.PPm_; PPd_ = blob.PPd_
    root_fds = PPm_[0].players.fds[:-1]  # root fds is the shorter fork?
    root = Cgraph(
        node_=PPm_, alt_graph_=PPd_, rng = PPm_[0].rng, fds = root_fds, rdn=blob.rdn, x0=PPm_[0].x0, xn=PPm_[0].xn, y0=PPm_[0].y0, yn=PPm_[0].yn)

    for fd, PP_ in [PPm_, PPd_]:
        for PP in PP_:
            # mostly wrong below:
            if fd:  # sum from altPP's players
                for altPP in PP.altPP_:
                    sum_pH(alt_players.H, altPP.players.H)
                    alt_valt[0] += altPP.players.valt[0]; alt_valt[1] += altPP.players.valt[1]
                    alt_fds[:] = deepcopy(altPP.players.fds)
            else:  # sum from players
                sum_pH(players.H, PP.players.H)
                valt[0] += PP.players.valt[0]; valt[1] += PP.players.valt[1]
                fds[:] = deepcopy(PP.players.fds)

            # compute rdn
            if fseg: PP = PP.roott[PP.fds[-1]]  # seg root
            PP_P_ = [P for P_ in PP.P__ for P in P_]  # PPs' Ps
            for altPP in PP.altPP_:  # overlapping Ps from each alt PP
                altPP_P_ = [P for P_ in altPP.P__ for P in P_]  # altPP's Ps
                alt_rdn = len(set(PP_P_).intersection(altPP_P_))
                PP.alt_rdn += alt_rdn  # count overlapping PPs, not bilateral, each PP computes its own alt_rdn
                root.alt_rdn += alt_rdn  # sum across PP_

            # convert and pack gPP
            gPP_ += [CPP2graph(PP, fseg, Cgraph)]
            root.x0=min(root.x0, PP.x0)
            root.xn=max(root.xn, PP.xn)
            root.y0=min(root.y0, PP.y0)
            root.yn=max(root.yn, PP.yn)
    return root

def CPP2graph(PP, fseg, Cgraph):

    alt_players = CpH()
    if not fseg and PP.altPP_:  # seg doesn't have altPP_
        alt_players.fds = PP.altPP_[0].players.fds
        for altPP in PP.altPP_[1:]:  # get fd sequence common for all altPPs:
            for i, (_fd, fd) in enumerate(zip(alt_players.fds, altPP.players.fds)):
                if _fd != fd:
                    alt_players.fds = alt_players.fds[:i]
                    break
        for altPP in PP.altPP_:
            sum_pH(alt_players.H, altPP.players.H[:len(alt_players.fds)])  # sum same-fd players only
            alt_players.valt[0] += altPP.players.valt[0];  alt_players.valt[1] += altPP.players.valt[1]
    # graph: plevels ( players ( ptuples:
    players = deepcopy(PP.players)
    caTree = CpH(H=[players], valt=deepcopy(players.valt), fds=deepcopy(players.fds))
    alt_caTree = CpH(H=[alt_players], valt=deepcopy(alt_players.valt), fds=deepcopy(players.fds))

    plevel = CpH(H = [caTree],  valt=deepcopy(caTree.valt))
    alt_plevel = CpH(H = [alt_caTree],  valt=deepcopy(alt_caTree.valt))
    x0 = PP.x0; xn = PP.xn; y0 = PP.y0; yn = PP.yn

    return Cgraph(
        node_=PP.P__, plevels=plevel, alt_plevels=alt_plevel, angle=(yn-y0,xn-x0), sparsity=1.0, x0=x0, xn=xn, y0=y0, yn=yn)
        # 1st plevel fd is always der+?