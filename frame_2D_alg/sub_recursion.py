from itertools import zip_longest
from copy import copy, deepcopy
import numpy as np

from comp_slice import PP_aves, ave_nsub, ave_g, ave_ga
from comp_slice import CP, Cptuple, CderP, CPP
from comp_slice import comp_ptuple, comp_vertuple, comp_angle, form_PP_t
from agg_convert import agg_recursion_eval
'''
comp_slice_ sub_recursion + utilities
'''
ave_rotate = 10
# n and val should be excluded?
PP_vars = ["I", "M", "Ma", "axis", "angle", "aangle", "G", "Ga", "x", "L"]

def sub_recursion_eval(root):  # for PP or dir_blob

    root_PP__, fd_ = [root.derH[-1][1]], [root.derH[-1][2] if isinstance(root, CPP) else root.derH[-1][2]]
    if isinstance(root, CPP):
        root_PP__, fd_ = [root.derH[-1][1]], [root.derH[-1][2]]
    else:
        root_PP__, fd_ = root.derH[-1][1], root.derH[-1][2]  # for blob, node and fd is node_t and fd_t

    for fd, PP_ in zip(fd_, root_PP__):
        PPm_, PPd_ = [], []

        for PP in PP_:  # fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple|vertuple?
            if fd:  PPd_ += [PP]
            else:   PPm_ += [PP]
            # fork val, rdn:
            val = PP.valt[fd]; alt_val = sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
            ave = PP_aves[fd] * (PP.rdnt[fd] + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP, fd)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs

            # root is PP:
            if isinstance(root, CPP):
                for i in 0,1:
                    root.valt[i] += PP.valt[i]  # vals
                    root.rdnt[i] += PP.rdnt[i]  # ad rdn too?
            else:  # root is Blob
                if fd: root.G += sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
                else:  root.M += PP.valt[fd]

def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert bottom-up to top-down
    P__ = comp_P_der(P__) if fd else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdnt[fd] += 1  # two-fork rdn, priority is not known?  rotate?

    cP__ = [copy(P_) for P_ in P__]
    sub_PPm_, sub_PPd_ = form_PP_t(cP__, base_rdn=PP.rdnt[PP.derH[-1][2]])
    sub_recursion_eval(PP)  # add layers to select sub_PPs

# __Ps compared in rng+ can be mediated through multiple layers of _Ps, with results are summed in derQ of the same link_[0].
# The number of layers is represented in corresponding PP.rng.
# mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)? gap: neg_olp, ave = olp-neg_olp?
# draft:

def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for _P_ in P__[:-rng]:  # higher compared row, skip last rng: no lower comparand rows
        for _P in _P_:
            for derP in _P.downlink_layers[-1][0]:
                _P, P = derP._P, derP.P
                # if single P.link_: keep going up(down) beyond last compared row?
                if isinstance(P.ptuple, Cptuple):
                    vertuple = comp_ptuple(_P.ptuple, P.ptuple)
                    derP.derQ = [vertuple]; derP.valt = copy(vertuple.valt); derP.rdnt = copy(vertuple.rdnt)
                else:
                    comp_layer(derP, 0, min(len(_P.ptuple)-1,len(P.ptuple)-1))  # this is actually comp derH, works the same here
                derP.L = len(_P.dert_)
    return P__


def comp_P_der(P__):  # der+ sub_recursion in PP.P__, over the same derPs

    if isinstance(P__[0][0].ptuple, Cptuple):
        for P_ in P__:
            for P in P_:
                P.ptuple = [P.ptuple]

    for P_ in P__[1:]:  # exclude 1st row: no +ve uplinks
        for P in P_:
            for derP in P.link_[1]:  # fd=1
                _P, P = derP._P, derP.P
                i= len(derP.derQ)-1
                j= 2*i
                if len(_P.ptuple)>j-1 and len(P.ptuple)>j-1: # ptuples are derHs, extend derP.derQ:
                    comp_layer(derP, i,j)

    return P__

def comp_layer(derP, i,j):  # list derH and derQ, single der+ count=elev, eval per PP

    for _ptuple, ptuple in zip(derP._P.ptuple[i:j], derP.P.ptuple[i:j]):  # P.ptuple is derH

        if isinstance(_ptuple, CQ):
            dtuple = comp_vertuple(_ptuple, ptuple)
        else:
            dtuple = comp_ptuple(_ptuple, ptuple)
        derP.derQ += [dtuple]
        for k in 0, 1:
            derP.valt[k] += dtuple.valt[k]
            derP.rdnt[k] += dtuple.rdnt[k]


def rotate_P_(P__, dert__, mask__):  # rotate each P to align it with direction of P gradient

    yn, xn = dert__[0].shape[:2]
    for P_ in P__:
        for P in P_:
            daxis = P.ptuple.angle[0] / P.ptuple.L  # dy: deviation from horizontal axis
            while P.ptuple.G * abs(daxis) > ave_rotate:
                P.ptuple.axis = P.ptuple.angle
                rotate_P(P, dert__, mask__, yn, xn)  # recursive reform P along new axis in blob.dert__
                _, daxis = comp_angle(P.ptuple.axis, P.ptuple.angle)
            # store P.daxis to adjust params?

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
        if P not in P__[current_ys.index(P.y0)]:
            P__[current_ys.index(P.y0)].append(P)  # append P row
    elif P.y0 > current_ys[0]:  # P.y0 > largest y in ys
        P__.insert(0, [P])
    elif P.y0 < current_ys[-1]:  # P.y0 < smallest y in ys
        P__.append([P])
    elif P.y0 < current_ys[0] and P.y0 > current_ys[-1]:  # P.y0 in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y0 > y: P__.insert(i, [P])  # PP.P__.insert(P.y0 - current_ys[-1], [P])


def copy_P(P, Ptype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not Ptype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):     Ptype = 2
        elif isinstance(P, CderP): Ptype = 1
        else:                      Ptype = 0  # CP

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.P__ = [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P, P.mlevels, P.dlevels = [], [], [], []  # reset

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
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P

# draft, copied from sum2PP
def sum2PPP(qPPP, base_rdn, fd):  # sum PP_segs into PP

    PP_, val, _ = qPPP
    from sub_recursion import append_P
    # init:
    P = P_[0]
    derH = [[P.link_t[fd][0], P.link_t[fd], fd]]
    if len(P.link_t[fd])>1: sum_links(derH, P.link_t[fd][1:])
    # ptuple_ = [P.ptuple] if isinstance(P.ptuple, Cptuple) else P.ptuple
    PP = CPP(derH=[[[P.ptuple], P_,fd]], box=[P.y0, P_[-1].y0, P.x0, P.x0 + len(P.dert_)], rdn=base_rdn)
    PP.valt[fd] = val; PP.rdnt[fd] += base_rdn
    # accum:
    for i, P in enumerate(P_):
        P.roott[fd] = PP
        if i:  # not init
            sum_links(derH, P.link_t[fd])
            sum_ptuple_(PP.derH[0][0], P.ptuple)
            # if isinstance(P.ptuple, Cptuple) else sum_derH(PP.derH, P.ptuple)
            PP.link_ = list(set(PP.link_ + P.link_))  # unique links only?
            # same for link_t?
            PP.box[0] = min(PP.box[0], P.y0)  # y0
            PP.box[2] = min(PP.box[2], P.x0)  # x0
            PP.box[3] = max(PP.box[3], P.x0 + len(P.dert_))  # xn
        # sum links into new layer
    if derH: # pack new derH
        PP.derH += derH  # 1st element of PP.derH is a derQ with only Cptuple

    return PPP