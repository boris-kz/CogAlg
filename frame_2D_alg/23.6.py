def reval_P_(P__, fd):  # prune qPP by (link_ + mediated link__) val

    prune_ = []; Val=0; reval = 0  # comb PP value and recursion value

    for P_ in P__:
        for P in P_:
            P_val = 0; remove_ = []
            for link in P.link_t[fd]:
                # recursive mediated link layers eval-> med_valH:
                _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)
                # link val + mlinks val: single med order, no med_valH in comp_slice?:
                link_val = link.valT[fd] + sum([mlink.valT[fd] for mlink in link._P.link_t[fd]]) * med_decay  # + med_valH
                if link_val < vaves[fd]:
                    remove_+= [link]; reval += link_val
                else: P_val += link_val
            for link in remove_:
                P.link_t[fd].remove(link)  # prune weak links
            if P_val < vaves[fd]:
                prune_ += [P]
            else:
                Val += P_val
    for P in prune_:
        for link in P.link_t[fd]:  # prune direct links only?
            _P = link._P
            _link_ = _P.link_t[fd]
            if link in _link_:
                _link_.remove(link); reval += link.valt[fd]

    if reval > aveB:
        P__, Val, reval = reval_P_(P__, fd)  # recursion
    return [P__, Val, reval]

def sum2PP(qPP, base_rdn, fd):  # sum Ps and links into PP

    P__,_,_ = qPP  # proto-PP is a list
    PP = CPP(box=copy(P__[0][0].box), fd=fd, P__ = P__)
    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]
    # accum:
    for P_ in P__:  # top-down
        for P in P_:  # left-to-right
            P.roott[fd] = PP
            sum_ptuple(PP.ptuple, P.ptuple)
            if P.derT[0]:  # P links and both forks are not empty
                for i in 0,1:
                    if isinstance(P.valT[0], list):  # der+: H = 1fork) 1layer before feedback
                        sum_unpack([DerT[i],ValT[i],RdnT[i]], [P.derT[i],P.valT[i],P.rdnT[i]])
                    else:  # rng+: 1 vertuple
                        if isinstance(ValT[0], list):  # we init as list, so we need to change it into ints here
                            ValT = [0,0]; RdnT = [0,0]
                        sum_ptuple(DerT[i], P.derT[i]); ValT[i]+=P.valT[i]; RdnT[i]+=P.rdnT[i]
                PP.link_ += P.link_
                for Link_,link_ in zip(PP.link_t, P.link_t):
                    Link_ += link_  # all unique links in PP, to replace n
            Y0,Yn,X0,Xn = PP.box; y0,yn,x0,xn = P.box
            PP.box = [min(Y0,y0), max(Yn,yn), min(X0,x0), max(Xn,xn)]
    if DerT[0]:
        PP.derT = DerT; PP.valT = ValT; PP.rdnT = RdnT
        """
        for i in 0,1:
            PP.derT[i]+=DerT[i]; PP.valT[i]+=ValT[i]; PP.rdnT[i]+=RdnT[i]  # they should be in same depth, so bracket is not needed here
        """
    return PP

def comp_slice(blob, verbose=False):  # high-G, smooth-angle blob, composite dert core param is v_g + iv_ga

    P__ = blob.P__
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            link_,link_m,link_d = [],[],[]  # empty in initial Ps
            derT=[[],[]]; valT=[0,0]; rdnT=[1,1]  # to sum links in comp_P
            for _P in _P_:
                _L = len(_P.dert_); L = len(P.dert_); _x0=_P.box[2]; x0=P.box[2]
                # test for x overlap(_P,P) in 8 directions, all derts positive:
                if (x0 - 1 < _x0 + _L) and (x0 + L > _x0):
                    comp_P(_P,P, link_,link_m,link_d, derT,valT,rdnT, fd=0)
                elif (x0 + L) < _x0:
                    break  # no xn overlap, stop scanning lower P_
            if link_:
                P.link_=link_; P.link_t=[link_m,link_d]
                P.derT=derT; P.valT=valT; P.rdnT=rdnT  # single Mtuple, Dtuple
        _P_ = P_
    PPm_,PPd_ = form_PP_t(P__, base_rdn=2)
    blob.PPm_, blob.PPd_  = PPm_, PPd_


def form_PP_t(P__, base_rdn):  # form PPs of derP.valt[fd] + connected Ps'val

    PP_t = []
    for fd in 0,1:
        fork_P__ = ([copy(P_) for P_ in reversed(P__)])  # scan bottom-up
        qPP_ = []  # form initial sequence-PPs:
        for P_ in fork_P__:
            for P in P_:
                if not P.roott[fd]:
                    qPP = [[[P]]]  # init PP is 2D queue of Ps, + valt of all layers?
                    P.roott[fd]=qPP; valt = [0,0]
                    uplink_ = P.link_t[fd]; uuplink_ = []
                    # next-line links for recursive search
                    while uplink_:
                        for derP in uplink_:
                            _P = derP._P; _qPP = _P.roott[fd]
                            if _qPP:
                                for i in 0, 1: valt[i] += _qPP[1][i]
                                merge(qPP,_qPP[0], fd); qPP_.remove(_qPP)  # merge P__s
                            else:
                                qPP[0].insert(0,_P)  # pack top down
                                _P.root[fd] = qPP
                                for i in 0,1: valt[i] += np.sum(derP.valT[i])
                                uuplink_ += derP._P.link_t[fd]
                        uplink_ = uuplink_
                        uuplink_ = []
                    qPP_ += [qPP + [valt,ave+1]]  # ini reval=ave+1
        # prune qPPs by med links val:
        rePP_= reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]
        PP_t += [CPP_]  # may be empty

    return PP_t  # add_alt_PPs_(graph_t)?

def reval_PP_(PP_, fd):  # recursive eval / prune Ps for rePP

    rePP_ = []
    while PP_:  # init P__
        P__, valt, reval = PP_.pop(0)
        Ave = ave * 1+(valt[fd] < valt[1-fd])  # * PP.rdnT[fd]?
        if valt[fd] > Ave:
            if reval < Ave:  # same graph, skip re-evaluation:
                rePP_ += [[P__,valt,0]]  # reval=0
            else:
                rePP = reval_P_(P__, fd)  # recursive node and link revaluation by med val
                if valt[fd] > Ave:  # min adjusted val
                    rePP_ += [rePP]
    if rePP_ and max([rePP[2] for rePP in rePP_]) > ave:  # recursion if any min reval:
        rePP_ = reval_PP_(rePP_, fd)

    return rePP_

# draft
def merge(qPP, _P__, fd):  # the P__s should not have shared Ps

    P__ = qPP[0]
    for _P_ in _P__:
        for _P in _P_:
            _P.roott[fd] = qPP
            ys = [P_[0].box[0] for P_ in P__]  # list of current-layer seg rows
            _y0 = _P.box[0]
            if _y0 in ys:  # append P row in an existing P_
                # pack P into P_ based on their x：
                ry = _y0-min(ys)  # relative y, we can't pack based on y0 because here is just a section of P__
                P_ = P__[ry]
                # current P x0
                cx0 = _P.box[2]
                # prior P's x0 in P_
                _x0 = P_[0].box[2]
                if cx0 < _x0:  # cP's x is smaller than 1st P in P_
                    P_.insert(0, _P)
                elif cx0 > P_[-1].box[2]:  # P's x is larger than last P in P_
                    P_ += [_P]
                else:  # P's x is somewhere between P_
                    for i, P in enumerate(P_[1:], start=1):
                        x0 = P.box[2]  # current P's x0
                        if cx0>_x0 and cx0 < x0:
                            P_.insert(i,_P)
                            break
                        _x0 = x0

            elif _y0 < ys[0]:  # _P.y0 < smallest y in ys
                P__.insert(0, [_P])
            elif _y0 > ys[-1]:  # _P.y0 > largest y in ys
                P__ += [[_P]]
            else:  # _P.y0 doesn't exist in existing P__ but in between P__
                prior_y0 = ys[0]
                for current_y0 in ys[1:]:
                    if _y0 > prior_y0 and _y0 < current_y0:
                        P__.insert(current_y0, [_P])
                    prior_y0 = current_y0
'''
    for dert, roots in up_rim_:
        if roots:
            for rdn, up_P in enumerate(sorted(roots, key=roots.ptuple[5] or get_G)):  # sort by G, rdn for lower-G _Ps only
                if up_P.ptuple[5] > ave*(rdn+1):  # rdn: up-fork redundancy
                    P.link_ += [up_P]  # represent uplinks only
                    form_link_(up_P, blob)
        elif dert[9] > ave*len(up_rim_):  # no adj root in strong dert, -= fork redundancy?
            up_P = rotate_P(None, blob.dert__, blob.mask__, ave, center=[y,x,dert])  # form new P from central dert
            if up_P.ptuple[5] > ave * (rdn + 1):  # rdn: up-fork redundancy
                P.link_ += [up_P]  # represent uplinks only
                form_link_(up_P, blob)
'''
def rotate_P(P, dert__, mask__, ave_a, center=None):

    dert_ext_ = []  # new P.dert_ext_, or not used in rotation?

    if center:  # form new P from central dert
        ycenter, xcenter, dert = center
        sin,cos = dert[3]/dert[9], dert[4]/dert[9]  # dy,dx / G
        P = CP()
    else:  # rotate arg P
        if ave_a is None: sin,cos = np.divide(P.ptuple[3], P.ptuple[5])
        else:             sin,cos = np.divide(ave_a, np.hypot(*ave_a))
        if cos < 0: sin,cos = -sin,-cos  # dx always >= 0, dy can be < 0
        y0,yn,x0,xn = P.box
        ycenter = (y0+yn)/2; xcenter = (x0+xn)/2
    new_axis = sin, cos
    rdert_ = []
    # scan left:
    rx=xcenter; ry=ycenter
    while True:  # terminating condition is in form_rdert()
        rdert = form_rdert(rx,ry, dert__, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ = [rdert] + rdert_  # append left
        rx-=cos; ry-=sin  # next rx,ry
        dert_ext_ += [[[P],ry,rx]] + dert_ext_  # append left external params: roots and coords per dert
    x0 = rx; yleft = ry
    # scan right:
    rx=xcenter+cos; ry=ycenter+sin  # center dert was included in scan left
    while True:
        rdert = form_rdert(rx,ry, dert__, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ += [rdert]  # append right
        rx+=cos; ry+=sin  # next rx,ry
        dert_ext_ += [[[P],ry,rx]]
    # form rP:
    if not rdert_: return
    rdert = rdert_[0]  # initialization:
    G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        dert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1); L = len(rdert_)
    # replace P:
    P.ptuple = [I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L]
    P.dert_ = dert_
    P.dert_ext_ = dert_ext_
    P.box = [min(yleft, ry), max(yleft, ry), x0, rx]  # P may go up-right or down-right
    P.axis = new_axis

    if center: return P

def Dert2P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, G, Ga, L, y, x, Pdert_, dert_roots__):

    P = CP(ptuple=[I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L], box=[y, y, x-L, x-1], dert_=Pdert_)

    # or after rotate?
    center_x = int((P.box[2] + P.box[3]) /2)
    P.dert_ext_ = [[[P], y, center_x] for dert in Pdert_]

    bx = P.box[2]
    while bx <= P.box[3]:
        dert_roots__[y][bx] += [P]
        bx += 1

    return P

def r