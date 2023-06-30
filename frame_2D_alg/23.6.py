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

def vectorize_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    slice_blob(blob, verbose=verbose)  # form 2D array of Ps: horizontal blob slices in dert__
    rotate_P_(blob)  # re-form Ps around centers along P.G, P sides may overlap, if sum(P.M s + P.Ma s)?
    cP_ = copy(blob.P_)  # for popping here, in scan_rim
    while cP_:
        form_link_(cP_.pop(0), cP_, blob)  # trace adjacent Ps, fill|prune if missing or redundant, add them to P.link_

    comp_slice(blob, verbose=verbose)  # scan rows top-down, compare y-adjacent, x-overlapping Ps to form derPs
    for fd, PP_ in enumerate([blob.PPm_, blob.PPd_]):
        sub_recursion_eval(blob, PP_, fd=fd)  # intra PP, no blob fb
        # compare PPs, cluster in graphs:
        if sum([PP.valt[fd] for PP in PP_]) > ave * sum([PP.rdnt[fd] for PP in PP_]):
            agg_recursion_eval(blob, copy(PP_), fd=fd)  # comp sub_PPs, form intermediate PPs
'''
or only compute params needed for rotate_P_?
'''
def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert ptuple of 2D arrays to ptuple of 1D arrays: blob rows
    dert__ = [list(zip(*dert_)) for dert_ in dert__]  # convert ptuple of 1D arrays to 2D array of ptuples
    blob.dert__ = dert__
    P_ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        if verbose: print(f"\rConverting to image... Processing line {y + 1}/{height}", end=""); sys.stdout.flush()
        _mask = True  # mask -1st dert
        x = 0
        for (i, *dert), mask in zip(dert_, mask_):
            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    I = ri; M = ave_g - g; Ma = ave_ga - ga; Dy = dy; Dx = dx
                    Sin_da0, Cos_da0, Sin_da1, Cos_da1 = sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params:
                    I +=ri; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx  # angle
                    Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1  # aangle
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, pack P:
                P_ += [term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x, Pdert_)]
            _mask = mask
            x += 1
        if not _mask:  # pack last P:
            P_ += [term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x, Pdert_)]

    if verbose: print("\r", end="")
    blob.P_ = P_
    return P_

def term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x, Pdert_):

    G = np.hypot(Dy, Dx); Ga = (Cos_da0 + 1) + (Cos_da1 + 1)  # recompute G,Ga, it can't reconstruct M,Ma
    L = len(Pdert_)  # params.valt = [params.M+params.Ma, params.G+params.Ga]?
    return CP(ptuple=[I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L], box=[y,y, x-L,x-1], dert_=Pdert_)

def rotate_P_(blob):  # rotate each P to align it with direction of P gradient

    P_, dert__, mask__ = blob.P_, blob.dert__, blob.mask__

    for P in P_:
        G = P.ptuple[5]
        daxis = P.ptuple[3][0] / G  # dy: deviation from horizontal axis
        _daxis = 0
        while abs(daxis) * G > ave_rotate:  # recursive reform P in blob.dert__ along new G angle:
            rotate_P(P, dert__, mask__, ave_a=None)  # rescan in the direction of ave_a, if any, P.daxis for future reval?
            maxis, daxis = comp_angle(P.ptuple[3], P.axis)
            ddaxis = daxis +_daxis  # cancel-out if opposite-sign
            if ddaxis * P.ptuple[5] < ave_rotate:  # terminate if oscillation
                rotate_P(P, dert__, mask__, ave_a=np.add(P.ptuple[3], P.axis))  # rescan in the direction of ave_a, if any
                break
        for _, y,x in P.dert_ext_:
            blob.dert_roots__[int(y)][int(x)] += [P]  # final rotated P

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
        dert_ext_.insert(0, [[[P],ry,rx]])  # append left external params: roots and coords per dert
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

    if center: return P  # when forming new P from dert

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derT,valT,rdnT in blob

    term = 1
    for PP in PP_:
        if np.sum(PP.valT[fd][-1]) > PP_aves[fd] * np.sum(PP.rdnT[fd][-1]) and len(PP.P__) > ave_nsub:
            term = 0
            sub_recursion(PP, fd)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[[PP.derT[fd][-1]], PP.valT[fd][-1],PP.rdnT[fd][-1]]]  # [derT,valT,rdnT]
            # feedback last layer, added in sum2PP
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only


def feedback(root, fd):  # append new der layers to root

    Fback = root.fback_.pop()  # init with 1st fback derT,valT,rdnT
    while root.fback_:
        sum_unpack(Fback, root.fback_.pop())  # sum | append fback in Fback
    derT,valT,rdnT = Fback
    for i in 0,1:
        root.derT[i]+=derT[i]; root.valT[i]+=valT[i]; root.rdnT[i]+=rdnT[i]  # concat Fback layers to root layers

    if isinstance(root.roott[fd], CPP):
        root = root.roott[fd]
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.P__[fd]):  # all nodes term, fed back to root.fback_
            feedback(root, fd)  # derT=[1st layer] in sum2PP, deeper layers(forks appended by recursive feedback


def sub_recursion(PP, fd):  # evaluate PP for rng+ and der+, add layers to select sub_PPs

    if fd:
        [[nest(P) for P in P_] for P_ in PP.P__]  # add layers and forks?
        P__ = comp_der(PP.P__)  # returns top-down
        PP.rdnT[fd][-1][-1][-1] += np.sum(PP.valT[fd][-1]) > np.sum(PP.valT[1 - fd][-1])
        base_rdn = PP.rdnT[fd][-1][-1][-1]  # link Rdn += PP rdn?
    else:
        P__ = comp_rng(PP.P__, PP.rng + 1)
        PP.rdnT[fd] += PP.valT[fd] > PP.valT[1 - fd]
        base_rdn = PP.rdnT[fd]

    cP__ = [[replace(P, roott=[None, None]) for P in P_] for P_ in P__]
    PP.P__ = form_PP_t(cP__, base_rdn=base_rdn)  # P__ = sub_PPm_, sub_PPd_

    for fd, sub_PP_ in enumerate(PP.P__):
        if sub_PP_:  # der+ | rng+
            for sub_PP in sub_PP_: sub_PP.roott[fd] = PP
            sub_recursion_eval(PP, sub_PP_, fd=fd)
        '''
        if PP.valt[fd] > ave * PP.rdnt[fd]:  # adjusted by sub+, ave*agg_coef?
            agg_recursion_eval(PP, copy(sub_PP_), fd=fd)  # comp sub_PPs, form intermediate PPs
        else:
            feedback(PP, fd)  # add aggH, if any: 
        implicit nesting: rngH(derT / sub+fb, aggH(subH / agg+fb: subH is new order of rngH(derT?
        '''

# mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)? gap: neg_olp, ave = olp-neg_olp?
# __Ps: above PP.rng layers of _Ps:
def comp_rng(iP__, rng):  # form new Ps and links in rng+ PP.P__, switch to rng+n to skip clustering?

    P__ = []
    for iP_ in reversed(iP__[:-rng]):  # lower compared row, follow uplinks, no uplinks in last rng rows
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            derT,valT,rdnT = [[],[]],[0,0],[1,1]
            for iderP in P.link_t[0]:  # mlinks
                _P = iderP._P
                for _derP in _P.link_t[0]:  # next layer of mlinks
                    __P = _derP._P  # next layer of Ps
                    comp_P(P,__P, link_,link_m,link_d, derT,valT,rdnT, fd=0)
            if np.sum(valT[0]) > P_aves[0] * np.sum(rdnT[0]):
                # add new P in rng+ PP:
                P_ += [CP(ptuple=deepcopy(P.ptuple), dert_=copy(P.dert_), box=copy(P.box),
                          derT=derT, valT=valT, rdnT=rdnT, link_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__

def comp_der(iP__):  # form new Ps and links in rng+ PP.P__, extend their link.derT, P.derT, _P.derT

    P__ = []
    for iP_ in reversed(iP__[:-1]):  # lower compared row, follow uplinks, no uplinks in last row
        P_ = []
        for P in iP_:
            link_, link_m, link_d = [],[],[]  # for new P
            derT,valT,rdnT = [[],[]],[[],[]],[[],[]]
            # trace dlinks:
            for iderP in P.link_t[1]:
                if iderP._P.link_t[1]:  # else no _P links and derT to compare
                    _P = iderP._P
                    comp_P(_P,P, link_,link_m,link_d, derT,valT,rdnT, fd=1, derP=iderP)
            if np.sum(valT[1]) > P_aves[1] * np.sum(rdnT[1]):
                # add new P in der+ PP:
                DerT = deepcopy(P.derT); ValT = deepcopy(P.valT); RdnT = deepcopy(P.rdnT)
                for i in 0,1:
                    DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]  # append layer

                P_ += [CP(ptuple=deepcopy(P.ptuple), dert_=copy(P.dert_), box=copy(P.box),
                          derT=DerT, valT=ValT, rdnT=RdnT, link_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__


def nest(P, ddepth=3):  # default ddepth is nest 3 times: tuple->fork->layer->H
    # agg+ adds depth: number brackets before the tested bracket: P.valT[0], P.valT[0][0], etc?

    if not isinstance(P.valT[0],list):
        curr_depth = 0
        while curr_depth < ddepth:
            P.derT[0]=[P.derT[0]]; P.valT[0]=[P.valT[0]]; P.rdnT[0]=[P.rdnT[0]]
            P.derT[1]=[P.derT[1]]; P.valT[1]=[P.valT[1]]; P.rdnT[1]=[P.rdnT[1]]
            curr_depth += 1

        for derP in P.link_t[1]:
            curr_depth = 0
            while curr_depth < ddepth:
                derP.derT[0]=[derP.derT[0]]; derP.valT[0]=[derP.valT[0]]; derP.rdnT[0]=[derP.rdnT[0]]
                derP.derT[1]=[derP.derT[1]]; derP.valT[1]=[derP.valT[1]]; derP.rdnT[1]=[derP.rdnT[1]]
                curr_depth += 1

def rotate_P(der__t, mask__, ave_a, pivot):

    y,x,dert = pivot
    P = CP()
    if dert and not ave_a:  # rotate to central dert G angle
        sin,cos = dert[3]/dert[9], dert[4]/dert[9]  # dy,dx / G
    else:  # rotate P to
        if ave_a is None: sin,cos = np.divide(P.ptuple[3], P.ptuple[5])
        else:             sin,cos = np.divide(ave_a, np.hypot(*ave_a))
        if cos < 0: sin,cos = -sin,-cos
        # dx always >= 0, dy can be < 0
    new_axis = sin, cos
    rdert_, dert_ext_ = [],[]
    # scan left, inclide pivot y,x:
    ry=y; rx=x
    while True:  # terminating condition is in form_rdert()
        rdert = form_rdert(rx,ry, der__t, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ = [rdert] + rdert_  # append left
        rx-=cos; ry-=sin  # next rx,ry
        dert_ext_.insert(0,[[P],ry,rx])  # append left external params: roots and coords per dert
    x0=rx; yleft=ry
    # scan right:
    rx=x+cos; ry=y+sin  # center dert was included in scan left
    while True:
        rdert = form_rdert(rx,ry, der__t, mask__)
        if rdert is None: break  # dert is not in blob: masked or out of bound
        rdert_ += [rdert]  # append right
        rx+=cos; ry+=sin  # next rx,ry
        dert_ext_ += [[[P],ry,rx]]
    # form rP:
    rdert = rdert_[0]  # initialization:
    G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        dert_ += [rdert]
    # re-form gradients:
    G = np.hypot(Dy,Dx);  Ga = (Cos_da0 + 1) + (Cos_da1 + 1); L = len(rdert_)
    P.ptuple = [I, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], G, Ga, L]
    P.dert_ = dert_
    P.dert_ext_ = dert_ext_
    P.y = yleft + ry*(L//2); P.x = x0 + rx*(L//2)  # central coords, P may go up-right or down-right
    P.axis = new_axis

    return P

def form_rdert(rx,ry, der__t, imask__):

    # coord, distance of four int-coord derts, overlaid by float-coord rdert in der__t, int for indexing
    x0 = int(np.floor(rx)); dx0 = abs(rx - x0)
    x1 = int(np.ceil(rx));  dx1 = abs(rx - x1)
    y0 = int(np.floor(ry)); dy0 = abs(ry - y0)
    y1 = int(np.ceil(ry));  dy1 = abs(ry - y1)

    mask__= []  # imask__ padded with rim of 1s:
    mask__ += [[mask_+[True] for mask_ in imask__]]
    mask__ += [True for mask in imask__[0]]
    if not mask__[y0][x0] or not mask__[y1][x0] or not mask__[y0][x1] or not mask__[y1][x1]:
        # scale all dert params in proportion to inverted distance from rdert, sum(distances) = 1
        # approximation, square of rpixel is rotated, won't fully match not-rotated derts
        k0 = 2 - dx0*dx0 - dy0*dy0
        k1 = 2 - dx0*dx0 - dy1*dy1
        k2 = 2 - dx1*dx1 - dy0*dy0
        k3 = 2 - dx1*dx1 - dy1*dy1
        K = k0 + k1 + k2 + k3
        ptuple = tuple(
            (
                par__[y0, x0] * k0 +
                par__[y1, x0] * k1 +
                par__[y0, x1] * k2 +
                par__[y1, x1] * k3
            ) / K
            for par__ in der__t[1:])    # exclude i
        return ptuple

    else: return None


def op_parT(_parT, parT, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    for i in 0, 1:
        if fcomp:
            dparT = comp_unpack(_parT, parT, rn)
        else:
            sum_unpack(_parT, parT)
        pass
        # use sum_unpack here?
        _parH, parH = _parT[i], parT[i]
        for _aggH, aggH in _parH, parH:
            daggH = []
            for _subH, subH in _aggH, aggH:
                dsubH = []
                for Que, que in _subH, subH:
                    if fcomp:
                        parT, valT, rdnT = comp_unpack(Ele, ele, rn)
                        for i, parH in enumerate(0, 1):
                            dparT[i] += [parT[1]]
                    else:
                        pass
                        # use sum_unpack here?
                daggH += [dsubH]
            dparH[i] += [daggH]
        """  
        elev, _idx, d_didx, last_i, last_idx = 0,0,0,-1,-1
        for _i, _didx in enumerate(_parH.Q):  # i: index in Qd (select param set), idx: index in ptypes (full param set)
            _idx += _didx; idx = last_idx+1; _fd = _parH.fds[elev]; _val = _parH.Qd[_i].valt[_fd]
            for i, didx in enumerate(parH.Q[last_i+1:]):  # start after last matching i and idx
                idx += didx; fd = _parH.fds[elev]; val = parH.Qd[_i+i].valt[fd]
                if _idx==idx:
                    if _fd==fd:
                        _sub = _parH.Qd[_i]; sub = parH.Qd[_i+i]
                        if fcomp:
                            if _val > G_aves[fd] and val > G_aves[fd]:
                                if sub.n:  # sub is ptuple
                                    dsub = op_ptuple(_sub, sub, fcomp, fd, fneg)  # sub is vertuple | ptuple | ext
                                else:  # sub is pH
                                    dsub = op_parH(_sub, sub, 0, fcomp)  # keep unpacking aggH | subH | derH
                                    if sub.ext[1]: comp_ext(_sub.ext[1],sub.ext[1], dsub)
                                dparH.valt[0]+=dsub.valt[0]; dparH.valt[1]+=dsub.valt[1]  # add rdnt?
                                dparH.Qd += [dsub]; dparH.Q += [_didx+d_didx]
                                dparH.fds += [fd]
                        else:  # no eval: no new dparH
                            if sub.n: op_ptuple(_sub, sub, fcomp, fd, fneg)  # sub is vertuple | ptuple | ext
                            else:
                                op_parH(_sub, sub, fcomp)  # keep unpacking aggH | subH | derH
                                if sub.ext[1]: sum_ext(_sub.ext, sub.ext)
                    last_i=i; last_idx=idx  # last matching i,idx
                    break
                elif fcomp:
                    if _idx < idx: d_didx+=didx  # += missing didx
                else:
                    _parH.Q.insert[idx, didx+d_didx]
                    _parH.Q[idx+1] -= didx+d_didx  # reduce next didx
                    _parH.Qd.insert[idx, deepcopy(parH.Qd[idx])]
                    d_didx = 0
                if _idx < idx: break  # no par search beyond current index
                # else _idx > idx: keep searching
                idx += 1  # 1 sub/loop
            _idx += 1
            if elev in (0,1) or not _i%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
                elev+=1  # elevation

        """
    if fcomp:
        return dparH
    else:
        _parH.valt[0] += parH.valt[0];
        _parH.valt[1] += parH.valt[1]
        _parH.rdnt[0] += parH.rdnt[0];
        _parH.rdnt[1] += parH.rdnt[1]

def comp_G_(G_, pri_G_=None, f1Q=1, fd = 0, fsub=0):  # cross-comp Graphs if f1Q, else comp G_s in comp_node_

    if not f1Q: dpH_=[]  # not sure needed

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph, initially converted PPs
        # follow links in der+, loop all Gs in rng+:
        for iG in _iG.link_ if fd \
            else G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng+, bilateral link assign, val accum:
            # no new G per sub+, just compare corresponding layers?
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_ for node in link.node_]:  # if f1Q? add frng to skip?
                continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((sum(_iG.pH.valt) + sum(iG.pH.valt)) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    # pass parT, valT, rdnT?
                    dparT,valT,rdnT = comp_unpack(_G.parT, G.parT, rn=1)  # comp layers while lower match?
                    dpH.ext[1] = [1,distance,[dy,dx]]  # pack in ds
                    mval, dval = dpH.valt
                    derG = Cgraph(valt=[mval,dval], G=[_G,G], pH=dpH, box=[])  # box is redundant to G
                    # add links:
                    _G.link_ += [derG]; G.link_ += [derG]  # no didx, no ext_valt accum?
                    if mval > ave_Gm:
                        _G.link_t[0] += [derG]; _G.link_.valt[0] += mval
                        G.link_t[0] += [derG]; G.link_.valt[0] += mval
                    if dval > ave_Gd:
                        _G.link_t[1] += [derG]; _G.link_.valt[1] += dval
                        G.link_t[1] += [derG]; G.link_.valt[1] += dval

                    if not f1Q: dpH_+= dpH  # comp G_s
                # implicit cis, alt pair nesting in mderH, dderH
    if not f1Q:
        return dpH_  # else no return, packed in links

    '''
    comp alts,val,rdn? cluster per var set if recurring across root: type eval if root M|D?
    '''

def sum_dert(Dert,Valt,Rdnt, derP):

    if isinstance(derP, list):  # derP is actually P ders
        dert,valt,rdnt = derP
    else:
        dert,valt,rdnt = derP.derT, derP.valT, derP.rdnT
    for i in 0,1:
        for Ptuple, ptuple in zip(Dert[i], dert[i]):  # may be single ptuple per layer, nest anyway?
            sum_ptuple(Ptuple, ptuple)
            Valt[i] += valt[i]  # scalar per layer
            Rdnt[i] += rdnt[i]

    for i, (Ptuple, ptuple) in enumerate(zip(Dert, dert)):  # single ptuple per layer?
        sum_ptuple(Ptuple, ptuple)
        Valt[i] += valt[i]  # scalar per layer
        Rdnt[i] += rdnt[i]

    # if fd: Rdnt = [[[rdn+base_rdn for rdn in rdnL] for rdnL in link.rdnH] for rdnH in link.rdnT]
    # else:  Rdnt = [rdn+base_rdn for rdn in link.rdnT]  # single mtuple rdn, dptuple rdn

def feedback(root, fd):  # append new der layers to root

    Fback = root.fback_.pop()  # init with 1st fback: [derT,valT,rdnT]
    while root.fback_:
        sum_unpack(Fback, root.fback_.pop())  # Fback += fback, both = [derT,valT,rdnT]
    sum_unpack([root.derT,root.valT,root.rdnT], Fback)  # root += Fback, fixed nesting?

    for Layer,layer in zip(Fback,fback):  # combined layer is scalars: [n_msubPP,n_dsubPP, mval,dval, mrdn,drdn]
        for i, (Scal,scal) in enumerate(zip(Layer,layer)):
            Layer[i]+=layer[i]

    if isinstance(root.roott[fd], CPP):  # not blob
        root = root.roott[fd]
        root.fback_ += [Fback]
        if len(root.fback_) == len(root.P__[fd]):  # all nodes term, fed back to root.fback_
            feedback(root, fd)  # derT/ rng layer in sum2PP, deeper rng layers are appended by feedback
