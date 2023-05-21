def intra_blob_root(root_blob, render, verbose, fBa):  # recursive evaluation of cross-comp slice| range| angle per blob

    # deep_blobs = []  # for visualization
    spliced_layers = []
    if fBa:
        blob_ = root_blob.dlayers[0]
    else:
        blob_ = root_blob.rlayers[0]

    for blob in blob_:  # fork-specific blobs, print('Processing blob number ' + str(bcount))

        blob.prior_forks = root_blob.prior_forks  # increment forking sequence: g -> r|a, a -> p
        blob.root_dert__ = root_blob.dert__
        blob_height = blob.box[1] - blob.box[0];
        blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions: Ly, Lx
            if root_blob.fBa:
                # comp_slice fork in angle blobs
                # add evaluate splice_blobs(root_blob), normally in frame_bblobs?
                AveB = aveB * (blob.rdn + 1)  # comp_slice is doubling the costs, likely higher, adjust per nsub_blobs?
                if blob.G * (1 / blob.Ga) > AveB * pcoef:  # value of comp_slice_blob is proportional to angle stability?
                    blob.fBa = 0;
                    blob.rdn = root_blob.rdn + 1
                    blob.prior_forks += 'p'
                    comp_slice_root(blob, verbose=verbose)
                    if verbose: print('\nslice_blob fork\n')  # if render and blob.A < 100: deep_blobs += [blob]
            else:
                ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
                ''' 
                comp_r || comp_a, gap or overlap version:
                if aveBa < 1: blobs of ~average G are processed by both forks
                if aveBa > 1: blobs of ~average G are not processed:
                '''
                if blob.G < aveB * blob.rdn:  # below-average G, eval for comp_r
                    # root values for sub_blobs:
                    blob.fBa = 0;
                    blob.rng = root_blob.rng + 1;
                    blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn
                    # comp_r 4x4:
                    new_dert__, new_mask__ = comp_r(ext_dert__, blob.rng, ext_mask__)
                    sign__ = ave * (blob.rdn + 1) - new_dert__[3] > 0  # m__ = ave - g__
                    blob.prior_forks += 'r'
                    # if min Ly and Lx, dert__>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = cluster_fork_recursive(blob, spliced_layers, new_dert__, sign__,
                                                                   new_mask__, verbose, render, fBa=0)

                if blob.G > aveB * aveBa * blob.rdn:  # above-average G, eval for comp_a
                    # root values for sub_blobs:
                    blob.fBa = 1;
                    blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn
                    # comp_a 2x2:
                    new_dert__, new_mask__ = comp_a(ext_dert__, ext_mask__)
                    Ave = ave * (blob.rdn + 1)
                    sign__ = (new_dert__[1] - Ave) + (Ave * pcoef - new_dert__[2]) > 0  # val_comp_slice_= dev_gr + inv_dev_ga
                    blob.prior_forks += 'a'
                    # if min Ly and Lx, dert__>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = cluster_fork_recursive(blob, spliced_layers, new_dert__, sign__,
                                                                   new_mask__, verbose, render, fBa=1)
            '''
            exclusive forks version:

            vG = blob.G - ave_G  # deviation of gradient, from ave per blob, combined max rdn = blob.rdn+1:
            vvG = abs(vG) - ave_vG * blob.rdn  # 2nd deviation of gradient, from fixed costs of if "new_dert__" loop below
            # vvG = 0 maps to max G for comp_r if vG < 0, and to min G for comp_a if vG > 0:

            if blob.sign:  # sign of pixel-level g, which corresponds to sign of blob vG, so we don't need the later
                if vvG > 0:  # below-average G, eval for comp_r...
                elif vvG > 0:  # above-average G, eval for comp_a...
            '''
    # if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)

    return spliced_layers


def cluster_fork_recursive(blob, spliced_layers, new_dert__, sign__, new_mask__, verbose, render, fBa):
    if verbose:
        print('fork:', blob.prior_forks)
    # form sub_blobs:
    sub_blobs, idmap, adj_pairs = flood_fill(new_dert__, sign__,
                                             verbose=False, mask__=new_mask__)
    '''
    adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
    sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
    + if sub_blob_val > alt_blob_val, else -?  
    '''
    adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
    blob.rdn += adj_rdn
    for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
    assign_adjacents(adj_pairs)
    # if render: visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (froot_Ba = {blob.fBa}, froot_Ba = {blob.prior_forks[-1] == 'a'})")

    if fBa:
        sublayers = blob.dlayers
    else:
        sublayers = blob.rlayers

    sublayers += [sub_blobs]  # r|a fork- specific sub_blobs, then add deeper layers of mixed-fork sub_blobs:
    sublayers += intra_blob_root(blob, render, verbose, fBa)  # recursive eval cross-comp range| angle| slice per blob

    new_spliced_layers = [spliced_layer + sublayer for spliced_layer, sublayer in
                          zip_longest(spliced_layers, sublayers, fillvalue=[])]
    return new_spliced_layers


def sub_recursion_eval(root):  # for PP or dir_blob

    root_PPm_, root_PPd_ = root.rlayers[-1], root.dlayers[-1]
    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:  # fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple|vertuple?
            if fd: comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:  comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]
            # fork val, rdn:
            val = PP.valt[fd]; alt_val = sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
            ave = PP_aves[fd] * (PP.rdnt[fd] + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # rdn incr, splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer
            # sub_PPs / sub+?
            # include empty comb_layers: # revise?
            if fd:
                PPmm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPmm_ for PP in PP_])
                PPmd_ = [PPm_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPmd_ for PP in PP_])
                root.dlayers = [PPmd_,PPmm_]
            else:
                PPdm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPdm_ for PP in PP_])
                PPdd_ = [PPd_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPdd_ for PP in PP_])
                root.rlayers = [PPdm_, PPdd_]
            # root is PP:
            if isinstance(root, CPP):
                for i in 0,1:
                    root.valt[i] += PP.valt[i]  # vals
                    root.rdnt[i] += PP.rdnt[i]  # ad rdn too?
            else:  # root is Blob
                if fd: root.G += sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
                else:  root.M += PP.valt[fd]


def sub_recursion(PP):  # evaluate PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert bottom-up to top-down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdnt[PP.fds[-1] ] += 1  # two-fork rdn, priority is not known?  rotate?

    cP__ = [copy(P_) for P_ in P__]
    sub_PPm_, sub_PPd_ = form_PP_t(cP__, base_rdn=PP.rdnt[PP.fds[-1]], fds=PP.fds)
    PP.rlayers[:] = [sub_PPm_]
    PP.dlayers[:] = [sub_PPd_]
    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs

def sum2PP(qPP, base_rdn, fd):  # sum PP_segs into PP

    P__, val, _ = qPP
    # init:
    P = P__[0][0]
    if P.link_t[fd]:
        derP = P.link_t[fd][0]
        derH = [[deepcopy(derP.derH[0][0]), copy(derP.derH[0][1]), fd]]
        if len(P.link_t[fd]) > 1: sum_links(derH, P.link_t[fd][1:], fd)
    else: derH = []
    PP = CPP(derH=[[[P.ptuple], [[P]],fd]], box=[P.y0, P__[-1][0].y0, P.x0, P.x0 + len(P.dert_)], rdn=base_rdn, link__=[[copy(P.link_t[fd])]])
    PP.valt[fd] = val
    PP.rdnt[fd] += base_rdn
    PP.nlink__ += [[nlink for nlink in P.link_ if nlink not in P.link_t[fd]]]
    # accum:
    for i, P_ in enumerate(P__):  # top-down
        P_ = []
        for j, P in enumerate(P_):
            P.roott[fd] = PP
            if i or j:  # not init
                sum_ptuple_(PP.derH[0][0], P.ptuple if isinstance(P.ptuple, list) else [P.ptuple])
                P_ += [P]
                if derH: sum_links(PP, derH, P.link_t[fd], fd)  # sum links into new layer
                PP.link__ += [[P.link_t[fd]]]  # pack top down
                # the links may overlap between Ps of the same row?
                PP.nlink__ += [[nlink for nlink in P.link_ if nlink not in P.link_t[fd]]]
                PP.box[0] = min(PP.box[0], P.y0)  # y0
                PP.box[2] = min(PP.box[2], P.x0)  # x0
                PP.box[3] = max(PP.box[3], P.x0 + len(P.dert_))  # xn
        PP.derH[0][0] += [P_]  # pack new P top down
    PP.derH += derH
    return PP

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, over the same derPs

    if isinstance(P__[0][0].ptuple, Cptuple):
        for P_ in P__:
            for P in P_:
                P.ptuple = [P.ptuple]

    for P_ in P__[1:]:  # exclude 1st row: no +ve uplinks
        for P in P_:
            for derP in P.link_t[1]:  # fd=1
                _P, P = derP._P, derP.P
                i= len(derP.derQ)-1
                j= 2*i
                if len(_P.derH)>j-1 and len(P.derH)>j-1:  # extend derP.derH:
                    comp_layer(derP, i,j)

    return P__

def comp_P_der(P__):  # form new Ps in der+ PP.P__, extend link.derH, P.derH, _P.derH for select links?

    for P_ in reversed(P__[:-1]):  # exclude 1st row: no +ve uplinks (reversed to scan it bottom up)
        for P in P_:
            PLay = []  # new layer in P.derH
            for derP in P.link_t[1]:  # fd=1
                _P = derP._P
                dL = len(_P.derH) > len(P.derH)  # dL = 0|1: _P.derH was extended by other P, compare _P.derH[-2]:
                _derLay, derLay = P.derH[-1], _P.derH[-(1+dL)]  # comp top P layer, no subLay selection till agg+
                linkLay = []
                for i, (_vertuple, vertuple, Dtuple) in enumerate(zip_longest(_derLay, derLay, PLay, fillvalue=[])):
                    dtuple = comp_vertuple(_vertuple, vertuple)  # max two vertuples in 2nd layer
                    linkLay += [dtuple]
                    if Dtuple: sum_vertuple(Dtuple, dtuple)
                    else: PLay += [dtuple]  # init Dtuple
                    if dL: sum_vertuple(_P.derH[-1][i], dtuple)  # bilateral sum
                if not dL: _P.derH += [linkLay]  # init Lay
                derP.derH += [linkLay]
                # or selective by linkLay valt, in the whole link_t?
            P.derH += [PLay]
    return P__


def comp_layer(derP, i,j):  # list derH and derQ, single der+ count=elev, eval per PP

    for _ptuple, ptuple in zip(derP._P.derH[i:j], derP.P.derH[i:j]):  # P.ptuple is derH

        dtuple = comp_vertuple(_ptuple, ptuple)
        derP.derH += [dtuple]
        for k in 0, 1:
            derP.valt[k] += dtuple.valt[k]
            derP.rdnt[k] += dtuple.rdnt[k]

'''
replace rotate_P_ with directly forming axis-orthogonal Ps:
'''
def slice_blob_ortho(blob):

    P_ = []
    while blob.dert__:
        dert = blob.dert__.pop()
        P = CP(dert_= [dert])  # init cross-P per dert
        # need to find/combine adjacent _dert in the direction of gradient:
        _dert = blob.dert__.pop()
        mangle,dangle = comp_angle(dert.angle, _dert.angle)
        if mangle > ave:
            P.dert_ += [_dert]  # also sum ptuple, etc.
        else:
            P_ += [P]
            P = CP(dert_= [_dert])  # init cross-P per missing dert
            # add recursive function to find/combine adjacent _dert in the direction of gradient:
            _dert = blob.dert__.pop()
            mangle, dangle = comp_angle(dert.angle, _dert.angle)
            if mangle > ave:
                P.dert_ += [_dert]  # also sum ptuple, etc.
            else:
                pass  # add recursive slice_blob

def comp_der(iP__):  # form new Ps and links in rng+ PP.P__, extend their link.derH, P.derH, _P.derH

    P__ = []
    for iP_ in reversed(iP__[:-1]):  # lower compared row, follow uplinks, no uplinks in last row
        P_ = []
        for P in iP_:
            Mt0,Dt0, Mtuple,Dtuple = [],[],[],[]  # 1st and 2nd layers
            link_, link_m, link_d = [],[],[]  # for new P
            mVal, dVal, mRdn, dRdn = 0,0,0,0
            for iderP in P.link_t[1]:  # dlinks
                _P = iderP._P
                # comp_P through 110?
                dL = len(_P.derH) > len(P.derH)  # 0|1: extended _P.derH: not with bottom-up? comp _P.derH[-2]:
                _vert0, vert0 = _P.derH[-1][0], P.derH[-(1+dL)][0]  # can be simpler but less generic (each vert0 is [mtuple, dtuple])
                mtuple, dtuple = comp_vertuple(_vert0, vert0, len(P.link_)/(max(1, len(_P.link_))))
                mval = sum(mtuple)+iderP.valt[0]
                dval = sum(dtuple)+iderP.valt[1]  # sum from both layers
                mrdn = 1+dval>mval; drdn = 1+(not mrdn)
                derP = CderP(derH=[[[mtuple,dtuple]]], _P=_P, P=P, valt=[mval,dval], rdnt=[mrdn,drdn], fds=copy(P.fds),
                             L=len(_P.dert_), x0=min(P.x0,_P.x0), y0=min(P.y0,_P.y0))
                link_ += [derP]  # all uplinks, not bilateral
                if mval > aveB*mrdn:
                    link_m+=[derP]; mVal+=mval; mRdn+=mrdn  # +ve links, fork selection in form_PP_t
                    sum_tuple(Mtuple, vert0[0])
                    sum_tuple(Mtuple1, mtuple)
                if dval > aveB*drdn:
                    link_d+=[derP]; dVal+=dval; dRdn+=drdn
                    sum_tuple(Dtuple, vert0[1])
                    sum_tuple(Dtuple1, dtuple)
            if dVal > ave_P * dRdn:
                P_ += [CP(ptuple=deepcopy(P.ptuple), derH=[[[Mtuple,Dtuple]],[[Mtuple1,Dtuple1]]], dert_=copy(P.dert_), fds=copy(P.fds)+[0],
                          x0=P.x0, y0=P.y0, valt=[mVal,dVal], rdnt=[mRdn,dRdn], rdnlink_=link_, link_t=[link_m,link_d])]
        P__+= [P_]
    return P__

def comp_P(_P,P, Mtuple, Dtuple, link_,link_m,link_d, Valt, Rdnt, fd, derP=None, Mt0=None, Dt0=None):  #  last three if der+

    if fd:
        _dtuple, dtuple = _P.derH[-1][0][1], P.derH[-1][0][1]  # 1-vertuple derH before feedback, compare dtuple
        mtuple,dtuple = comp_dtuple(_dtuple, dtuple, rn=len(_P.link_t[1])/len(P.link_t[1]))  # comp_tuple
    else:
        mtuple,dtuple = comp_ptuple(_P.ptuple, P.ptuple)
    mval = sum(mtuple)
    if fd: mval += derP.valt[0]
    dval = sum(dtuple)
    if fd: dval += derP.valt[1]
    mrdn = 1+ dval>mval; drdn = 1+(not mrdn)

    derP = CderP(derH=[[[mtuple,dtuple]]],fds=P.fds+[fd], valt=[mval,dval],rdnt=[mrdn,drdn], P=P,_P=_P, x0=_P.x0,y0=_P.y0,L=len(_P.dert_))
    link_ += [derP]  # all links
    if mval > aveB*mrdn:
        link_m+=[derP]; Valt[0]+=mval; Rdnt[0]+=mrdn  # +ve links, fork selection in form_PP_t
        sum_tuple(Mtuple, mtuple)
        if fd: sum_tuple(Mt0, derP.derH[0][0][0])
    if dval > aveB*drdn:
        link_d+=[derP]; Valt[1]+=dval; Rdnt[1]+=drdn
        sum_tuple(Dtuple, dtuple)
        if fd: sum_tuple(Dt0, derP.derH[0][0][1])

'''
    nvalt = int  # of links to alt PPs?
    alt_rdn = int  # overlapping redundancy between core and edge
    alt_PP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    altuple = list  # summed from alt_PP_, sub comp support, agg comp suppression?
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
'''
def feedback(root, fd, fb):  # bottom-up update root.rngH, breadth-first, separate for each fork?

    DerLay, VAL, RDN = fb  # new rng|der lays, not in root.derH, summed across fb layers and nodes
    while True:
        Val = 0; Rdn = 1
        DerLay = []  # not in root.derH
        root.fterm = 1; root.derH = [root.derH]  # derH->rngH
        for PP in root.P__[fd]:
            [sum_vertuple(T,t) for T,t in zip(DerLay,PP.derH[-1])]; Val += PP.valt[fd]; Rdn += PP.rdnt[fd]
        DerH += [DerLay]
        if fd:  # der+
            root.derH[-1] += [DerLay]   # or DerH?  new der lay in last derH
        else:  # rng+
            root.derH += RngH if RngH else [DerH]  # append new rng lays or rng lay = terminated DerH?
            RngH += [DerH]  # not sure; comp in agg+ only

        root.valt[fd] += Val; root.rdnt[fd] += Rdn
        VAL += Val; RDN += Rdn
        root = root.root
        # continue while sub+ terminated in all nodes and root is not blob:
        if not isinstance(root,CPP) or not all([[node.fterm for node in root.P__[fd]]]) or VAL/RDN < G_aves[root.fds[-1]]:
            root.fb += [DerH, RngH, VAL, RDN]  # for future root termination.
            break

def comp_angle(_angle, angle):  # rn doesn't matter for angles

    _Dy, _Dx = _angle
    Dy, Dx = angle
    _G = np.hypot(_Dy,_Dx); G = np.hypot(Dy,Dx)
    sin = Dy / (.1 if G == 0 else G);     cos = Dx / (.1 if G == 0 else G)
    _sin = _Dy / (.1 if _G == 0 else _G); _cos = _Dx / (.1 if _G == 0 else _G)
    sin_da = (cos * _sin) - (sin * _cos)  # sin(α - β) = sin α cos β - cos α sin β
    cos_da = (cos * _cos) + (sin * _sin)  # cos(α - β) = cos α cos β + sin α sin β

    dangle = np.arctan2(sin_da, cos_da)  # scalar, vertical difference between angles
    mangle = ave_dangle - abs(dangle)  # inverse match, not redundant as summed across sign

    return [mangle, dangle]
