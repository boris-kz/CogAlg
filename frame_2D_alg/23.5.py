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