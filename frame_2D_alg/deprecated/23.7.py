def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.derH
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CQ(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_parH(root.H[0].H[fd].derH, sub_node.derH)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root
        else:
            break

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH in blob

    # add RVal=0, DVal=0 to return?
    term = 1
    for PP in PP_:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P_) > ave_nsub:
            term = 0
            sub_recursion(PP)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only
        '''        
        for i, sG_ in enumerate(sG_t):
            val,rdn = 0,0
            for sub_G in sG_:
                val += sum(sub_G.valt); rdn += sum(sub_G.rdnt)
            Val = Valt[fd]+val; Rdn = Rdnt[fd]+rdn
        or
        Val = Valt[fd] + sum([sum(G.valt) for G in sG_])
        Rdn = Rdnt[fd] + sum([sum(G.rdnt) for G in sG_])
        '''
# old:
def op_parT(_graph, graph, fcomp, fneg=0):  # unpack aggH( subH( derH -> ptuples

    _parT, parT = _graph.parT, graph.parT

    if fcomp:
        dparT,valT,rdnT = comp_unpack(_parT, parT, rn=1)
        return dparT,valT,rdnT
    else:
        _valT, valT = _graph.valT, graph.valT
        _rdnT, rdnT = _graph.rdnT, graph.rdnT
        for i in 0,1:
            sum_unpack([_parT[i], _valT[i], _rdnT[i]], [parT[i], valT[i],rdnT[i]])

# same as comp|sum unpack?:
def op_ptuple(_ptuple, ptuple, fcomp, fd=0, fneg=0):  # may be ptuple, vertuple, or ext

    aveG = G_aves[fd]
    if fcomp:
        dtuple=CQ(n=_ptuple.n)  # + ptuple.n / 2: average n?
        rn = _ptuple.n/ptuple.n  # normalize param as param*rn for n-invariant ratio: _param/ param*rn = (_param/_n)/(param/n)
    _idx, d_didx, last_i, last_idx = 0,0,-1,-1

    for _i, _didx in enumerate(_ptuple.Q):  # i: index in Qd: select param set, idx: index in full param set
        _idx += _didx; idx = last_idx+1
        for i, didx in enumerate(ptuple.Q[last_i+1:]):  # start after last matching i and idx
            idx += didx
            if _idx == idx:
                _par = _ptuple.Qd[_i]; par = ptuple.Qd[_i+i]
                if fcomp:  # comp ptuple
                    if ptuple.Qm: val =_par+par if fd else _ptuple.Qm[_i]+ptuple.Qm[_i+i]
                    else:         val = aveG+1  # default comp for 0der pars
                    if val > aveG:
                        if isinstance(par,list):
                            if len(par)==4: m,d = comp_aangle(_par,par)
                            else: m,d = comp_angle(_par,par)
                        else: m,d = comp_par(_par, par*rn, aves[idx], finv = not i and not ptuple.Qm)
                            # finv=0 if 0der I
                        dtuple.Qm+=[m]; dtuple.Qd+=[d]; dtuple.Q+=[d_didx+_didx]
                        dtuple.valt[0]+=m; dtuple.valt[1]+=d  # no rdnt, rdn = m>d or d>m?)
                else:  # sum ptuple
                    D, d = _ptuple.Qd[_i], ptuple.Qd[_i+i]
                    if isinstance(d, list):  # angle or aangle
                        for j, (P,p) in enumerate(zip(D,d)): D[j] = P-p if fneg else P+p
                    else: _ptuple.Qd[i] += -d if fneg else d
                    if _ptuple.Qm:
                        mpar = ptuple.Qm[_i+i]; _ptuple.Qm[i] += -mpar if fneg else mpar
                last_i=i; last_idx=idx  # last matching i,idx
                break
            elif fcomp:
                if _idx < idx: d_didx+=didx  # no dpar per _par
            else: # insert par regardless of _idx < idx:
                _ptuple.Q.insert[idx, didx+d_didx]
                _ptuple.Q[idx+1] -= didx+d_didx  # reduce next didx
                _ptuple.Qd.insert[idx, ptuple.Qd[idx]]
                if _ptuple.Qm: _ptuple.Qm.insert[idx, ptuple.Qm[idx]]
                d_didx = 0
            if _idx < idx: break  # no par search beyond current index
            # else _idx > idx: keep searching
            idx += 1
        _idx += 1
    if fcomp: return dtuple


def form_PP_t(P_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = []
    for fd in 0,1:
        qPP_ = []  # initial sequence_PP s
        for P in P_:
            if not P.root_tH[-1][fd]:  # else already packed in qPP
                qPP = [[P]]  # init PP is 2D queue of (P,val)s of all layers?
                P.root_tH[-1][fd] = qPP; val = 0
                uplink_ = P.link_tH[-1][fd]
                uuplink_ = []  # next layer of links
                while uplink_:
                    for derP in uplink_:
                        _P = derP._P; _qPP = _P.root_tH[-1][fd]
                        if _qPP:
                            if _qPP is not qPP:  # _P may be added to qPP via other downlinked P
                                val += _qPP[1]  # merge _qPP in qPP:
                                for qP in _qPP[0]:
                                    qP.root_tH[-1][fd] = qPP; qPP[0] += [qP]  # append qP_
                                qPP_.remove(_qPP)
                        else:
                            qPP[0] += [_P]  # pack bottom up
                            _P.root_tH[-1][fd] = qPP
                            val += derP.valt[fd]
                            uuplink_ += derP._P.link_tH[-1][fd]
                    uplink_ = uuplink_
                    uuplink_ = []
                qPP += [val, ave + 1]  # ini reval=ave+1, keep qPP same object for ref in P.roott
                qPP_ += [qPP]

        # prune qPPs by mediated links vals:
        rePP_ = reval_PP_(qPP_, fd)  # PP = [qPP,valt,reval]
        CPP_ = [sum2PP(qPP, base_rdn, fd) for qPP in rePP_]

        PP_t += [CPP_]  # least one PP in rePP_, which would have node_ = P_

    return PP_t  # add_alt_PPs_(graph_t)?

# draft:
def merge_PP(PP, _PP, fd, fder):

    node_=PP.node_
    for _node in _PP.node_:
        if _node not in node_:
            node_ += [_node]
            _node.root_tt[-1][fder][fd] = PP  # reassign root
    sum_derH([PP.derH, PP.valt, PP.rdnt], [_PP.derH, _PP.valt, _PP.rdnt], base_rdn=0)

    Y0,Yn,X0,Xn = PP.box; y0,yn,x0, xn = _PP.box
    PP.box = [min(X0,x0),max(Xn,xn),min(Y0,y0),max(Yn,yn)]
    # mask__, ptuple as etc.

def med_eval(last_link_, old_link_, med_valH, fd):  # recursive eval of mediated link layers, in form_graph only?

    curr_link_ = []; med_val = 0
    # compute med_valH, layer= val of links mediated by incremental number of nodes:

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += np.sum(_link.valT[fd])
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH

# currently not used:

def sum_unpack(Q,q):  # recursive unpack of two pairs of nested sequences, to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # alternating rngH( derH( rngH... nesting, down to ptuple|val|rdn
    for i, (Ele,Val,Rdn, ele,val,rdn) in enumerate(zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[])):
        if ele:
            if Ele:
                if isinstance(val,list):  # element is layer or fork
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # ptuple
                    Val_[i] += val; Rdn_[i] += rdn
                    sum_ptuple(Ele, ele)
            else:
                Que += [deepcopy(ele)]; Val_+= [deepcopy(val)]; Rdn_+= [deepcopy(rdn)]

def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]  # alternating rngH( derH( rngH.. nesting,-> ptuple|val|rdn

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:
                # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [int(mval<dval),int(mval>=dval)]  # to use np.sum

            for i in 0,1:  # adds nesting per recursion
                DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]

    return DerT,ValT,RdnT

def add_unpack(H, incr):  # recursive unpack hierarchy of unknown nesting to add input
    # new_H = []
    for i, e in enumerate(H):
        if isinstance(e,list):
            add_unpack(e,incr)
        else: H[i] += incr
    return H

def last_add(H, i):  # recursive unpack hierarchy of unknown nesting to add input
    while isinstance(H,list):
        H=H[-1]
    H+=i

def unpack(H):  # recursive unpack hierarchy of unknown nesting
    while isinstance(H,list):
        last_H = H
        H=H[-1]
    return last_H

def nest(P, ddepth=2):  # default ddepth is nest 2 times: tuple->layer->H, rngH is ptuple, derH is 1,2,4.. ptuples'layers?

    # fback adds alt fork per layer, may be empty?
    # agg+ adds depth: number brackets before the tested bracket: P.valT[0], P.valT[0][0], etc?

    if not isinstance(P.valT[0],list):
        curr_depth = 0
        while curr_depth < ddepth:
            P.derT[0]=[P.derT[0]]; P.valT[0]=[P.valT[0]]; P.rdnT[0]=[P.rdnT[0]]
            P.derT[1]=[P.derT[1]]; P.valT[1]=[P.valT[1]]; P.rdnT[1]=[P.rdnT[1]]
            curr_depth += 1

        if isinstance(P, CP):
            for derP in P.link_t[1]:
                curr_depth = 0
                while curr_depth < ddepth:
                    derP.derT[0]=[derP.derT[0]]; derP.valT[0]=[derP.valT[0]]; derP.rdnT[0]=[derP.rdnT[0]]
                    derP.derT[1]=[derP.derT[1]]; derP.valT[1]=[derP.valT[1]]; derP.rdnT[1]=[derP.rdnT[1]]
                    curr_depth += 1

def form_graph_(G_, fder):  # form list graphs and their aggHs, G is node in GG graph

    mnode_, dnode_ = [],[]  # Gs with >0 +ve fork links:

    for G in G_:
        if G.link_tH[0]: mnode_ += [G]  # all nodes with +ve links, not clustered in graphs yet
        if G.link_tH[1]: dnode_ += [G]
    graph_t = []
    for fd, node_ in enumerate([mnode_, dnode_]):
        graph_ = []  # init graphs by link val:
        while node_:  # all Gs not removed in add_node_layer
            G = node_.pop(); gnode_ = [G]
            val = init_graph(gnode_, node_, G, fd, val=0)  # recursive depth-first gnode_ += [_G]
            graph_ += [[gnode_,val]]
        # prune graphs by node val:
        regraph_ = graph_reval_(graph_, [G_aves[fd] for graph in graph_], fd)  # init reval_ to start
        if regraph_:
            graph_[:] = sum2graph_(regraph_, fd)  # sum proto-graph node_ params in graph
        graph_t += [graph_]

    # add_alt_graph_(graph_t)  # overlap+contour, cluster by common lender (cis graph), combined comp?
    return graph_t

def sum_derHt(T, t, base_rdn):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T
    derH, valt, rdnt = t
    for i in 0, 1:
        Valt[i] += valt[i]; Rdnt[i] += rdnt[i] + base_rdn
    if DerH:
        for Layer, layer in zip_longest(DerH,derH, fillvalue=[]):
            if layer:
                if Layer:
                    if layer[0]: sum_derH(Layer, layer)
                else:
                    DerH += [deepcopy(layer)]
    else:
        DerH[:] = deepcopy(derH)

def frame_blobs_root(image, intra=False, render=False, verbose=False):

    if verbose: start_time = time()
    Y, X = image.shape[:2]

    dir__t = comp_axis(image)  # nested tuple of 2D arrays: [i,[4|8 ds]]: single difference per axis
    i__, g__t = dir__t
    # combine ds: (diagonal is projected to orthogonal, cos(45) = sin(45) = 0.5**0.5)
    dy__ = (g__t[3]-g__t[2])*(0.5**0.5) + g__t[0]
    dx__ = (g__t[2]-g__t[3])*(0.5**0.5) + g__t[1]
    g__ = np.hypot(dy__, dx__)  # gradient magnitude
    der__t = i__, dy__, dx__, g__

    # compute signs
    g_sqr__t = g__t*g__t
    val__ = np.sqrt(
        # value is ratio between edge ones and the rest:
        # https://www.rastergrid.com/blog/2011/01/frei-chen-edge-detector/#:~:text=When%20we%20are%20using%20the%20Frei%2DChen%20masks%20for%20edge%20detection%20we%20are%20searching%20for%20the%20cosine%20defined%20above%20and%20we%20use%20the%20first%20four%20masks%20as%20the%20elements%20of%20importance%20so%20the%20first%20sum%20above%20goes%20from%20one%20to%20four.
        (g_sqr__t[0] + g_sqr__t[1] + g_sqr__t[2] + g_sqr__t[3]) /
        (g_sqr__t[0] + g_sqr__t[1] + g_sqr__t[2] + g_sqr__t[3] + g_sqr__t[4] + g_sqr__t[5] + g_sqr__t[6] + g_sqr__t[7] + g_sqr__t[8])
    )
    dsign__ = ave - val__ > 0   # max d per kernel
    gsign__ = ave - g__   > 0   # below-average g
    # https://en.wikipedia.org/wiki/Flood_fill
    # edge_, idmap, adj_pairs = flood_fill(dir__t, dsign__, prior_forks='', verbose=verbose, cls=CEdge)
    # assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    # I, Ddl, Dd, Ddr, Dr = 0, 0, 0, 0, 0
    # for edge in edge_: I += edge.I; Ddl += edge.Ddl; Dd += edge.Dd; Ddr += edge.Ddr; Dr += edge.Dr
    # frameE = CEdge(I=I, Ddl=Ddl, Dd=Dd, Ddr=Ddr, Dr=Dr, dir__t=dir__t, node_tt=[[[], blob_], [[], []]], box=(0, Y, 0, X))

    blob_, idmap, adj_pairs = flood_fill(der__t, gsign__, prior_forks='', verbose=verbose)  # overlap or for negative edge blobs only?
    assign_adjacents(adj_pairs)

def comp_axis(image):

    pi__ = np.pad(image, pad_width=1, mode='edge')      # pad image with edge values

    g___ = np.zeros((9,) + image.shape, dtype=float)    # g is gradient per axis

    # take 3x3 kernel slices of pixels:
    tl = pi__[ks.tl]; tc = pi__[ks.tc]; tr = pi__[ks.tr]
    ml = pi__[ks.ml]; mc = pi__[ks.mc]; mr = pi__[ks.mr]
    bl = pi__[ks.bl]; bc = pi__[ks.bc]; br = pi__[ks.br]

    # apply Frei-chen filter to image:
    # https://www.rastergrid.com/blog/2011/01/frei-chen-edge-detector/
    # First 4 values are edges:
    g___[0] = (tl+tr-bl-br)/DIAG_DIST + (tc-bc)/ORTHO_DIST
    g___[1] = (tl+bl-tr-br)/DIAG_DIST + (ml-mr)/ORTHO_DIST
    g___[2] = (ml+bc-tc-mr)/DIAG_DIST + (tr-bl)/ORTHO_DIST
    g___[3] = (mr+bc-tc-ml)/DIAG_DIST + (tr-bl)/ORTHO_DIST
    # The next 4 are lines
    g___[4] = (tc+bc-ml-mr)/ORTHO_DIST
    g___[5] = (tr+bl-tl-br)/ORTHO_DIST
    g___[6] = (mc*4-(tc+bc+ml+mr)*2+(tl+tr+bl+br))/6
    g___[7] = (mc*4-(tl+br+tr+bl)*2+(tc+bc+ml+mr))/6
    # The last one is average
    g___[8] = (tl+tc+tr+ml+mc+mr+bl+bc+br)/9

    return (pi__[ks.mc], g___)


def intra_blob_root(root_blob, render, verbose, fBa):  # recursive evaluation of cross-comp slice| range| angle per blob

    # deep_blobs = []  # for visualization
    spliced_layers = []
    if fBa:
        blob_ = root_blob.dlayers[0]
    else:
        blob_ = root_blob.rlayers[0]

    for blob in blob_:  # fork-specific blobs, print('Processing blob number ' + str(bcount))
        # increment forking sequence: g -> r|a, a -> v
        extend_der__t(blob)  # der__t += 1: cross-comp in larger kernels or possible rotation
        blob.root_der__t = root_blob.der__t
        blob_height = blob.box[1] - blob.box[0];
        blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions: Ly, Lx
            if root_blob.fBa:  # vectorize fork in angle blobs
                if (blob.G - aveA * (blob.rdn + 2)) + (aveA * (blob.rdn + 2) - blob.Ga) > 0 and blob.sign:  # G * angle match, x2 costs
                    blob.fBa = 0;
                    blob.rdn = root_blob.rdn + 1
                    blob.prior_forks += 'v'
                    if verbose: print('fork: v')  # if render and blob.A < 100: deep_blobs += [blob]
                    vectorize_root(blob, verbose=verbose)
            else:
                if blob.G < aveR * blob.rdn and blob.sign:  # below-average G, eval for comp_r
                    blob.fBa = 0;
                    blob.rng = root_blob.rng + 1;
                    blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                    # comp_r 4x4:
                    new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                    sign__ = ave * (blob.rdn + 1) - new_der__t[3] > 0  # m__ = ave - g__
                    # if min Ly and Lx, der__t>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = \
                            cluster_fork_recursive(blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=0)
                # || forks:
                if blob.G > aveA * blob.rdn and not blob.sign:  # above-average G, eval for comp_a
                    blob.fBa = 1; blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                    # comp_a 2x2:
                    new_der__t, new_mask__ = comp_a(blob.der__t, blob.mask__)
                    sign__ = ave_a - new_der__t.ga > 0
                    # vectorize if dev_gr + inv_dev_ga, if min Ly and Lx, der__t>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = \
                            cluster_fork_recursive(blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=1)
            '''
            this is comp_r || comp_a, gap or overlap version:
            if aveBa < 1: blobs of ~average G are processed by both forks
            if aveBa > 1: blobs of ~average G are not processed

            else exclusive forks:
            vG = blob.G - ave_G  # deviation of gradient, from ave per blob, combined max rdn = blob.rdn+1:
            vvG = abs(vG) - ave_vG * blob.rdn  # 2nd deviation of gradient, from fixed costs of if "new_der__t" loop below
            # vvG = 0 maps to max G for comp_r if vG < 0, and to min G for comp_a if vG > 0:

            if blob.sign:  # sign of pixel-level g, which corresponds to sign of blob vG, so we don't need the later
                if vvG > 0:  # below-average G, eval for comp_r...
                elif vvG > 0:  # above-average G, eval for comp_a...
            '''
    # if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)

    return spliced_layers

"""
Cross-comparison of pixels or gradient angles in 2x2 kernels
"""

import numpy as np
from collections import namedtuple

from frame_blobs import idert
from utils import kernel_slice_3x3 as ks
# no ave_ga = .78, ave_ma = 2  # at 22.5 degrees
# https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png

def comp_a(dert__, mask__=None):  # cross-comp of gradient angle in 3x3 kernels

    if mask__ is not None:
        majority_mask__ = np.sum(
            (
                mask__[ks.tl], mask__[ks.tc], mask__[ks.tr],
                mask__[ks.ml], mask__[ks.mc], mask__[ks.mr],
                mask__[ks.bl], mask__[ks.bc], mask__[ks.br],
            ),
            axis=0) > 2.25  # 1/4 of maximum values?
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__= dert__[:4]  # day__,dax__,ma__ are recomputed

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        uv__ = dert__[1:3] / g__
        uv__[np.where(np.isnan(uv__))] = 0  # set nan to 0, to avoid error later

    # uv__ comparison in 3x3 kernels:
    lcol = angle_diff(uv__[ks.br], uv__[ks.tr])  # left col
    ccol = angle_diff(uv__[ks.bc], uv__[ks.tc])  # central col
    rcol = angle_diff(uv__[ks.bl], uv__[ks.tl])  # right col
    trow = angle_diff(uv__[ks.tr], uv__[ks.tl])  # top row
    mrow = angle_diff(uv__[ks.mr], uv__[ks.ml])  # middle row
    brow = angle_diff(uv__[ks.br], uv__[ks.bl])  # bottom row

    # compute mean vectors
    mday__ = 0.25*lcol + 0.5*ccol + 0.25*rcol
    mdax__ = 0.25*trow + 0.5*mrow + 0.25*brow

    # normalize mean vectors into unit vectors
    dyy__, dyx__ = mday__ / np.hypot(*mday__)
    dxy__, dxx__ = mdax__ / np.hypot(*mdax__)

    # v component of mean unit vector represents similarity of angles
    # between compared vectors, goes from -1 (opposite) to 1 (same)
    ga__ = np.hypot(1-dyx__, 1-dxx__)     # +1 for all positives
    # or ga__ = np.hypot(np.pi + np.arctan2(dyy__, dyx__), np.pi + np.arctan2(dxy__, dxx__)?

    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    in conventional notation: G = (Ix, Iy), A = (Ix, Iy) / hypot(G), DA = (dAdx, dAdy), abs_GA = hypot(DA)?
    '''
    i__ = i__[ks.mc]
    dy__ = dy__[ks.mc]
    dx__ = dx__[ks.mc]
    g__ = g__[ks.mc]

    return adert(g__, ga__, i__, dy__, dx__, dyy__, dyx__, dxy__, dxx__), majority_mask__

def angle_diff(uv2, uv1):  # compare angles of uv1 to uv2 (uv1 to uv2)

    u2, v2 = uv2[:]
    u1, v1 = uv1[:]

    # sine and cosine of difference between angles of uv1 and uv2:
    u3 = (v1 * u2) - (u1 * v2)
    v3 = (v1 * v2) + (u1 * u2)

    return np.stack((u3, v3))

'''
alternative versions below:
'''
def comp_r_odd(dert__, ave, rng, root_fia, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel,
    ...
    Sobel coefficients to decompose ds into dy and dx:
    YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
    XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
        |--(clockwise)--+  |--(clockwise)--+
        YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
                0       0  ¦          -2       2  ¦
                1   2   1  ¦          -1   0   1  ¦
    Scharr coefs:
    YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
    XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, using Sobel coeffs, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_d.drawio
    '''

    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    rotate in first call only: same orientation as from frame_blobs?
    '''
    i__center = i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
    i__topleft = i__[:-2:2, :-2:2]
    i__top = i__[:-2:2, 1:-1:2]
    i__topright = i__[:-2:2, 2::2]
    i__right = i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom = i__[2::2, 1:-1:2]
    i__bottomleft = i__[2::2, :-2:2]
    i__left = i__[1:-1:2, :-2:2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[1:-1:2, 1:-1:2].astype(int)
                          + mask__[:-2:2, :-2:2].astype(int)
                          + mask__[:-2:2, 1:-1: 2].astype(int)
                          + mask__[:-2:2, 2::2].astype(int)
                          + mask__[1:-1:2, 2::2].astype(int)
                          + mask__[2::2, 2::2].astype(int)
                          + mask__[2::2, 1:-1:2].astype(int)
                          + mask__[2::2, :-2:2].astype(int)
                          + mask__[1:-1:2, :-2:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function
    '''
    can't happen:
    if root_fia:  # initialize derivatives:  
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)
    else: 
    '''
     # root fork is comp_r, accumulate derivatives:
    dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
    dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
    m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs * rim skip ratio:

    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 9x9, *4 for 17x17

    dy__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * -2  * rngSkip +
             (i__topright - i__bottomleft) * -1 * rngSkip +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 * rngSkip+
             (i__right - i__left) * 2 * rngSkip)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    '''
    m__ += ( abs(i__center - i__topleft) * 1 * rngSkip
           + abs(i__center - i__top) * 2 * rngSkip
           + abs(i__center - i__topright) * 1 * rngSkip
           + abs(i__center - i__right) * 2 * rngSkip
           + abs(i__center - i__bottomright) * 1 * rngSkip
           + abs(i__center - i__bottom) * 2 * rngSkip
           + abs(i__center - i__bottomleft) * 1 * rngSkip
           + abs(i__center - i__left) * 2 * rngSkip
           )

    return idert(i__center, dy__, dx__, g__, m__), majority_mask__