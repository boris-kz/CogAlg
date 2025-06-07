def sum2graph(root, node_,link_,llink_,Et,olp, Lay, rng, fi):  # sum node and link params into graph, aggH in agg+ or player in sub+
    '''
            N = sum_N_(node_ if fnode_ else link_, fCG = fnode_)
            m_,M = centroid_M(N.derTT[0],ave*nrc)  # weigh by match to mean m|d
            d_,D = centroid_M(N.derTT[1],ave*nrc); derTT = np.array([m_,d_])
            Et += np.array([M, D, Et[2]]) * int_w
    '''
    n0 = node_[0]
    graph = CG( fi=fi,rng=rng,olp=olp, Et=Et,et=Lay.Et, N_=node_,L_=link_, root=root,
                box=n0.box, baseT=Lay.baseT+n0.baseT, derTT=Lay.derTT+n0.derTT, derH=copy(n0.derH))
    graph.hL_ = llink_
    n0.root = graph; yx_ = [n0.yx]; fg = fi and isinstance(n0.N_[0],CG)  # not PPs
    Nt = copy_(n0)  #->CN, comb forks: add_N(Nt,Nt.Lt)?
    for N in node_[1:]:
        add_H(graph.derH,N.derH,graph); graph.baseT+=N.baseT; graph.derTT+=N.derTT; graph.box=extend_box(graph.box,N.box); yx_+=[N.yx]
        if fg: add_N(Nt, N)
    if fg: graph.H = Nt.H + [Nt]  # pack prior top level
    graph.derH += [[Lay,[]]]  # append flat
    yx = np.mean(yx_,axis=0); dy_,dx_ = (yx_-yx).T; dist_ = np.hypot(dy_,dx_)
    graph.span = dist_.mean()  # node centers distance to graph center
    graph.angle = np.sum([l.angle for l in link_],axis=0)
    graph.yx = yx
    if not fi:  # add mfork as link.node_(CL).root dfork, 1st layer, higher layers are added in cross_comp
        for L in link_:  # LL from comp_link_
            LR_ = set([n.root for n in L.N_ if isinstance(n.root,CG)])  # nodet, skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:
                    LR.Et += dfork.Et; LR.derTT += dfork.derTT  # lay0 += dfork
                    if LR.derH[-1][1]: LR.derH[-1][1].add_lay(dfork)  # direct root only
                    else:              LR.derH[-1][1] =dfork.copy_()  # was init by another node
                    if LR.lH: LR.lH[-1].N_ += [graph]  # last lev
                    else:     LR.lH += [CN(N_=[graph])]  # init
        alt_=[]  # add mGs overlapping dG
        for L in node_:
            for n in L.N_:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in alt_:  # root is not frame
                    mG.alt_ += [graph]  # cross-comp|sum complete alt before next agg+ cross-comp, multi-layered?
                    alt_ += [mG]
    return graph
'''
            n_ = node_ if fnode_ else link_
            n0 = n_[0]; derH, derTT, baseT, Et, olp = copy_(n0.derH), n0.derTT.copy, n0.baseT.copy, n0.Et.copy, n0.olp
            for n in n_[1:]:
                add_H(derH,n.derH); derTT += n.derTT; baseT += n.baseT; Et += n.Et; olp+=n.olp
'''

def comp_H(H,h, rn, root, fi):  # one-fork derH if not fi, else two-fork derH

    derH, derTT, Et = [], np.zeros((2,8)), np.zeros(3)
    for _lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if _lay and lay:
            if fi:  # two-fork lays
                dLay = []
                for _fork,fork in zip_longest(_lay,lay):
                    if _fork and fork:
                        dlay = _fork.comp_lay(fork, rn,root=root)
                        if dLay: dLay.add_lay(dlay)  # sum ds between input forks
                        else:    dLay = dlay
            else:  # one-fork lays
                 dLay = _lay.comp_lay(lay, rn, root=root)
            derTT = dLay.derTT; Et += dLay.Et
            derH += [dLay]
    return derH, derTT, Et

def comp_N(_N,N, ave, fi, angle=None, span=None, dir=1, fdeep=0, rng=1):  # compare links, relative N direction = 1|-1, no need for angle, dist?

    dderH = []
    derTT, Et, rn = base_comp(_N, N, dir)
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box)
    # spec / lay:
    if fdeep and (val_(Et, mw=len(N.derH)-2, aw=o) > 0 or N.name=='link'):  # else derH is dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fi)  # comp shared layers, if any
        # comp int proj x ext ders:
        # comp_H( proj_dH(_N.derH[1:]), dderH[:-1])  why 1: and :-1?
        # spec/ comp_node_(node_|link_)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT)), *dderH]
    for lay in dderH:
        derTT += lay.derTT; Et += lay.Et
    # spec / alt:
    if fi and _N.alt_ and N.alt_:
        et = _N.alt_.Et + N.alt_.Et  # comb val
        if val_(et, aw=2+o, fi=0) > 0:  # eval Ds
            Link.alt_L = comp_N(N2G(_N.alt_), N2G(N.alt_), ave*2, fi=1, angle=angle); Et += Link.alt_L.Et
    Link.Et = Et
    if Et[0] > ave * Et[2]:
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            node.et += Et
            node.nrim += [node]
            if fi: node.rim += [(Link,rev)]
            else: node.rim[1-rev] += [(Link,rev)]  # rimt opposite to _N,N dir
            add_H(node.extH, Link.derH, root=node, rev=rev, fi=0)
    return Link

def add_H(H, h, root, rev=0, fi=1):  # add fork L.derHs

    for Lay,lay in zip_longest(H,h):  # different len if lay-selective comp
        if lay:
            if fi:  # two-fork lays
                if Lay:
                    for Fork,fork in zip_longest(Lay,lay):
                        if fork:
                            if Fork: Fork.add_lay(fork,rev)
                            else:    Lay+= [fork.copy_(rev)]
                else:
                    Lay = []
                    for fork in lay:
                        if fork: Lay += [fork.copy_(rev)]; root.derTT += fork.derTT; root.Et += fork.Et
                        else:    Lay += [[]]
                    H += [Lay]
            else:  # one-fork lays
                if Lay: Lay.add_lay(lay,rev)
                else:   H += [lay.copy_(rev)]

