class Cgraph(ClusterStructure):  # params of single-fork node_ cluster per pplayers

    G = lambda: None  # same-scope lower-der|rng G.G.G., in all nodes beyond PP
    root = lambda: None  # root graph or inder_ G, element of ex.H[-1][fd]
    # up,down trees:
    ex = object    # Inder_ ) link_) uH: context, Lev+= root tree slice: forward, comp summed up-forks?
    inder_ = list  # inder_ ) node_) wH: contents, Lev+= node tree slice: feedback, Lev/agg+, lev/sub+?
    # inder_ params:
    node_ = list  # single-fork, conceptually H[0], concat sub-node_s in ex.H levs
    fterm = lambda: 0  # G.node_ sub-comp or feedback was terminated
    Link_ = lambda: Clink_()  # unique links within node_, -> inder_?
    H = list  # Lev+= node'tree'slice, ~up-forking Levs in ex.H?
    val = int
    fds = list
    rdn = lambda: 1
    rng = lambda: 1
    nval = int  # of open links: base alt rep
    box = lambda: [0,0,0,0,0,0]  # center y,x, y0,x0, yn,xn, no need for max distance?
    x0 = float
    y0 = float  # center: box x0|y0 + L/2
    xn = float  # max distance from center
    yn = float
    L = list  # der L, init None
    S = int  # sparsity: ave len link
    A = list  # area|axis: Dy,Dx, ini None
    alt_graph_ = list  # contour + overlapping contrast graphs
    alt_Graph = None  # conditional, summed and concatenated params of alt_graph_

class CderG(ClusterStructure):  # graph links, within root node_

    node0 = lambda: Cgraph()  # converted to list in recursion
    node1 = lambda: Cgraph()
    minder_ = list  # in alt/contrast if open
    dinder_ = list
    S = int  # sparsity: ave len link
    A = list  # area and axis: Dy,Dx

def fforward(root):  # top-down update node.ex.H, breadth-first

    for node in root.node_:
        for i, (rLev,nLev) in enumerate(zip_longest(root.ex.H, node.ex.H[1:], fillvalue=[])):  # root.ex.H maps to node.ex.H[1:]
            if rLev:
                j = sum(fd*(2**k) for k,fd in enumerate(rLev.fds[i:]))
                if not nLev:  # init:
                    nLev=CpH(H=[Cgraph() for fork in rLev.fds[i:]])  # fds=copy(rLev.fds)?
                sum_inder_(nLev.H[j].inder_, rLev.H[j].inder_)  # same fork
        ffV,ffR = 0,0
        for Lev in root.ex.H:
            ffV += Lev.val; ffR += Lev.rdn

        if node.fterm and ffV/ffR > aveG:
            node.fterm = 0
            fforward(node)


def sum2graph_(root, graph_, fd):  # sum node and link params into graph, inder_ in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.val < aveG:  # form graph if val>min only
            continue
        X0,Y0 = 0,0
        for G in graph.H:  # CpH
            X0 += G.x0; Y0 += G.y0
        L = len(graph.H); X0/=L; Y0/=L; Xn,Yn = 0,0
        # conditional ex.inder_: remove if few links?
        Graph = Cgraph(fds=copy(G.fds), ex=Cgraph(node_=Clink_(),A=[0,0]), x0=X0, xn=Xn, y0=Y0, yn=Yn)
        # form G, keep iG:
        node_,Link_= [],[]
        for iG in graph.H:
            Xn = max(Xn, (iG.x0 + iG.xn) - X0)  # box xn = x0+xn
            Yn = max(Yn, (iG.y0 + iG.yn) - Y0)
            sum_G(Graph, iG)  # sum(Graph.uH[-1][fd], iG.pplayers), higher levs += G.G.pplayers | G.uH, lower scope than iGraph
            link_ = [iG.ex.node_.Qm, iG.ex.node_.Qd][fd]
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds), G=iG, root=Graph, ex=Cgraph(node_=Clink_(),A=[0,0]))
            # sum quasi-gradient of links in ex.inder_: redundant to Graph.inder_, if len node_?:
            for derG in link_:
                sum_inder_(G.ex.inder_, [derG.minder_, derG.dinder_][fd]) # local feedback
                G.ex.S += derG.S; G.ex.A[0]+=derG.A[0]; G.ex.A[1]+=derG.A[1]
            l=len(link_); G.ex.L=l; G.ex.S/=l
            sum_ex_H(root, iG)
            node_ += [G]
        Graph.node_ = node_ # lower nodes = G.G..; Graph.root = iG.root
        for Link in Link_:  # sum unique links
            sum_inder_(Graph.inder_, [Link.minder_, Link.dinder_][fd])
            if Graph.inder_[-1]: # top ext
                Graph.inder_[-1][1]+=Link.S; Graph.inder_[-1][2][0]+=Link.A[0]; Graph.inder_[-1][2][1]+=Link.A[1]
            else: Graph.inder_[-1] = [1,Link.S,Link.A]
        L = len(Link_); Graph.inder_[-1][0] = L; Graph.inder_[-1][1] /= L  # last pplayers LSA per Link_
        # inder_ LSA per node_:
        Graph.A = [Xn*2,Yn*2]; L=len(node_); Graph.L=L; Graph.S = np.hypot(Xn-X0,Yn-Y0) / L
        if Graph.ex.H:
            Graph.val += sum([lev.val for lev in Graph.ex.H]) / sum([lev.rdn for lev in Graph.ex.H])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_

def comp_GQ(_G,G,fsub):  # compare lower-derivation G.G.s, pack results in minder__,dinder__

    minder__,dinder__ = [],[]; Mval,Dval = 0,0; Mrdn,Drdn = 1,1
    Tval= aveG+1  # start
    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G

        minder_, dinder_, mval, dval, mrdn, drdn = comp_G(_G, G, fsub, fex=0)
        minder__+=minder_; dinder__+=dinder_
        Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # also /rdn+1: to inder_?
        # comp ex:
        if (Mval + Dval) * _G.ex.val * G.ex.val > aveG:
            mex_, dex_, mval, dval, mrdn, drdn = comp_G(_G.ex, G.ex, fsub, fex=1)
            minder__+=mex_; dinder__+=dex_
            Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn
        else:
            minder__+=[[]]; dinder__+=[[]]  # always after not-empty m|dinder_: list until sum2graph?
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return minder__, dinder__, Mval, Dval, Tval  # ext added in comp_G_, not within GQ

def comp_G(_G, G, fsub, fex):

    minder_,dinder_ = [],[]  # ders of implicitly nested list of pplayers in inder_
    Mval, Dval = 0,0; Mrdn, Drdn = 1,1

    minder_,dinder_, Mval,Dval, Mrdn,Drdn = comp_inder_(_G.inder_,G.inder_, minder_,dinder_, Mval,Dval, Mrdn,Drdn)
    # spec:
    _node_, node_ = _G.node_.Q if fex else _G.node_, G.node_.Q if fex else G.node_  # node_ is link_ if fex
    if (Mval+Dval) * _G.val*G.val * len(_node_)*len(node_) > aveG:
        if fex:  # comp link_
            sub_minder_,sub_dinder_ = comp_derG_(_node_, node_, G.fds[-1])
        else:    # comp node_
            sub_minder_,sub_dinder_ = comp_G_(_node_, node_, f1Q=0, fsub=fsub)
        Mval += sum([mxpplayers.val for mxpplayers in sub_minder_])  # + rdn?
        Dval += sum([dxpplayers.val for dxpplayers in sub_dinder_])
        # + sub node_?
        if (Mval+Dval) * _G.val*G.val * len(_G.ex.H)*len(G.ex.H) > aveG:
            # comp uH
            for _lev, lev in zip(_G.ex.H, G.ex.H):  # wH is empty in top-down comp
                for _fork, fork in zip(_lev.H, lev.H):
                    if _fork and fork:
                        fork_minder_, fork_dinder_, mval, dval, tval = comp_GQ(_fork, fork, fsub=0)
                        Mval+= mval; Dval+= dval
                    # else +[[]], if m|dexH:
    # pack m|dnode_ and m|dexH in minder_, dinder_: still implicit or nested?
    else: _G.fterm=1
    # no G.fterm=1: it has it's own specification?
    # comp alts,val,rdn?
    return minder_, dinder_, Mval, Dval, Mrdn, Drdn

def comp_derG_(_derG_, derG_, fd):

    mlink_,dlink_ = [],[]
    for _derG in _derG_:
        for derG in derG_:
            mlink, dlink,_,_,_,_,= comp_inder_([_derG.minder_,_derG.dinder_][fd], [derG.dinder_,_derG.dinder_][fd], [],[],0,0,1,1)
            # add comp fds: may be different?
            mext, dext = comp_ext(1,_derG.S,_derG.A, 1,derG.S,derG.A)
            mlink_ += [mlink] + [mext]  # not sure
            dlink_ += [dlink] + [dext]

    return mlink_, dlink_

def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # sum nodes in root, sub_nodes in root.H:
                for sub_node in node.node_:
                    fd = sub_node.fds[-1]
                    if not root.H: root.H = [CpH(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    if isinstance(sub_node.G, list):
                        sub_inder_ = sub_node.inder_[fd]
                    else:
                        sub_inder_ = sub_node.inder_
                    sum_inder_(root.H[0].H[fd].inder_, sub_inder_)
                    # or sum_G?
                    # sum_H(root.H[1:], sub_node.H)
                    for i, (Lev,lev) in enumerate(zip_longest(root.H[1:], sub_node.H, fillvalue=[])):
                        if lev:
                            j = sum(fd*(2**k) for k,fd in enumerate(sub_node.fds[i:]))
                            if not Lev: Lev = CpH(H=[[] for fork in range(2**(i+1))])  # n forks *=2 per lev
                            if not Lev.H[j]: Lev.H[j] = Cgraph()
                            sum_inder_(Lev.H[j].inder_, lev.H[j].inder_)
                            # or sum_G?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root

def sum_G(G, g, fmerge=0):  # g is a node in G.node_

    sum_inder_(G.inder_, g.inder_)  # direct node representation
    # if g.uH: sum_H(G.uH, g.uH[1:])  # sum g->G
    if g.H:
        sum_H(G.H[1:], g.H)  # not in sum2graph
    G.L += g.L; G.S += g.S
    if isinstance(g.A, list):
        if G.A:
            G.A[0] += g.A[0]; G.A[1] += g.A[1]
        else: G.A = copy(g.A)
    else: G.A += g.A
    G.val += g.val; G.rdn += g.rdn; G.nval += g.nval
    Y,X,Y0,Yn,X0,Xn = G.box[:]; y,x,y0,yn,x0,xn = g.box[:]
    G.box[:] = [Y+y, X+x, min(X0,x0), max(Xn,xn), min(Y0,y0), max(Yn,yn)]
    if fmerge:
        for node in g.node_:
            if node not in G.node_: G.node_ += [node]
        for link in g.Link_.Q:  # redundant?
            if link not in G.Link_.Q: G.Link_.Q += [link]
        for alt_graph in g.alt_graph_:
            if alt_graph not in G.alt_graph: G.alt_graph_ += [alt_graph]
        if g.alt_Graph:
            if G.alt_Graph: sum_G(G.alt_Graph, g.alt_Graph)
            else:           G.alt_Graph = deepcopy(g.alt_graph)
    else: G.node_ += [g]

def sum2graph_(graph_, fd, fsub=0):  # sum node and link params into graph, derH in agg+ or player in sub+

    Graph_ = []  # Cgraphs
    for graph in graph_:  # CpHs
        if graph.valt[fd] < aveG:  # form graph if val>min only
            continue
        Graph = Cgraph(fds=copy(graph.H[0].fds)+[fd])  # incr der
        ''' if mult roots: 
        sum_derH(Graph.uH[0][fd].derH,root.derH) or sum_G(Graph.uH[0][fd],root)? init if empty
        sum_H(Graph.uH[1:], root.uH)  # root of Graph, init if empty
        '''
        node_,Link_ = [],[]  # form G, keep iG:
        for iG in graph.H:
            sum_G(Graph, iG, fmerge=0)  # local subset of lower Gs in new graph
            link_ = [iG.link_.Qm, iG.link_.Qd][fd]  # mlink_,dlink_
            Link_ = list(set(Link_ + link_))  # unique links in node_
            G = Cgraph(fds=copy(iG.fds)+[fd], root=Graph, node_=link_, box=copy(iG.box))  # no sub_nodes in derG, remove if <ave?
            for derG in link_:
                sum_box(G.box, derG.G[0].box if derG.G[1] is iG else derG.G[1].box)
                sum_derH(G.derH, derG.derH[fd])  # two-fork derGs are not modified
                Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
            add_ext(G.box, len(link_), G.derH[-1])  # composed node ext, not in derG.derH
            # if mult roots: sum_H(G.uH[1:], Graph.uH)
            node_ += [G]
        Graph.root = iG.root  # same root, lower derivation is higher composition
        Graph.node_ = node_  # G| G.G| G.G.G..
        for derG in Link_:  # sum unique links, not box
            sum_derH(Graph.derH, derG.derH[fd])
            Graph.valt[0] += derG.valt[0]; Graph.valt[1] += derG.valt[1]
        Ext = [0,0,[0,0]]
        Ext = [sum_ext(Ext, G.derH[-1][1]) for G in node_]  # add composed node Ext
        add_ext(Graph.box, len(node_), Ext) # add composed graph Ext
        Graph.derH += [Ext]  # [node Ext, graph Ext]
        # if Graph.uH: Graph.val += sum([lev.val for lev in Graph.uH]) / sum([lev.rdn for lev in Graph.uH])  # if val>alt_val: rdn+=len_Q?
        Graph_ += [Graph]

    return Graph_

def comp_G_(G_, pri_G_=None, f1Q=1, fsub=0):  # cross-comp Graphs if f1Q, else G_s in comp_node_, or segs inside PP?

    if not f1Q: minder__,dinder__ = [],[]

    for i, _iG in enumerate(G_ if f1Q else pri_G_):  # G_ is node_ of root graph
        for iG in G_[i+1:] if f1Q else G_:  # compare each G to other Gs in rng, bilateral link assign, val accum:
            # if the pair was compared in prior rng+:
            if iG in [node for link in _iG.link_.Q for node in link.node_]:  # if f1Q? add frng to skip?
                continue
            dy = _iG.box[0]-iG.box[0]; dx = _iG.box[1]-iG.box[1]  # between center x0,y0
            distance = np.hypot(dy, dx)  # Euclidean distance between centers, sum in sparsity, proximity = ave-distance
            if distance < ave_distance * ((_iG.val + iG.val) / (2*sum(G_aves))):
                # same for cis and alt Gs:
                for _G, G in ((_iG, iG), (_iG.alt_Graph, iG.alt_Graph)):
                    if not _G or not G:  # or G.val
                        continue
                    minder_, dinder_, mval, dval, tval = comp_GQ(_G,G)  # comp_G while G.G, H/0G: GQ is one distributed node?
                    ext = [1,distance,[dy,dx]]
                    derG = Cgraph(G=[_G,G], inder_=[minder_+[[ext]], dinder_+[[ext]]], box=[])  # box is redundant to 2 nodes
                    # add links:
                    _G.link_.Q += [derG]; _G.link_.val += tval  # combined +-links val
                    G.link_.Q += [derG]; G.link_.val += tval
                    if mval > aveGm:
                        _G.link_.Qm += [derG]; _G.link_.mval += mval  # no dval for Qm
                        G.link_.Qm += [derG]; G.link_.mval += mval
                    if dval > aveGd:
                        _G.link_.Qd += [derG]; _G.link_.dval += dval  # no mval for Qd
                        G.link_.Qd += [derG]; G.link_.dval += dval

                    if not f1Q:  # comp G_s
                        minder__+= minder_; dinder__+= dinder_
                # implicit cis, alt pair nesting in minder_, dinder_
    if not f1Q:
        return minder__, dinder__  # else no return, packed in links


def comp_GQ(_G, G):  # compare lower-derivation G.G.s, pack results in minder__,dinder__

    minder__,dinder__ = [],[]; Mval,Dval = 0,0; Mrdn,Drdn = 1,1
    Tval= aveG+1
    while (_G and G) and Tval > aveG:  # same-scope if sub+, no agg+ G.G

        minder_, dinder_, mval, dval, mrdn, drdn = comp_G(_G, G)
        minder__+=minder_; dinder__+=dinder_; Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn  # rdn+=1: to inder_?
        _G = _G.G
        G = G.G
        Tval = (Mval+Dval) / (Mrdn+Drdn)

    return minder__, dinder__, Mval, Dval, Tval  # ext added in comp_G_, not within GQ

def comp_G(_G, G):  # in GQ

    mderH,dderH = [],[]  # ders of implicitly nested list of pplayers in derH
    # or single dderH:
    Mval, Dval = 0,0
    Mrdn, Drdn = 1,1
    if _G.box: _derH, derH = _G.derH, G.derH
    else:
        _fd = _G.root.fds[-1] if _G.root.fds else 0; fd = G.root.fds[-1] if G.root.fds else 0
        _derH, derH = _G.derH[_fd], G.derH[fd]  # derG if comp node_?

    mderH,dderH, Mval,Dval, Mrdn,Drdn = comp_derH(_derH,derH, mderH,dderH, Mval,Dval, Mrdn,Drdn)
    # spec:
    _node_, node_ = _G.node_, G.node_  # link_ if fd, sub_node should be empty
    if (Mval+Dval)* sum(_G.valt)*sum(G.valt) * len(_node_)*len(node_) > aveG:  # / rdn?

        sub_mderH,sub_dderH = comp_G_(_node_, node_, f1Q=0)
        Mval += sum([mxpplayers.val for mxpplayers in sub_mderH])  # + rdn?
        Dval += sum([dxpplayers.val for dxpplayers in sub_dderH])
        # pack m|dnode_ in m|dderH: implicit?
    else: _G.fterm=1  # no G.fterm=1: it has it's own specification?
    '''
    comp alts,val,rdn?
    comp H in comp_G_?
    select >ave m|d vars only: addressable salient mset / dset in derH? 
    cluster per var or set if recurring across root: type eval if root M|D?
    '''
    return mderH, dderH, Mval, Dval, Mrdn, Drdn

def comp_derH(_derH, derH, mderH,dderH, Mval,Dval, Mrdn,Drdn):

    i=0; end=1; Tval = aveG+1; elev=0
    while end <= min(len(_derH),len(derH)) and Tval > aveG:

        _Lev, Lev = _derH[i:end], derH[i:end]  # each Lev of implicit nesting is derH,ext formed by comp_G
        for _der,der in zip(_Lev,Lev):
            if der:
                if isinstance(der,CpH):  # pplayers, incr implicit nesting in m|dpplayers:
                    mpplayers, dpplayers = comp_pH(_der, der)
                    mderH += [mpplayers]; Mval += mpplayers.valt[0]; Mrdn += mpplayers.rdn  # add rdn in form_?
                    dderH += [dpplayers]; Dval += dpplayers.valt[1]; Drdn += dpplayers.rdn
                else:
                    mextt, dextt = [],[]
                    for _extt, extt in _der, der:  # [node_ext, graph_ext], both are full|empty per der+?
                        mext, dext = comp_ext(_extt,extt)
                        mextt+=[mext]; dextt+=[dext]; Mval+=sum(mext); Dval+=sum(dext)
                    mderH += [mextt]; dderH += [dextt]
            else:
                mderH+=[[]]; dderH+=[[]]
        Tval = (Mval+Dval) / (Mrdn+Drdn)  # eval if looping Levs
        i = end
        end = (end*2) + 1*elev
        elev = 1
    '''
    lenLev = (end*2)+1: 1, 1+1, 3+1, 7+1, 15+1.: +[der_ext, agg_ext] per G in GQ, levs | Levs? same fds till += [fd]?
    Lev1: pps: 1 pplayers  # 0der
    Lev2: pps,ext: lenLev = 2   
    Lev3: pps; pps,ext; ext: lenLev = 4
    Lev4: pps; pps,ext; pps,pps,ext,ext; ext: lenLev = 8
    '''
    return mderH,dderH, Mval,Dval, Mrdn,Drdn

def comp_derH1(_derH, derH, Mval,Dval, Mrdn,Drdn):

    dderH = []
    for _Lev, Lev in zip(derH, derH):  # each Lev or subLev is CpH,ext formed by comp_G
        for _der,der in zip(_Lev,Lev):
            if _der and der:  # probably not needed
                if isinstance(der,CpH):  # players, incr implicit nesting in dplayers?
                    dplayers = comp_pH(_der, der)
                    Mval += dplayers.valt[0]; Mrdn += dplayers.rdnt[0]  # add rdn in form_?
                    Dval += dplayers.valt[1]; Drdn += dplayers.rdnt[1]
                    dderH += [dplayers]
                else:  # list
                    dder, Mval, Dval, Mrdn, Drdn = comp_derH1(_der[0], der[0], Mval, Dval, Mrdn, Drdn)
                    mext, dext = comp_ext(_der[1],der[1])
                    Mval+=sum(mext); Dval+=sum(dext)
                    dderH += [[dder, [mext,dext]]]
            else:
                dderH += [[]]  # probably not needed
        if (Mval+Dval) / (Mrdn+Drdn) < ave_G:
            break

    return dderH, Mval,Dval, Mrdn,Drdn
'''
    Lev1: lays: CpH 0der players, ext is added per G Lev:
    Lev2: [dlays, ext]: list   
    Lev3: [[dlays, [ddlays,dext]], ext]: nested list, max 1 sub_lev
    Lev4: [[[dlays, [ddlays,dext]], [[ddlays, [dddlays,ddext],dext]]], ext]: nnested list, max 2 sub_levs
'''

def sum_derH(DerH, derH, fext=1):

    for i, (Der, der) in enumerate(zip_longest(DerH, derH, fillvalue=None)):
        if der is not None:
            if der:
                if Der:
                    if isinstance(der,CpH): sum_pH(Der,der)
                    else:
                        for Ext,ext in zip(Der,der):  # pair
                            sum_ext(Ext,ext)
                else: derH.insert(i,deepcopy(der))  # for different-length DerH, derH

            elif Der is None: derH += [deepcopy(der)]
            else:             derH[i] = deepcopy(der)

def comp_ptuple(_params, params, fd=0):  # compare lateral or vertical tuples, similar operations for m and d params

    dtuple, mtuple = Cptuple(), Cptuple()
    rn = _params.n / params.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    if fd:  # vertuple, all params are scalars:
        comp("val", _params.val, params.val*rn, dtuple, mtuple, ave_mval, finv=0)
        comp("axis", _params.axis, params.axis*rn, dtuple, mtuple, ave_dangle, finv=0)
        comp("angle", _params.angle, params.angle*rn, dtuple, mtuple, ave_dangle, finv=0)
        comp("aangle", _params.aangle, params.aangle*rn, dtuple, mtuple, ave_daangle, finv=0)
    else:  # latuple
        comp("G", _params.G, params.G*rn, dtuple, mtuple, ave_G, finv=0)
        comp("Ga", _params.Ga, params.Ga*rn, dtuple, mtuple, ave_Ga, finv=0)
        comp_angle("axis", _params.axis, params.axis, dtuple, mtuple)  # rotated, thus no adjustment by daxis?
        comp_angle("angle", _params.angle, params.angle, dtuple, mtuple)
        comp_aangle(_params.aangle, params.aangle, dtuple, mtuple)
    # either:
    comp("I", _params.I, params.I*rn, dtuple, mtuple, ave_dI, finv=not fd)  # inverse match if latuple
    comp("M", _params.M, params.M*rn, dtuple, mtuple, ave_M, finv=0)
    comp("Ma",_params.Ma, params.Ma*rn, dtuple, mtuple, ave_Ma, finv=0)
    comp("L", _params.L, params.L*rn, dtuple, mtuple, ave_L, finv=0)
    comp("x", _params.x, params.x, dtuple, mtuple, ave_x, finv=not fd)
    # adjust / daxis+dx: Dim compensation in same area, alt axis definition?

    return mtuple, dtuple

def comp(param_name, _param, param, dtuple, mtuple, ave, finv):

    d = _param-param
    if finv: m = ave - abs(d)  # inverse match for primary params, no mag/value correlation
    else:    m = min(_param,param) - ave
    dtuple.val += abs(d)
    mtuple.val += m
    setattr(dtuple, param_name, d)  # dtuple.param_name = d
    setattr(mtuple, param_name, m)  # mtuple.param_name = m


def comp_ptuple_(_layers, layers):  # unpack and compare der layers, if any from der+, no fds, same within PP

    vertuples = []; mval, dval = 0,0
    pri_fd = 0  # ptuple, else vert dptuple
    for _layer, layer in zip(_layers[0], layers[0]):

        for _ptuple, ptuple in zip(_layer[0], layer[0]):
            vertuple = comp_ptuple(_ptuple, ptuple, pri_fd)
            vertuples += [vertuple];  mval+=vertuple.valt[0]; dval+=dLine.valt[1]
        pri_fd=1

    return [vertuple,mval,dval]


def comp_ptuple(_params, params, fd=0):  # compare latuples or vertuples, similar operations for m and d params
    ptuple = Cptuple()
    Valt = [0,0]
    rn = _params.n / params.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    comp_p("I", _params.I[fd], params.I[fd]*rn, ave_dI, Valt, ptuple, finv=not fd)
    comp_p("M", _params.M[fd], params.M[fd]*rn, ave_M, Valt, ptuple, finv=0)
    comp_p("Ma", _params.Ma[fd], params.Ma[fd]*rn, ave_Ma, Valt, ptuple, finv=0)
    comp_p("L", _params.L[fd], params.L[fd]*rn, ave_L, Valt, ptuple, finv=0)
    comp_p("x", _params.x[fd], params.x[fd], ave_x, Valt, ptuple, finv=not fd)
    comp_p("G", _params.G[fd], params.G[fd]*rn, ave_G, Valt, ptuple, finv=0)
    comp_p("Ga", _params.Ga[fd], params.Ga[fd]*rn, ave_Ga, Valt, ptuple, finv=0)
    if fd:
        comp_p("axis", _params.axis[fd], params.axis[fd]*rn, ave_dangle, Valt, ptuple, finv=0)
        comp_p("angle", _params.angle[fd], params.angle[fd]*rn, ave_dangle, Valt, ptuple, finv=0)
        comp_p("aangle", _params.aangle[fd], params.aangle[fd]*rn, ave_daangle, Valt, ptuple, finv=0)
    else:
        comp_angle("axis", _params.axis[fd], params.axis[fd], Valt, ptuple)  # rotated, thus no adjustment by daxis?
        comp_angle("angle", _params.angle[fd], params.angle[fd], Valt, ptuple)
        comp_aangle(_params.aangle[fd], params.aangle[fd], ptuple, Valt)
    # adjust / daxis+dx: Dim compensation in same area, alt axis definition?
    return ptuple, Valt

def sum_ptuple(Layers, layers, fneg=0):  # same fds from comp_ptuple

    if Layers:
        for Layer, layer in zip_longest(Layers[0], layers[0], fillvalue=[]):
            if layer:
                if Layer:
                    if isinstance(Layer, Cptuple):
                        sum_ptuple(Layer, layer, fneg)  # Layer is ptuple
                    else:
                        for ptuple, ptuple in zip(Layer[0], layer[0]):  # ptuples, ptuples
                            sum_ptuple(ptuple, ptuple, fneg)
                        Layer[1] += layer[1]  # sum ptuples val
                elif not fneg:
                    Layers[0].append(deepcopy(layer))
        Layers[1] += layers[1]  # sum vertuple val
    elif not fneg:
        Layers[:] = deepcopy(layers)

def sum_ptuple(Ptuple, ptuple, fneg=0):

    for pname in ptuple.numeric_params:
        if pname != "G" and pname != "Ga":
            Param = getattr(Ptuple, pname)
            param = getattr(ptuple, pname)
            if fneg: out = Param - param
            else:    out = Param + param
            setattr(ptuple, pname, out)  # update value
    if isinstance(ptuple.angle, list):
        for i, angle in enumerate(ptuple.angle):
            if fneg: Ptuple.angle[i] -= angle
            else:    Ptuple.angle[i] += angle
        for i, aangle in enumerate(ptuple.aangle):
            if fneg: Ptuple.aangle[i] -= aangle
            else:    Ptuple.aangle[i] += aangle
    else:
        if fneg: Ptuple.angle -= ptuple.angle; Ptuple.aangle -= ptuple.aangle
        else:    Ptuple.angle += ptuple.angle; Ptuple.aangle += ptuple.aangle



