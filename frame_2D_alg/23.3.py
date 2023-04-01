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

def comp_ptuple(_ptuple, ptuple, _fds, fds, fd):

    vertuple = Cptuple()
    Valt = [0,0]
    Rdnt = [1,1]  # not sure we need it at this point
    rn = _ptuple.n / ptuple.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)

    for pname, ave in zip(pnames, aves):  # comp full derH of each param in ptuples:

        _derH = getattr(_ptuple, pname); derH = getattr(ptuple, pname)
        if not isinstance(_ptuple.I, list):
            _derH = [_derH]; derH = [derH]
        _par = _derH[0]; par = derH[0]  # layer0 = par, same fd
        if pname=="aangle": dderH = [[comp_aangle(_par, par, Valt, ptuple=None)]]
        elif pname in ("axis","angle"): dderH = [[comp_angle(pname, _par, par, Valt, ptuple=None)]]
        else:
            if pname!="x": par *= rn  # normalize by relative accum count
            if pname=="x" or pname=="I": finv = not fds[0]
            else: finv=0
            dderH = [[comp_p(_par, par, ave, Valt, finv)]]
        # lev2:
        if len(_derH)>1 or len(derH)>1:  # and tval?
            if fd: _selH, selH = _derH[1:], derH[1:]  # select in derH
            else:  _selH, selH = _derH[1:-len(_derH)/2], derH[1:-len(derH)/2]  # skip top lev in rng+
            dderH += comp_derH(pname, _selH, selH, Valt, Rdnt, rn, _fds, fds, ave, first=1)  # also selH in recursion?

        setattr(vertuple, pname, derH+dderH)

    return vertuple, Valt, Rdnt

def comp_derH(pname, _derH, derH, Valt, Rdnt, rn, _fds, fds, ave, first):  # similar sum_derH

    dderH = []
    i=idx = 1-first; last = 2-first
    tval = ave+1
    while len(_derH)>last and len(derH)>last and _fds[idx]==fds[idx] and tval > ave:

        _lay = [md[1] for md in _derH[i:last]]; _lay_fds =_fds[:i+1]
        lay = [md[1]*rn for md in derH[i:last]]; lay_fds = fds[:i+1]
        if not _lay_fds[-1]:
            _sel, sel = _lay[:-len(_derH)/2], lay[:-len(derH)/2]  # skip top lev in rng+
        dderH += comp_derH(pname, _lay,lay, Valt,Rdnt,rn, _lay_fds, lay_fds, ave, first=0)

        i = last; last += i  # last = i*2, lenlev: 1,1,2,4,8...
        idx += 1  # elevation in derH
        tval = sum(Valt) / sum(Rdnt)

    return dderH

'''
from derH per par scheme:
'''
class Cptuple(ClusterStructure):  # bottom-layer tuple of compared params in P, derH per par in derP, or PP

    I = int  # [m,d] in higher layers:
    M = int
    Ma = float
    axis = lambda: [1, 0]  # ini dy=1,dx=0, old angle after rotation
    angle = lambda: [0, 0]  # in latuple only, replaced by float in vertuple
    aangle = lambda: [0, 0, 0, 0]
    G = float  # for comparison, not summation:
    Ga = float
    x = int  # median: x0+L/2
    L = int  # len dert_ in P, area in PP
    n = lambda: 1  # accum count, combine from CpH?
    '''
    lay1: par     # derH per param in vertuple, each layer is derivatives of all lower layers:
    lay2: [m,d]   # implicit nesting, brackets for clarity:
    lay3: [[m,d], [md,dd]]: 2 sLevs,
    lay4: [[m,d], [md,dd], [[md1,dd1],[mdd,ddd]]]: 3 sLevs, <=2 ssLevs
    '''
class CP(ClusterStructure):  # horizontal blob slice P, with vertical derivatives per param if derP, always positive

    ptuple = object  # latuple: I, M, Ma, G, Ga, angle(Dy, Dx), aangle( Sin_da0, Cos_da0, Sin_da1, Cos_da1), ?[n, val, x, L, A]?
    x0 = int
    y0 = int  # for vertical gap in PP.P__
    dert_ = list  # array of pixel-level derts, redundant to uplink_, only per blob?
    uplink_layers = lambda: [[], [[],[]]]  # init a layer of derPs and a layer of match_derPs
    downlink_layers = lambda: [[], [[],[]]]
    roott = lambda: [None,None]  # m,d seg|PP that contain this P
    rdn = int  # blob-level redundancy, ignore for now
    # only in Pd:
    dxdert_ = list
    # only in Pm:
    Pd_ = list
    # if comp_dx:
    Mdx = int
    Ddx = int

class CderP(ClusterStructure):  # tuple of derivatives in P uplink_ or downlink_: binary tree with latuple root and vertuple forks

    ptuple = list  # vertuple: len layer = sum len lower layers: 1, 1, 2, 4, 8..: one selected fork per comp_ptuple
    fds = list  # fd: der+|rng+, forming m,d per par of derH, same for clustering by m->rng+ or d->der+
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # mrdn + uprdn if branch overlap?
    _P = object  # higher comparand
    P = object  # lower comparand
    roott = lambda: [None,None]  # for der++
    x0 = int
    y0 = int
    uplink_layers = lambda: [[], [[],[]]]  # init a layer of dderPs and a layer of match_dderPs, not updated
    downlink_layers = lambda: [[], [[],[]]]
    fdx = NoneType  # if comp_dx

class CPP(CderP):  # derP params include P.ptuple

    ptuple = list  # vertuple: lenlayer = sum len lower layers: 1, 1, 2, 4, 8.., zipped with altuple in comp_ptuple
    fds = list
    valt = lambda: [0,0]
    rdnt = lambda: [1,1]  # recursion count + Rdn / nderPs + mrdn + uprdn if branch overlap?
    Rdn = int  # for accumulation only?
    rng = lambda: 1
    nval = int
    box = lambda: [0,0,0,0]  # y0,yn, x0,xn
    alt_rdn = int  # overlapping redundancy between core and edge
    alt_PP_ = list  # adjacent alt-fork PPs per PP, from P.roott[1] in sum2PP
    altuple = list  # summed from alt_PP_, sub comp support, agg comp suppression?

    P_cnt = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    derP_cnt = int  # redundant per P
    uplink_layers = lambda: [[]]  # the links here will be derPPs from discontinuous comp, not in layers?
    downlink_layers = lambda: [[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    nderP_ = list  # miss links, add with nvalt for complemented PP?
    mask__ = bool
    P__ = list  # input + derPs, common root for downward layers and upward levels:
    rlayers = list  # or mlayers: sub_PPs from sub_recursion within PP
    dlayers = list  # or alayers
    mseg_levels = list  # from 1st agg_recursion[fPd], seg_levels[0] is seg_, higher seg_levels are segP_..s
    dseg_levels = list
    roott = lambda: [None,None]  # PPPm, PPPd that contain this PP
    cPP_ = list  # rdn reps in other PPPs, to eval and remove


def comp_slice_root(blob, verbose=False):  # always angle blob, composite dert core param is v_g + iv_ga

    from sub_recursion import sub_recursion_eval, rotate_P_, agg_recursion_eval

    P__ = slice_blob(blob, verbose=False)  # form 2D array of Ps: blob slices in dert__
    # rotate each P to align it with the direction of P gradient:
    rotate_P_(P__, blob.dert__, blob.mask__)  # rotated Ps are sparse or overlap via redundant derPs, results are not biased?
    # scan rows top-down, comp y-adjacent, x-overlapping Ps, form derP__:
    _P_ = P__[0]  # higher row
    for P_ in P__[1:]:  # lower row
        for P in P_:
            for _P in _P_:  # test for x overlap(_P,P) in 8 directions, derts are positive in all Ps:
                _L = len(_P.dert_); L = len(P.dert_)
                if (P.x0 - 1 < _P.x0 + _L) and (P.x0 + L > _P.x0):
                    vertuple, valt, rdnt = comp_ptuple(_P.ptuple, P.ptuple, _fds=[1], fds=[1], fd=1)  # fd=0 only in sub+
                    derP = CderP(ptuple=vertuple, valt=valt, rdnt=rdnt, fds=[1,1], P=P, _P=_P, x0=_P.x0, y0=_P.y0)
                    P.uplink_layers[-2] += [derP]  # uplink_layers[-1] is match_derPs
                    _P.downlink_layers[-2] += [derP]
                elif (P.x0 + L) < _P.x0:
                    break  # no xn overlap, stop scanning lower P_
        _P_ = P_
    # form segments: stacks of (P,derP)s:
    segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=[0])  # shallow copy: same Ps in different lists
    segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=[0])  # initial latuple fd=0
    # form PPs: graphs of segs:
    blob.PPm_, blob.PPd_ = form_PP_root((segm_, segd_), base_rdn=2)
    # re comp, cluster:
    sub_recursion_eval(blob)  # intra PP, add rlayers, dlayers, seg_levels to select PPs, sum M,G
    agg_recursion_eval(blob, [copy(blob.PPm_), copy(blob.PPd_)])  # cross PP, Cgraph conversion doesn't replace PPs?


def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    mask__ = blob.mask__  # same as positive sign here
    dert__ = zip(*blob.dert__)  # convert 10-tuple of 2D arrays into 1D array of 10-tuple blob rows
    dert__ = [zip(*dert_) for dert_ in dert__]  # convert 1D array of 10-tuple rows into 2D array of 10-tuples per blob
    P__ = []
    height, width = mask__.shape
    if verbose: print("Converting to image...")

    for y, (dert_, mask_) in enumerate(zip(dert__, mask__)):  # unpack lines, each may have multiple slices -> Ps:
        P_ = []
        _mask = True
        for x, (dert, mask) in enumerate(zip(dert_, mask_)):  # dert = i, g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1

            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()
            g, ga, ri, angle, aangle = dert[1], dert[2], dert[3], list(dert[4:6]), list(dert[6:])
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert (m, ma, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1):
                    Pdert_ = [dert]
                    params = Cptuple(M=ave_g-g,Ma=ave_ga-ga,I=ri, angle=angle, aangle=aangle)
                else:
                    # dert and _dert are not masked, accumulate P params:
                    params.M+=ave_g-g; params.Ma+=ave_ga-ga; params.I+=ri; params.angle[0]+=angle[0]; params.angle[1]+=angle[1]
                    params.aangle = [sum(aangle_tuple) for aangle_tuple in zip(params.aangle, aangle)]
                    Pdert_.append(dert)
            elif not _mask:
                # _dert is not masked, dert is masked, terminate P:
                params.G = np.hypot(*params.angle)  # Dy, Dx  # recompute G, Ga, which can't reconstruct M, Ma
                params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)  # Cos_da0, Cos_da1
                L = len(Pdert_)
                params.L = L; params.x = x-L/2
                P_.append( CP(ptuple=params, x0=x-(L-1), y0=y, dert_=Pdert_))
            _mask = mask
        # pack last P, same as above:
        if not _mask:
            params.G = np.hypot(*params.angle); params.Ga = (params.aangle[1] + 1) + (params.aangle[3] + 1)
            L = len(Pdert_); params.L = L; params.x = x-L/2
            P_.append(CP(ptuple=params, x0=x - (L - 1), y0=y, dert_=Pdert_))
        P__ += [P_]

    blob.P__ = P__
    return P__


def form_seg_root(P__, fd, fds):  # form segs from Ps

    for P_ in P__[1:]:  # scan bottom-up, append link_layers[-1] with branch-rdn adjusted matches in link_layers[-2]:
        for P in P_: link_eval(P.uplink_layers, fd)  # uplinks_layers[-2] matches -> uplinks_layers[-1]
                     # forms both uplink and downlink layers[-1]
    seg_ = []
    for P_ in reversed(P__):  # get a row of Ps bottom-up, different copies per fPd
        while P_:
            P = P_.pop(0)
            if P.uplink_layers[-1][fd]:  # last matching derPs layer is not empty
                form_seg_(seg_, P__, [P], fd, fds)  # test P.matching_uplink_, not known in form_seg_root
            else:
                seg_.append( sum2seg([P], fd, fds))  # no link_s, terminate seg_Ps = [P]
    return seg_


def link_eval(link_layers, fd):
    # sort derPs in link_layers[-2] by their value param:
    derP_ = sorted( link_layers[-2], key=lambda derP: derP.valt[fd], reverse=True)

    for i, derP in enumerate(derP_):
        if not fd:
            rng_eval(derP, fd)  # reset derP.valt, derP.rdn
        mrdn = derP.valt[1-fd] > derP.valt[fd]  # sum because they are valt
        derP.rdnt[fd] += not mrdn if fd else mrdn

        if derP.valt[fd] > vaves[fd] * derP.rdnt[fd] * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
            link_layers[-1][fd].append(derP)
            derP._P.downlink_layers[-1][fd] += [derP]
            # misses = link_layers[-2] not in link_layers[-1], sum as PP.nvalt[fd] in sum2seg and sum2PP

# not sure:
def rng_eval(derP, fd):  # compute value of combined mutual derPs: overlap between P uplinks and _P downlinks

    _P, P = derP._P, derP.P
    common_derP_ = []

    for _downlink_layer, uplink_layer in zip(_P.downlink_layers[1::2], P.uplink_layers[1::2]):
        # overlap between +ve P uplinks and +ve _P downlinks:
        common_derP_ += list( set(_downlink_layer[fd]).intersection(uplink_layer[fd]))
    rdn = 1
    olp_val = 0
    nolp = len(common_derP_)
    for derP in common_derP_:
        rdn += derP.valt[fd] > derP.valt[1-fd]
        olp_val += derP.valt[fd]  # olp_val not reset for every derP?
        derP.valt[fd] = olp_val / nolp
    '''
    for i, derP in enumerate( sorted( link_layers[-2], key=lambda derP: derP.params[fPd].val, reverse=True)):
    if fPd: derP.rdn += derP.params[fPd].val > derP.params[1-fPd].val  # mP > dP
    else: rng_eval(derP, fPd)  # reset derP.val, derP.rdn
    if derP.params[fPd].val > vaves[fPd] * derP.rdn * (i+1):  # ave * rdn to stronger derPs in link_layers[-2]
    '''
#    derP.rdn += (rdn / nolp) > .5  # no fractional rdn?

def form_seg_(seg_, P__, seg_Ps, fd, fds):  # form contiguous segments of vertically matching Ps

    if len(seg_Ps[-1].uplink_layers[-1][fd]) > 1:  # terminate seg
        seg_.append( sum2seg( seg_Ps, fd, fds))  # convert seg_Ps to CPP seg
    else:
        uplink_ = seg_Ps[-1].uplink_layers[-1][fd]
        if uplink_ and len(uplink_[0]._P.downlink_layers[-1][fd])==1:
            # one P.uplink AND one _P.downlink: add _P to seg, uplink_[0] is sole upderP:
            P = uplink_[0]._P
            [P_.remove(P) for P_ in P__ if P in P_]  # remove P from P__ so it's not inputted in form_seg_root
            seg_Ps += [P]  # if P.downlinks in seg_down_misses += [P]
            if seg_Ps[-1].uplink_layers[-1][fd]:
                form_seg_(seg_, P__, seg_Ps, fd, fds)  # recursive compare sign of next-layer uplinks
            else:
                seg_.append( sum2seg(seg_Ps, fd, fds))
        else:
            seg_.append( sum2seg(seg_Ps, fd, fds))  # terminate seg at 0 matching uplink


def form_PP_root(seg_t, base_rdn):  # form PPs from match-connected segs

    PP_t = []  # PP fork = PP.fds[-1]
    for fd in 0, 1:
        PP_ = []
        seg_ = seg_t[fd]
        for seg in seg_:  # bottom-up
            if not isinstance(seg.roott[fd], CPP):  # seg is not already in PP initiated by some prior seg
                PP_segs = [seg]
                # add links in PP_segs:
                if seg.P__[-1].uplink_layers[-1][fd]:
                    form_PP_(PP_segs, seg.P__[-1].uplink_layers[-1][fd].copy(), fup=1, fd=fd)
                if seg.P__[0].downlink_layers[-1][fd]:
                    form_PP_(PP_segs, seg.P__[0].downlink_layers[-1][fd].copy(), fup=0, fd=fd)
                # convert PP_segs to PP:
                PP_ += [sum2PP(PP_segs, base_rdn, fd)]
        # P.roott = seg.root after term of all PPs, for form_PP_
        for PP in PP_:
            for P_ in PP.P__:
                for P in P_:
                    P.roott[fd] = PP  # update root from seg to PP
                    if fd:
                        PPm = P.roott[0]
                        if PPm not in PP.alt_PP_:
                            PP.alt_PP_ += [PPm]  # bilateral assignment of alt_PPs
                        if PP not in PPm.alt_PP_:
                            PPm.alt_PP_ += [PP]  # PPd here
        PP_t += [PP_]
    return PP_t


def form_PP_(PP_segs, link_, fup, fd):  # flood-fill PP_segs with vertically linked segments:

    # PP is a graph of 1D segs, with two sets of edges/branches: seg.uplink_, seg.downlink_.
    for derP in link_:  # uplink_ or downlink_
        if fup: seg = derP._P.roott[fd]
        else:   seg = derP.P.roott[fd]
        if seg and seg not in PP_segs:  # top and bottom row Ps are not in segs
            PP_segs += [seg]
            uplink_ = seg.P__[-1].uplink_layers[-1][fd]  # top-P uplink_
            if uplink_:
                form_PP_(PP_segs, uplink_, fup=1, fd=fd)
            downlink_ = seg.P__[0].downlink_layers[-1][fd]  # bottom-P downlink_
            if downlink_:
                form_PP_(PP_segs, downlink_, fup=0, fd=fd)


def sum2seg(seg_Ps, fd, fds):  # sum params of vertically connected Ps into segment

    uplink_, uuplink_t  = seg_Ps[-1].uplink_layers[-2:]  # uplinks of top P? or this is a bottom?
    miss_uplink_ = [uuplink for uuplink in uuplink_t[fd] if uuplink not in uplink_]  # in layer-1 but not in layer-2

    downlink_, ddownlink_t = seg_Ps[0].downlink_layers[-2:]  # downlinks of bottom P, downlink.P.seg.uplinks= lower seg.uplinks
    miss_downlink_ = [ddownlink for ddownlink in ddownlink_t[fd] if ddownlink not in downlink_]
    # seg rdn: up ini cost, up+down comp_seg cost in 1st agg+? P rdn = up+down M/n?
    P = seg_Ps[0]  # top P in stack
    seg = CPP(P__= seg_Ps, uplink_layers=[miss_uplink_], downlink_layers = [miss_downlink_], fds=copy(fds)+[fd],
              box=[P.y0, P.y0+len(seg_Ps), P.x0, P.x0+len(P.dert_)-1])
    Ptuple = deepcopy(P.ptuple)
    Dertuple = deepcopy(P.uplink_layers[-1][fd][0].ptuple) if len(seg_Ps)>1 else []  # 1 up derP per P in stack
    for P in seg_Ps[1:-1]:
        sum_ptuple(Ptuple, P.ptuple, fds,fds, fneg=0)
        derP = P.uplink_layers[-1][fd][0]  # must exist
        sum_ptuple(Dertuple, derP.ptuple, seg.fds, derP.fds)
        seg.box[2]= min(seg.box[2],P.x0); seg.box[3]= max(seg.box[3],P.x0+len(P.dert_)-1)
        # AND seg.fds?
        P.roott[fd] = seg
        for derP in P.uplink_layers[-2]:
            if derP not in P.uplink_layers[-1][fd]:
                seg.nval += derP.valt[fd]  # negative link
                seg.nderP_ += [derP]
    P = seg_Ps[-1]  # sum last P only, last P uplink_layers are not part of seg:
    sum_ptuple(Ptuple, P.ptuple, fds,fds)
    seg.box[2] = min(seg.box[2],P.x0); seg.box[3] = max(seg.box[3],P.x0+len(P.dert_)-1)
    if Dertuple:
        for pname in pnames:
            DerH = getattr(Ptuple, pname)  # 0der: single-par derH
            derH = getattr(Dertuple, pname)
            if fd: DerH += derH  # der+
            else:  DerH[:] = DerH[len(derH):] + derH  # rng+
    seg.ptuple = Ptuple  # now includes Dertuple

    return seg

def sum2PP(PP_segs, base_rdn, fd):  # sum PP_segs into PP

    from sub_recursion import append_P

    PP = CPP(x0=PP_segs[0].x0, rdn=base_rdn, fds=copy(PP_segs[0].fds)+[fd], rlayers=[[]], dlayers=[[]])
    if fd: PP.dseg_levels, PP.mseg_levels = [PP_segs], [[]]  # empty alt_seg_levels
    else:  PP.mseg_levels, PP.dseg_levels = [PP_segs], [[]]

    for seg in PP_segs:
        seg.roott[fd] = PP
        # selection should be alt, not fd, only in convert?
        if isinstance(PP.ptuple, Cptuple):
            sum_ptuple(PP.ptuple, seg.ptuple, PP.fds, seg.fds)  # not empty
        else: PP.ptuple = deepcopy(seg.ptuple)
        Y0,Yn,X0,Xn = PP.box; y0,yn,x0,xn = PP.box
        PP.box[:] = min(Y0,y0),max(Yn,yn),min(X0,x0),max(Xn,xn)
        for i in range(2):
            PP.valt[i] += seg.valt[i]
            PP.rdnt[i] += seg.rdnt[i]  # base_rdn + PP.Rdn / PP: recursion + forks + links: nderP / len(P__)?
        PP.derP_cnt += len(seg.P__[-1].uplink_layers[-1][fd])  # redundant derivatives of the same P
        # only PPs are sign-complemented, seg..[not fd]s are empty:
        PP.nderP_ += seg.nderP_
        # += miss seg.link_s:
        for i, (PP_layer, seg_layer) in enumerate(zip_longest(PP.uplink_layers, seg.uplink_layers, fillvalue=[])):
            if seg_layer:
               if i > len(PP.uplink_layers)-1: PP.uplink_layers.append(copy(seg_layer))
               else: PP_layer += seg_layer
        for i, (PP_layer, seg_layer) in enumerate(zip_longest(PP.downlink_layers, seg.downlink_layers, fillvalue=[])):
            if seg_layer:
               if i > len(PP.downlink_layers)-1: PP.downlink_layers.append(copy(seg_layer))
               else: PP_layer += seg_layer
        for P in seg.P__:
            if not PP.P__: PP.P__.append([P])  # pack 1st P
            else: append_P(PP.P__, P)  # pack P into PP.P__

        for derP in seg.P__[-1].uplink_layers[-2]:  # loop terminal branches
            if derP in seg.P__[-1].uplink_layers[-1][fd]:  # +ve links
                PP.valt[0] += derP.valt[0]  # sum val
                PP.valt[1] += derP.valt[1]
            else:  # -ve links
                PP.nval += derP.valt[fd]  # different from altuple val
                PP.nderP_ += [derP]
    return PP

def sum_ptuple(Ptuple, ptuple, Fds, fds, fneg=0):

    FH, fH = isinstance(Ptuple.I, list), isinstance(ptuple.I, list)
    for pname, ave in zip(pnames, aves):
        DerH = getattr(Ptuple, pname); derH = getattr(ptuple, pname)
        if not FH: DerH = [DerH]
        if not fH: derH = [derH]
        DerH = sum_derH(pname, DerH, derH, Fds, fds, fneg)
        setattr(Ptuple, pname, DerH)
    Ptuple.n += 1

def sum_derH(pname, DerH, derH, Fds, fds, fneg=0):  # not sure about fds

    for i, (Par, par, Fd, fd) in enumerate(zip(DerH, derH, Fds, fds)):  # loop flat list of m, d
        if not i:  # 0 is 1st lay
            if pname in ("angle","axis"):
                sin_da0 = (Par[0] * par[1]) + (Par[1] * par[0])  # sin(A+B)= (sinA*cosB)+(cosA*sinB)
                cos_da0 = (Par[1] * par[1]) - (Par[0] * par[0])  # cos(A+B)=(cosA*cosB)-(sinA*sinB)
                DerH[i] = [sin_da0, cos_da0]
            elif pname == "aangle":
                _sin_da0, _cos_da0, _sin_da1, _cos_da1 = Par
                sin_da0, cos_da0, sin_da1, cos_da1 = par
                sin_dda0 = (_sin_da0 * cos_da0) + (_cos_da0 * sin_da0)
                cos_dda0 = (_cos_da0 * cos_da0) - (_sin_da0 * sin_da0)
                sin_dda1 = (_sin_da1 * cos_da1) + (_cos_da1 * sin_da1)
                cos_dda1 = (_cos_da1 * cos_da1) - (_sin_da1 * sin_da1)
                DerH[i] = [sin_dda0, cos_dda0, sin_dda1, cos_dda1]
            else:
                DerH[i] = Par + (-par if fneg else par)
        elif Fd==fd:
            if isinstance(Par, list):
                for j, der in enumerate(par):  # par is m,d
                    Par[j] += -der if fneg else der
            else:  # 1st level par
                DerH[i] = Par + (-par if fneg else par)
        else:
            break
    return DerH

def comp_ptuple(_ptuple, ptuple, _fds, fds, fd):

    vertuple = Cptuple()
    rn = _ptuple.n / ptuple.n  # normalize param as param*rn for n-invariant ratio: _param / param*rn = (_param/_n) / (param/n)
    Rdnt = [1,1]  # not sure we need it at this point
    Valt = [0,0]
    for pname, ave in zip(pnames, aves):  # comp full derH of each param between ptuples:

        _derH = getattr(_ptuple, pname); derH = getattr(ptuple, pname)
        if not isinstance(_ptuple.I, list):
            _derH = [_derH]; derH = [derH]  # single-par derH
        if fd:  # der+
            _dH = _derH; dH = derH
        else:   # rng+, skip top lay
            _dH = _derH[:int(len(_derH)/2)]; dH = _derH[:int(len(derH)/2)]

        dderH = comp_derH(pname, _dH, dH, Valt,Rdnt,rn, _fds,fds, ave, first=1)
        setattr(vertuple, pname, dH+dderH)  # if rng+: dderH replaces top lay in derH, fd->int fr: der+ if 0, else nrng?

    return vertuple, Valt, Rdnt

def comp_derH(pname, _derH, derH, _fds, fds, Valt, Rdnt, rn, ave, first):  # similar sum_derH

    # lay1 = par or [m,d], default, then test layers 2 and 2+, for lenlay = 1,1,2,4,8..:
    _par = _derH[0]; par = derH[0]
    if first:  # lay1=par, same fd
        if pname=="aangle": dderH = [comp_aangle(_par, par, Valt, ptuple=None)]
        elif pname in ("axis","angle"): dderH = [comp_angle(pname, _par, par, Valt, ptuple=None)]
        else:
            if pname!="x": par *= rn  # normalize by relative accum count
            if pname=="x" or pname=="I": finv = not fds[0]
            else: finv=0
            dderH = [comp_p(_par, par, ave, Valt, finv)]
    elif _fds[0]==fds[0] and sum(Valt)/sum(Rdnt) > ave:
        dderH = [comp_p(_par[1], par[1], ave, Valt, finv=0)]  # comp_d in [m,d]
    # lay2 = [m,d]
    if len(_derH)>1 or len(derH)>1:
        if _fds[1]==fds[1] and sum(Valt)/sum(Rdnt) > ave:
            dderH += [comp_p(_derH[1][1], derH[1][1], ave, Valt, finv=0)]  # comp_d in [m,d]
            i=ilay=2; last=4
            # lay 2+ is len>1 subH, unpack in sub comp_derH:
            while len(_derH)>i and len(derH)>i:
                if _fds[ilay]==fds[ilay] and sum(Valt)/sum(Rdnt) > ave:  # next-lay fd
                   dderH += comp_derH(pname, _derH[i:last],derH[i:last], Valt,Rdnt,rn, _fds[:ilay],fds[:ilay], ave,first=0)
                   i=last; last+=i  # last = i*2
                   ilay += 1  # elevation in derH

    return dderH


# simplified alternative to form_graph with multi-pass inclusion,
# may also need to include m,d from pair-wise comps before recursion

def comp_centroid(G_):  # comp PP to average PP in G, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for G in G_:
        G_valt = [0 ,0]  # new total, may delete G
        G_rdn = 0  # rdn of PPs to cPPs in other Gs
        G_players_t = [[], []]
        DerPP = CderG(player=[[], []])  # summed across PP_:
        Valt = [0, 0]  # mval, dval

        for i, (PP, _, fint) in enumerate(G.PP_):  # comp PP to G centroid, derPP is replaced, use comp_plevels?
            Mplayer, Dplayer = [],[]
            # both PP core and edge are compared to G core, results are summed or concatenated:
            for fd in 0, 1:
                if PP.players_t[fd]:  # PP.players_t[1] may be empty
                    mplayer, dplayer = comp_players(G.players_t[0], PP.players_t[fd], G.fds, PP.fds)  # params norm in comp_ptuple
                    player_t = [Mplayer + mplayer, Dplayer + dplayer]
                    valt = [sum([mtuple.val for mtuple in mplayer]), sum([dtuple.val for dtuple in dplayer])]
                    Valt[0] += valt[0]; Valt[1] += valt[1]  # accumulate mval and dval
                    # accum DerPP:
                    for Ptuple, ptuple in zip_longest(DerPP.player_t[fd], player_t[fd], fillvalue=[]):
                        if ptuple:
                            if not Ptuple: DerPP.player_t[fd].append(ptuple)  # pack new layer
                            else:          sum_players([[Ptuple]], [[ptuple]])
                    DerPP.valt[fd] += valt[fd]
            # compute rdn:
            cPP_ = PP.cPP_  # sort by derPP value:
            cPP_ = sorted(cPP_, key=lambda cPP: sum(cPP[1].valt), reverse=True)
            rdn = 1
            fint = [0, 0]
            for fd in 0, 1:  # sum players per fork
                for (cPP, CderG, cfint) in cPP_:
                    if valt[fd] > PP_aves[fd] and PP.players_t[fd]:
                        fint[fd] = 1  # PPs match, sum derPP in both G and _G:
                        sum_players(G.players_t[fd], PP.players_t[fd])
                        sum_players(G.players_t[fd], PP.players_t[fd])  # all PP.players in each G.players

                    if CderG.valt[fd] > Valt[fd]:  # cPP is instance of PP
                        if cfint[fd]: G_rdn += 1  # n of cPPs redundant to PP, if included and >val
                    else:
                        break  # cPP_ is sorted by value

            fnegm = Valt[0] < PP_aves[0] * rdn;  fnegd = Valt[1] < PP_aves[1] * rdn  # rdn per PP
            for fd, fneg, in zip([0, 1], [fnegm, fnegd]):

                if (fneg and fint[fd]) or (not fneg and not fint[fd]):  # re-clustering: exclude included or include excluded PP
                    G.PP_[i][2][fd] = 1 -  G.PP_[i][2][fd]  # reverse 1-0 or 0-1
                    update_val += abs(Valt[fd])  # or sum abs mparams?
                if not fneg:
                    G_valt[fd] += Valt[fd]
                    G_rdn += 1  # not sure
                if fint[fd]:
                    # include PP in G:
                    if G_players_t[fd]: sum_players(G_players_t[fd], PP.players_t[fd])
                    else: G_players_t[fd] = copy(PP.players_t[fd])  # initialization is simpler
                    # not revised:
                    G.PP_[i][1] = derPP   # no derPP now?
                    for i, cPPt in enumerate(PP.cPP_):
                        cG = cPPt[0].root
                        for j, PPt in enumerate(cG.cPP_):  # get G and replace their derPP
                            if PPt[0] is PP:
                                cG.cPP_[j][1] = derPP
                        if cPPt[0] is PP: # replace cPP's derPP
                            G.cPP_[i][1] = derPP
                G.valt[fd] = G_valt[fd]

        if G_players_t: G.players_t = G_players_t

        # not revised:
        if G_val < PP_aves[fPd] * G_rdn:  # ave rdn-adjusted value per cost of G

            update_val += abs(G_val)  # or sum abs mparams?
            G_.remove(G)  # Gs are hugely redundant, need to be pruned

            for (PP, derPP, fin) in G.PP_:  # remove refs to local copy of PP in other Gs
                for (cPP, _, _) in PP.cPP_:
                    for i, (ccPP, _, _) in enumerate(cPP.cPP_):  # ref of ref
                        if ccPP is PP:
                            cPP.cPP_.pop(i)  # remove ccPP tuple
                            break

    if update_val > sum(PP_aves):
        comp_centroid(G_)  # recursion while min update value

    return G_

def comp_derH(_derH, derH, _fds, fds):

    dderH = []; valt = [0,0]; rdnt = [1,1]; elev=0

    for i, (_ptuple,ptuple, _fd,fd) in enumerate(zip(_derH, derH, _fds, fds)):
        if _fd!=fd:
            break
        if elev in (0,1) or not (i+1)%(2**elev):  # first 2 levs are single-element, higher levs are 2**elev elements
            elev += 1
            _ptuple,_ext = _ptuple; ptuple,ext = ptuple  # ext per lev only
            mext,dext = comp_ext(_ext[0],ext[0])  # comp dext only
            valt[0]+=sum(mext); valt[1]+=sum(dext)
            felev=1
        else:
            felev=0
        dtuple = comp_vertuple(_ptuple,ptuple) if i else comp_ptuple(_ptuple,ptuple)
        for j in 0,1:
            valt[j] += ptuple.valt[j]; rdnt[j] += ptuple.rdnt[j]
        if felev:
            dderH += [[dtuple, [dext, mext]]]
        else:
            dderH += [dtuple]

    return dderH, valt, rdnt
