def agg_level(inputs):  # draft parallel

    frame, H, elevation = inputs
    lev_G = H[elevation]
    cross_comp(lev_G, rc=0, iN_=lev_G.node_, fi=1)  # return combined top composition level, append frame.derH
    if lev_G:
        # feedforward
        if len(H) < elevation+1: H += [lev_G]  # append graph hierarchy
        else: H[elevation+1] = lev_G
        # feedback
        if elevation > 0:
            if np.sum( np.abs(lev_G.aves - lev_G._aves)) > ave:  # filter update value, very rough
                m, d, n, o = lev_G.Et
                k = n * o
                m, d = m/k, d/k
                H[elevation-1].aves = [m, d]
            # else break?

def agg_H_par(focus):  # draft parallel level-updating pipeline

    frame = frame_blobs_root(focus)
    intra_blob_root(frame)
    vect_root(frame)
    if frame.node_:  # converted edges
        G_ = []
        for edge in frame.node_:
            comb_alt_(edge.node_, ave)
            # cluster_C_(edge, ave)  # no cluster_C_ in vect_edge
            G_ += edge.node_  # unpack edges
        frame.node_ = G_
        manager = Manager()
        H = manager.list([CG(node_=G_)])
        maxH = 20
        with Pool(processes=4) as pool:
            pool.map(agg_level, [(frame, H, e) for e in range(maxH)])

        frame.aggH = list(H)  # convert back to list

def cross_comp(root, rc, iN_, fi=1):  # rc: redundancy count; (cross-comp, exemplar selection, clustering), recursion

    N__,L_,Et = comp_node_(iN_,rc) if fi else comp_link_(iN_,rc)  # root.olp is in rc
    if N__:  # CLs if not fi
        nG, n__ = [],[]  # mfork
        for n in [N for N_ in N__ for N in N_]:
            n__ += [n]; n.sel = 0  # for cluster_N_
        if val_(Et, mw=(len(n__)-1)*Lw, aw=rc+loop_w) > 0:  # rc += is local
            E_,eEt = sel_exemplars(n__, rc+loop_w, fi)  # sel=1 for typical sparse nodes
            # call from sel_exemplars?:
            if val_(eEt, mw=(len(E_)-1)*Lw, aw=rc+clust_w) > 0:
                C_,cEt = cluster_C_(root, E_, rc+clust_w)  # refine _N_+_N__ by mutual similarity
                # internal cross_comp C_, recursion?
                if val_(cEt, mw=(len(C_)-1)*Lw, aw=rc+loop_w) > 0:  # refine exemplars by new et
                    S_,sEt = sel_exemplars([n for C in C_ for n in C.node_], rc+loop_w, fi, fC=1)
                else: S_,sEt = [],np.zeros(3)
                if val_(sEt+eEt, mw=(len(S_+E_)-1)*Lw, aw=rc+clust_w) > 0:
                    for rng, N_ in enumerate(N__, start=1):  # bottom-up, cluster via rng exemplars:
                        en_ = [n for n in N_ if n.sel]; eet = np.sum([n.et for n in en_])
                        if val_(eet, mw=(len(en_)-1)*Lw, aw=rc+clust_w*rng) > 0:
                            nG = cluster_N_(root, en_, rc+clust_w*rng, fi, rng)
                    if nG and val_(nG.Et, Et, mw=(len(nG.node_)-1)*Lw, aw=rc+clust_w*rng+loop_w) > 0:
                        nG = cross_comp(nG, rc+clust_w*rng+loop_w, nG.node_)
                        # top-composition, select unpack lower-rng nGs for deeper cross_comp
        lG = []  # dfork
        dval = val_(Et, mw=(len(L_)-1)*Lw, aw=rc+3+clust_w, fi=0)
        if dval > 0:
            if dval > ave:  # recursive derivation forms lH in each node_H level, Len banded?
                lG = cross_comp(sum_N_(L_), rc+loop_w*2, L2N(L_), fi=0)  # comp_link_, no centroids?
            else:  # lower res, dL_?
                lG = cluster_N_(sum_N_(L_),L2N(L_), rc+clust_w*2, fi=0, fnodet=1)  # overlaps the mfork above
            if nG: comb_alt_(nG.node_, rc+clust_w*3)
        if nG or lG:
            root.H += [[nG,lG]]  # current lev
            if nG: add_N(root,nG); add_node_H(root.H, nG.H, root)  # appends derH,H if recursion
            if lG: add_N(root,lG); root.lH += lG.H + [sum_N_(copy(lG.node_), root=lG)]  # lH: H within node_ level
        if nG:
            return nG
        # add_node_H(H, Fg.H, Fg)  # not needed?

    class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
        # graph / node
        def __init__(G, **kwargs):
            super().__init__()
            G.node_ = kwargs.get('node_', [])
            G.link_ = kwargs.get('link_', [])  # spliced node link_s
            G.H = kwargs.get('H', [])  # list of lower levels: [nG,lG]: pack node_,link_ in sum2graph; lG.H: packed-level lH:
            G.lH = kwargs.get('lH', [])  # link_ agg+ levels in top node_H level: node_,link_,lH
            G.Et = kwargs.get('Et', np.zeros(3))  # sum all M,D,n from link_ and node_
            G.et = kwargs.get('et', np.zeros(3))  # sum from rim
            G.olp = kwargs.get('olp', 1)  # overlap to other Gs
            G.baseT = kwargs.get('baseT', np.zeros(4))  # I,G,Dy,Dx  # from slice_edge, + baset from rim?
            G.derTT = kwargs.get('derTT', np.zeros((2, 8)))  # m,d / Et,baseT: [M,D,n,o, I,G,A,L], summed across derH lay forks
            G.extTT = kwargs.get('extTT', np.zeros((2, 8)))  # sum across extH
            G.derH = kwargs.get('derH', [])  # each lay is [m,d]: Clay(Et,node_,link_,derTT), sum|concat links across fork tree
            G.extH = kwargs.get('extH', [])  # sum from rims, single-fork
            G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y+Y)/2,(x,X)/2], then ave node yx
            G.box = kwargs.get('box', np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y,x,Y,X area: (Y-y)*(X-x)
            G.rng = kwargs.get('rng', 1)  # rng of link_ Ls
            G.aRad = kwargs.get('aRad', 0)  # average distance between graph center and node center
            # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
            # G.fback_ = []  # node fb buffer, n in fb[-1]
            G.fi = kwargs.get('fi', 0)  # or fd_: list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
            G.rim = kwargs.get('rim', [])  # external links
            G._N_ = kwargs.get('_N_', [])  # linked nodes
            G.alt_ = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG, empty alt.alt_: select+?
            G.fin = kwargs.get('fin', 0)  # in cluster, temporary?
            G.root = kwargs.get('root')
            G.C_ = kwargs.get('C_', [])  # centroids

        def __bool__(G): return bool(G.node_)  # never empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.L = kwargs.get('L',0)  # distance between nodes
        l.Et = kwargs.get('Et', np.zeros(3))
        l.olp = kwargs.get('olp',1)  # ave nodet overlap
        l.fi = kwargs.get('fi',0)  # nodet fi
        l.yx = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet
        l.box = kwargs.get('box', np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        l.rng = kwargs.get('rng',1) # loop in comp_node_
        l.baseT = kwargs.get('baseT', np.zeros(4))
        l.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ [M,D,n,o, I,G,A,L], sum across derH
        l.derH  = kwargs.get('derH', [])  # list of single-fork CLays
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

    def sum_N_(node_, root_G=None, root=None):  # form cluster G

        fi = isinstance(node_[0], CG);
        lenn = len(node_)
        if root_G is not None:
            G = root_G
        else:
            G = copy_(node_.pop(0), init=1, root=root);
            G.fi = fi
        for n in node_:
            add_N(G, n, fi, fappend=1)
            if root: n.root = root
        if not fi:
            G.derH = [[lay] for lay in G.derH]  # nest
        G._N_ = list(set(G._N_))
        G.olp /= lenn
        return G

    def add_N(N, n, fi=1, fappend=0):

        rn = n.Et[2] / N.Et[2]
        N.baseT += n.baseT * rn;
        N.derTT += n.derTT * rn;
        N.Et += n.Et * rn;
        N.olp += n.olp * rn  # olp is normalized later
        N.yx += n.yx;
        N.box = extend_box(N.box, n.box)  # *rn?
        if isinstance(n, CG):
            N._N_ += n._N_
            if hasattr(n, 'extTT'):  # node, = fi?
                N.extTT += n.extTT;
                N.aRad += n.aRad
                if n.extH: add_H(N.extH, n.extH, root=N, fi=0)
            # if n.alt_: N.alt_ = add_N(N.alt_ if N.alt_ else CG(), n.alt_)  # n.alt_ must be a CG here?
        if fappend:
            N.node_ += [n]
            if isinstance(n, CG): N.C_ += n.C_  # centroids, if any (skip CL)
            if fi: N.link_ += n.link_  # splice if CG
        elif fi:  # empty in append and altG
            if n.H: add_NH(N.H, n.H, root=N)
            if n.Lt: add_NH(N.Lt.H, n.Lt.H, root=N)
        if n.derH:
            add_H(N.derH, n.derH, root=N, fi=fi)
        return N

    def add_node_H(H, h, root):

        for Lev, lev in zip_longest(H, h, fillvalue=None):  # always aligned?
            if lev:
                if Lev:
                    for i, (F, f) in enumerate(zip_longest(Lev, lev, fillvalue=None)):
                        if f:
                            if F:
                                add_N(F, f)  # nG|lG
                            elif F is None:
                                Lev += [f]
                            else:
                                Lev[i] = copy_(f, root=root)  # empty fork
                else:
                    H += [copy_(f, root=root) for f in lev]

def sum2graph(root, node_,link_,llink_,Et,olp, Lay, rng, fi, C_):  # sum node and link params into graph, aggH in agg+ or player in sub+

    n0 = node_[0]
    graph = CG(fi=fi,rng=rng, olp=olp, root=root, Et=Et, node_=node_, link_=link_, box=n0.box, baseT=copy(n0.baseT), derTT=Lay.derTT, derH=[[Lay]], C_=C_)
    graph.llink_ = llink_
    n_,l_,lH,yx_ = [],[],[],[]
    fg = fi and isinstance(n0.node_[0],CG)  # no PPs
    for i,N in enumerate(node_):
        N.root=graph; yx_ += [N.yx]
        if i:
            graph.baseT+=N.baseT; graph.box=extend_box(graph.box,N.box)
            if fg and N.H: add_node_H(graph.H, N.H, root=graph)
        if fg and isinstance(N.node_[0],CG): # skip Ps
            n_ += N.node_; l_ += N.link_
            if N.lH: add_node_H(lH, N.lH, graph)  # top level only
    if fg:  # pack prior top level
        graph.H += [[sum_N_(n_), sum_N_(l_)]]
        graph.lH = lH
    yx = np.mean(yx_, axis=0)
    dy_,dx_ = (graph.yx - yx_).T; dist_ = np.hypot(dy_,dx_)
    graph.aRad = dist_.mean()  # ave distance from graph center to node centers
    graph.yx = yx
    if not fi:  # add mfork as link.nodet(CL).root dfork
        for L in link_:  # higher derH layers are added by feedback, dfork added from comp_link_:
            LR_ = set([n.root for n in L.nodet if isinstance(n.root,CG)]) # skip frame, empty roots
            if LR_:
                dfork = reduce(lambda F,f: F.add_lay(f), L.derH, CLay())  # combine lL.derH
                for LR in LR_:  # lay0 += dfork
                    if len(LR.derH[0])==2: LR.derH[0][1].add_lay(dfork)  # direct root only
                    else:                  LR.derH[0] += [dfork.copy_()]  # init by another node
                    LR.derTT += dfork.derTT
        alt_=[]  # add mGs overlapping dG
        for L in node_:
            for n in L.nodet:  # map root mG
                mG = n.root
                if isinstance(mG, CG) and mG not in alt_:  # root is not frame
                    mG.alt_ += [graph]  # cross-comp|sum complete alt before next agg+ cross-comp, multi-layered?
                    alt_ += [mG]
    return graph

