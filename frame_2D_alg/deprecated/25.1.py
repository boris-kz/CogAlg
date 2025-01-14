def sum_G_(node_):
    G = CG()
    for n in node_:
        G.latuple += n.latuple; G.vert += n.vert; G.aRad += n.aRad; G.box = extend_box(G.box, n.box)
        if n.derH: G.derH.add_tree(n.derH, root=G)
        if n.extH: G.extH.add_tree(n.extH)
    return G

class CH(CBase):  # generic derivation hierarchy of variable nesting: extH | derH, their layers and sub-layers
# rename to CT?
    name = "H"
    def __init__(He, **kwargs):
        super().__init__()
        # move to CG:
        He.H = kwargs.get('H', [])  # list of layers below n_l_t, each is Et summed across fork tree
        He.Et = kwargs.get('Et', np.zeros(4))
        He.m_d_t = kwargs.get('m_d_t',[])  # derivative arrays m_t, d_t
        He.n_l_t = kwargs.get('n_l_t',[])  # element arrays, each element has derT with m_d_t and n_l_t
        He.root = kwargs.get('root')   # N or higher-composition He
        He.node_ = kwargs.get('node_',[])  # concat bottom nesting order if CG, may be redundant to G.node_
        He.altH = CH(altH=object) if kwargs.get('altH', None) is None else kwargs.get('altH')  # sum altLays, None blocks cyclic assign
        # He.i = kwargs.get('i', 0)  # lay index in root.n_l_t, to revise olp
        # He.i_ = kwargs.get('i_',[])  # priority indices to compare node H by m | link H by d
        # He.fd = kwargs.get('fd', 0)  # 0: sum CGs, 1: sum CLs
        # He.ni = 0  # exemplar in node_, trace in both directions?
        # He.deep = kwargs.get('deep',0)  # nesting in root H
        # He.nest = kwargs.get('nest',0)  # nesting in H
    def __bool__(H): return bool(H.m_d_t)  # empty CH

    def copy_(He, root=None, rev=0, fc=0, i=None):  # comp direction may be reversed to -1
    # add H:

        if i:  # reuse self
            C = He; He = i; C.n_l_t = []; C.m_d_t=[]; C.root=root; C.node_=copy(i.node_); C.H =copy(i.H)
        else:  # init new C
            C = CH(root=root, node_=copy(He.node_), H =copy(He.H))
        C.Et = He.Et * -1 if (fc and rev) else copy(He.Et)

        for fd, tt in enumerate(He.m_d_t):  # nested array tuples
            C.m_d_t += [tt * -1 if rev and (fd or fc) else deepcopy(tt)]
        for fork in He.n_l_t:  # empty in bottom layer
            C.n_l_t += [fork.copy_(root=C, rev=rev, fc=fc)]

        if not i: return C

    def add_tree(HE, He_, root=None, rev=0, fc=0):  # rev = dir==-1, unpack derH trees down to numericals and sum/subtract them
        if not isinstance(He_,list): He_ = [He_]

        for He in He_:
            if HE:  # sum m_d_t:
                for fd, (F_t, f_t) in enumerate(zip(HE.m_d_t, He.m_d_t)):  # m_t and d_t
                    for F_,f_ in zip(F_t, f_t):
                        F_ += f_ * -1 if rev and (fd or fc) else f_  # m_| d_ in [dext,dlat,dver]

                for F, f in zip_longest(HE.n_l_t, He.n_l_t, fillvalue=None):  # CH forks
                    if f:  # not bottom layer
                        if F: F.add_tree(f,rev,fc)  # unpack both forks
                        else:
                            f.root=HE; HE.n_l_t += [f]; HE.Et += f.Et
                # H is [Et]
                for L, l in zip_longest(HE.H, He.H, fillvalue=None):
                    if L and l: L += l
                    elif l: HE.H += [copy(l)]
                HE.node_ += [node for node in He.node_ if node not in HE.node_]  # empty in CL derH?
                HE.Et += He.Et * -1 if rev and fc else He.Et
                HE.copy_(root,rev,fc, i=He)
        return HE

    def comp_tree(_He, He, rn, root, dir=1):  # unpack derH trees down to numericals and compare them

        _d_t, d_t = _He.m_d_t[1], He.m_d_t[1]  # comp_m_d_t:
        d_t = d_t * rn  # norm by accum span
        dd_t = (_d_t - d_t * dir)  # np.arrays
        md_t = np.array([np.minimum(_d_,d_) for _d_,d_ in zip(_d_t, d_t)], dtype=object)
        for i, (_d_,d_) in enumerate(zip(_d_t, d_t)):
            md_t[i][(_d_<0) != (d_<0)] *= -1  # negate if only one of _d_ or d_ is negative
        M = sum([sum(md_) for md_ in md_t])
        D = sum([sum(dd_) for dd_ in dd_t])
        n = .3 if len(d_t)==1 else 2.3  # n comp params / 6 (2/ext, 6/Lat, 6/Ver)

        derH = CH(root=root, m_d_t = [np.array(md_t),np.array(dd_t)], Et=np.array([M,D,n, (_He.Et[3]+He.Et[3])/2]))

        for _fork, fork in zip(_He.n_l_t, He.n_l_t):  # comp shared layers
            if _fork and fork:  # same depth
                subH = _fork.comp_tree(fork, rn, root=derH )  # deeper unpack -> comp_md_t
                derH.n_l_t += [subH]  # always fd=0 first
                derH.Et += subH.Et
        # add comp H?
        return derH

    def norm_(He, n):
        for f in He.m_d_t:  # arrays
            f *= n
        for fork in He.n_l_t:  # CHs
            fork.norm_(n)
            fork.Et *= n
        He.Et *= n

def comp_N(_N,N, rn, angle=None, dist=None, dir=1):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = isinstance(N,CL)  # compare links, relative N direction = 1|-1
    # comp externals:
    if fd:
        _L, L = _N.dist, N.dist;  dL = _L - L*rn; mL = min(_L, L*rn) - ave_L  # direct match
        mA,dA = comp_angle(_N.angle, [d*dir *rn for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L- L*rn; mL = min(_L, L*rn) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
    # der ext
    m_t = np.array([mL,mA],dtype=float); d_t = np.array([dL,dA],dtype=float)
    _o,o = _N.Et[3],N.Et[3]; olp = (_o+o) / 2  # inherit from comparands?
    Et = np.array([mL+mA, abs(dL)+abs(dA), .3, olp])  # n = compared vars / 6
    if fd:  # CH
        m_t = np.array([m_t]); d_t = np.array([d_t])  # add nesting
    else:   # CG
        (mLat, dLat), et1 = comp_latuple(_N.latuple, N.latuple, _o,o)
        (mVer, dVer), et2 = comp_md_(_N.vert[1], N.vert[1], dir)
        m_t = np.array([m_t, mLat, mVer], dtype=object)
        d_t = np.array([d_t, dLat, dVer], dtype=object)
        Et += np.array([et1[0]+et2[0], et1[1]+et2[1], 2, 0])
        # same olp?
    Lay = CH(fd_=[fd], m_d_t=[m_t,d_t], Et=Et)
    if _N.derH and N.derH:  # rename to derT?
         derH = _N.derH.comp_tree(N.derH, rn, root=Lay)  # comp shared layers
         Lay.Et += derH.Et; Lay.n_l_t = [derH]
    # spec: comp_node_(node_|link_), combinatorial, node_ nested / rng-)agg+?
    Et = copy(Lay.Et)
    if not fd and _N.altG_ and N.altG_:  # not for CL, eval M?
        # altG_ was converted to altG
        alt_Link = comp_N(_N.altG_, N.altG_, _N.altG_.Et[2] / N.altG_.Et[2])
        Lay.altH = alt_Link.derH
        Et += Lay.altH.Et
    Link = CL(
        fd_=[fd], Et=Et, m_d_t=[m_t,d_t], nodet=[_N,N], derH=Lay, yx=np.add(_N.yx,N.yx)/2, angle=angle,dist=dist,box=extend_box(N.box,_N.box), nest=max(_N.nest,N.nest))
    Lay.root = Link
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            node.extH.add_tree(Link.derH)
            node.Et += Et
    return Link

    # not used:
    def sort_H(He, fd):  # re-assign olp and form priority indices for comp_tree, if selective and aligned

        i_ = []  # priority indices
        for i, lay in enumerate(sorted(He.n_l_t, key=lambda lay: lay.Et[fd], reverse=True)):
            di = lay.i - i  # lay index in H
            lay.olp += di  # derR- valR
            i_ += [lay.i]
        He.i_ = i_  # comp_tree priority indices: node/m | link/d
        if not fd:
            He.root.node_ = He.n_l_t[i_[0]].node_
            # no He.node_ in CL?

def feedback(node):  # propagate node.derH to higher roots

    fd_ = node.fd_; L = len(fd_) - 1
    while node.root:
        root = node.root
        rH = root.derH; iH = node.derH
        add = 1
        for i, fd in enumerate(fd_):  # unpack root H top-down, each fd was assigned by corresponding level of roots
            if len(rH.n_l_t) > fd:  # each fd maps to higher fork n_l_t: 0|1
                # iH is below rH: fd_ maps to rH.H, below n_l_t
                if L-i > len(rH.H): rH.H += [copy(iH.Et)]
                else: rH.H[i] += iH.Et  # fork layer+= fb
                iH = rH; rH = rH.n_l_t[fd]  # root layer is higher than feedback, keep unpacking
            else:
                rH.n_l_t += [iH.copy_()]  # fork was empty, init with He
                add = 0; break

        if add:  # add feedback to fork formed by prior feedback, else append above
            rH.add_tree(iH, root)
        node = root

def cross_comp(root, nest=0):  # breadth-first node_,link_ cross-comp, connect.clustering, recursion

    N_,L_,Et = comp_node_(root.node_)  # cross-comp exemplars, extrapolate to their node_s
    # mfork
    if val_(Et, fo=1) > 0:
        mlay = CH().add_tree([L.derH for L in L_]); H=root.derH; mlay.root=H; H.Et += mlay.Et; H.lft = [mlay]
        pL_ = {l for n in N_ for l,_ in get_rim(n, fd=0)}
        if len(pL_) > ave_L:
            cluster_N_(root, pL_, nest, fd=0)  # nested distance clustering, calls centroid and higher connect.clustering
        # dfork
        if val_(Et, _Et=Et, fo=1) > 0:  # same root for L_, root.link_ was compared in root-forming for alt clustering
            for L in L_:
                L.extH, L.root, L.Et, L.mL_t, L.rimt, L.aRad, L.visited_ = CH(),root,copy(L.derH.Et), [[],[]], [[],[]], 0,[L]
            lN_,lL_,dEt = comp_link_(L_,Et)
            if val_(dEt, _Et=Et, fo=1) > 0:
                dlay = CH().add_tree([L.derH for L in lL_]); dlay.root=H; H.Et += dlay.Et; H.lft += [dlay]
                plL_ = {l for n in lN_ for l,_ in get_rim(n, fd=1)}
                if len(plL_) > ave_L:
                    cluster_N_(root, plL_, nest, fd=1)

        if len(pL_) > ave_L:  # else no higher clusters
            comb_altG_(root)  # combine node altG_(contour) by sum,cross-comp -> CG altG
            # move into cluster_N_:
            cluster_C_(root)  # get (G,altG) exemplars, altG_ may reinforce G by borrowing from extended surround?

def zlast(G):
    if G.derH: return G.derH[-1]
    else:
        m_d_t = [np.array([G.vert[0]]), np.array([G.vert[1]])]  # add nesting for missing md_ext and md_vert?
        # add empty md_ext for [md_ext, md_lat]?
        return CLay(root=G, Et=np.array([G.vert[0].sum(),G.vert[1].sum(),1,1]), node_=G.node_,link_=G.link_, m_d_t=m_d_t)
'''
                            PP = CG(root=edge, fd_=[0], Et=Et, node_=P_, link_=[], vert=vert, latuple=lat, box=box, yx=np.array([y,x]))
                            m_t = np.array([np.zeros(2), vert[0], np.zeros(6)],dtype=object)  # for mostly formal PP.derH:
                            d_t = np.array([np.zeros(2), vert[1], np.zeros(6)],dtype=object)
                            lay0 = CLay(root=PP, m_d_t = [m_t, d_t], Et=copy(Et), node_=P_, link_=[]); PP.derH = [lay0]
'''
def centroid_cluster(N):  # refine and extend cluster with extN_

    # add proximity bias, for both match and overlap?
    _N_ = {n for L, _ in N.rim for n in L.nodet if not n.fin}
    N.fin = 1;
    n_ = _N_ | {N}  # include seed node
    C = centroid(n_, n_)
    while True:
        N_, dN_, extN_, M, dM, extM = [], [], [], 0, 0, 0  # included, changed, queued nodes and values
        med = 0  # extN_ is mediated by <=3 _Ns, loop all / refine cycle:
        while med < 3:
            medN_ = []
            for _N in _N_:
                if _N in N_ or _N in extN_: continue  # skip meidated Ns
                m = comp_C(C, _N)  # Et if proximity-weighted overlap?
                vm = m - ave  # deviation
                if vm > 0:
                    N_ += [_N];
                    M += m
                    if _N.m:
                        dM += m - _N.m  # was in C.node_, add adjustment
                    else:
                        dN_ += [_N]; dM += vm  # new node
                    _N.m = m  # to sum in C
                    for link, _ in _N.rim:
                        n = link.nodet[0] if link.nodet[1] is _N else link.nodet[1]
                        if n.fin or n.m: continue  # in other C or in C.node_
                        extN_ += [n];
                        extM += n.Et[0]  # external N for next loop
                        medN_ += [n]  # for mediation loop
                elif _N.m:  # was in C.node_, subtract from C
                    _N.sign = -1;
                    _N.m = 0;
                    dN_ += [_N];
                    dM += -vm  # dM += abs m deviation
            med += 1;
            _N_ = medN_
        if dM > ave and M + extM > ave:  # update for next loop, terminate if low reform val
            if dN_:  # recompute C if any changes in node_
                C = centroid(set(dN_), N_, C)
            _N_ = set(N_) | set(extN_)  # next loop compares both old and new nodes to new C
            C.M = M;
            C.node_ = N_
        else:
            if C.M > ave * 10:  # add proximity-weighted overlap
                C.root = N.root  # C.nest = N.nest+1
                for n in C.node_:
                    n.root = C;
                    n.fin = 1;
                    delattr(n, "sign")
                return C  # centroid cluster
            else:  # unpack C.node_
                for n in C.node_: n.m = 0
                N.nest += 1
                return N  # keep seed node
            # break

# get representative centroids of complemented Gs: mCore + dContour, initially in unpacked edges
N_ = sorted([N for N in graph.node_ if any(N.Et)], key=lambda n: n.Et[0], reverse=True)
G_ = []
for N in N_:
    N.sign, N.m, N.fin = 1, 0, 0  # setattr C update sign, inclusion val, C inclusion flag
for i, N in enumerate(N_):  # replace some of connectivity cluster by exemplar centroids
    if not N.fin:  # not in prior C
        if val_(N.Et, coef=10) > 0:
            G_ += [centroid_cluster(N)]  # extend from N.rim, return C if packed else N
        else:  # the rest of N_ M is lower
            G_ += [N for N in N_[i:] if not N.fin]
            break
graph.node_ = G_  # mix of Ns and Cs: exemplars of their network?
if len(G_) > ave_L:
    cross_comp(graph)
    # selective connectivity clustering between exemplars, extrapolated to their node_
