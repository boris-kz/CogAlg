def comp_N(_N,N, ave, angle=None, span=None, dir=1, fdeep=0, fproj=0, rng=1):  # compare links, relative N direction = 1|-1, no angle,span?

    derTT, Et, rn = base_comp(_N, N, dir); fi = N.fi
    # link M,D,A:
    baseT = np.array([*(_N.baseT[:2] + N.baseT[:2]*rn) / 2, *angle])  # redundant angle for generic base_comp, also span-> density?
    _y,_x = _N.yx; y,x = N.yx; box = np.array([min(_y,y),min(_x,x),max(_y,y),max(_x,x)])
    o = (_N.olp+N.olp) / 2
    Link = CN(rng=rng, olp=o, N_=[_N,N], baseT=baseT, derTT=derTT, yx=np.add(_N.yx,N.yx)/2, span=span, angle=angle, box=box, fi=0)
    Link.derH = [CLay(root=Link, Et=Et, node_=[_N,N],link_=[Link], derTT=copy(derTT))]
    # contour:
    if fi and (_N.alt and N.alt) and val_(_N.alt.Et+N.alt.Et, aw=2+o, fi=0) > 0:  # eval Ds
        Link.altL = comp_N(L2N(_N.alt), L2N(N.alt), ave*2, angle,span); Et += Link.altL.Et
    # comp derH:
    if fdeep and (val_(Et, len(N.derH)-2, o) > 0 or fi==0):  # else derH is dext,vert
        ddH = comp_H(_N.derH, N.derH, rn, Link, derTT, Et)
        if fproj and val_(Et, len(ddH)-2, o) > 0:
            M,D,_ = Et; dec = M / (M+D)  # final Et
            comp_prj_dH(_N,N, ddH, rn, Link, angle, span, dec)  # surprise = comp combined-N prj_dH to actual ddH
        Link.derH += ddH  # flat
    Link.Et = Et
    if Et[0] > ave * Et[2]:
        for rev, node, _node in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
            # simplified sum_N_:
            if fi: node.rim.L_ += [(Link,rev)]
            else:  node.rim.L_[1-rev] += [(Link,rev)]  # rimt opposite to _N,N dir
            node.rim.Et += Et; node.rim.N_ += [_node]; node.rim.baseT += baseT
            node.rim.derTT += derTT  # simplified add_H(node.rim.derH, Link.derH, root=node, rev=rev)?

    return Link

def norm_(n, L):  # not needed, comp is normalized anyway?

    if n.rim: norm_(n.rim, L)
    if n.alt: norm_(n.alt, L)
    for val in n.Et, n.baseT, n.derTT, n.span, n.yx: val /= L
    for lay in n.derH:
        lay.derTT /= L; lay.Et /= L

def cross_comp(root, rc, fi=1):  # cross-comp, clustering, recursion

    N__,L_,Et = comp_node_(root.N_,rc) if fi else comp_link_(root.N_,rc)  # rc: redundancy+olp
    if N__:
        mV,dV = val_(Et, (len(L_)-1)*Lw, rc+loop_w, fi=2)
        if dV > 0:    # dfork is still G-external but nodet-mediated?
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_=L_; root.Et += Et
            if dV >avd: Lt = cross_comp(CN(N_=L2N(L_)), rc+clust_w, fi=0)  # -> Lt.nH, +2 der layers
            else:       Lt = cluster_N_(L_, rc+clust_w, fi=0, fL=1)  # low-res cluster / nodet, +1 layer
            if Lt:      root.lH += [Lt] + Lt.nH; root.Et += Lt.Et; root.derH += Lt.derH  # append new der lays
        # m cluster, mm,md xcomp
        if fi: n__ = {N for N_ in N__ for N in N_}
        else:  n__ = N__; N__ = [N__]
        for n in n__: n.sel=0
        if mV > 0:
            Ct,Nt, rng = [],[], 1
            if fi:
                E_, eEt = get_exemplars(root, n__, rc+loop_w)  # point cloud of focal nodes
                if isinstance(E_,CN):
                    Ct = E_  # from cluster_C_) _NC_, C-internal
                elif E_ and val_(eEt, (len(E_)-1)*Lw, rc+clust_w) > 0:
                    for rng, N_ in enumerate(N__, start=1):  # bottom-up rng incr
                        rE_ = [n for n in N_ if n.sel]       # cluster via rng exemplars
                        if rE_ and val_(np.sum([n.Et for n in rE_], axis=0), (len(rE_)-1)*Lw, rc) > 0:
                            Nt = cluster_N_(rE_,rc,1,rng) or Nt  # keep top-rng Gt
            else:  # trace all arg links via llinks
                Nt = cluster_N_(n__, rc,0)
            if Nt:
                if val_(Nt.Et, (len(Nt.N_)-1)*Lw, rc+loop_w, _Et=Et) > 0:
                    Nt = cross_comp(Nt, rc+clust_w*rng) or Nt
                    # agg+
            if Nt := Ct or Nt:
                _H = root.nH; root.nH = []
                Nt.nH = _H + [root] + Nt.nH  # pack root in Nt.nH, has own L_,lH
                # recursive feedback:
                return Nt

def get_rim(N):  # test N.rim.N_ fi for further nodet forking?
    return N.rim.L_ if N.fi else N.rim.L_[0] + N.rim.L_[1]

def lolp(N, L_, fR=0):  # relative N.rim or R.link_ olp eval for clustering
    oL_ = [L for L in (N.L_ if fR else [l for l,_ in flatten(N.rim.L_)]) if L in L_]
    if oL_:
        oEt = np.sum([l.Et for l in oL_], axis=0)
        _Et = N.Et if fR else N.rim.Et
        rM  = (oEt[0]/oEt[2]) / (_Et[0]/_Et[2])
        return val_(_Et * rM, contw)
    else: return 0

def cluster(root, N_, rc, fi):
    Nf_ = list(set([N for n_ in N_ for N in n_])) if fi else N_

    E_, Et = get_exemplars(root, Nf_, rc, fi)
    if val_(Et, (len(E_)-1)*Lw, rc+centw) > 0:  # replace exemplars with centroids
        for n in root.N_: n.C_, n.vo_, n._C_, n._vo_ = [], [], [], []
        Ct = cluster_C_(root, E_, rc+centw, fi)
        if Ct:
            return Ct
        elif val_(Et, (len(E_)-1)*Lw, rc+contw, fi=fi) > 0:
            for n in Nf_: n.sel=0
            if fi:
                Nt = []  # rng-banded clustering:
                for rng, rN_ in enumerate(N_, start=1):  # bottom-up rng incr
                    rE_ = [n for n in rN_ if n.sel]      # cluster via rng exemplars
                    if rE_ and val_(np.sum([n.Et for n in rE_], axis=0), (len(rE_)-1)*Lw, rc) > 0:
                        Nt = cluster_N_(root, rE_,rc,rng) or Nt  # keep top-rng Gt
            else:
                Nt = cluster_L_(root, N_, E_, rc)
            return Nt

def C_merge(N,_N, root, rc, fi):  # eval _N.c_ merge in N.c_, may overlap
    '''
    N is incomplete connectivity cluster with sub-centroids N.c_: internally coherent sub-clusters of c.N_, each a subset of N.N_.
    '''
    ln = len(N.c_)
    if ln:
        for c in N.c_:
            if _N.c_:
                for _c in _N.c_:  # all unique
                    dy, dx = np.subtract(_N.yx, N.yx)
                    link = comp_N(_c,c, ave, angle=np.array([dy,dx]),span=np.hypot(dy,dx))
                    if val_(link.Et, aw=rc+contw) > 0:
                        _c.N_ += c.N_; _c.Et += c.Et  # no overlap?
                    # else Ns are different but may be connected?
            if len(c.N_) / ln > arn:
                C_form(_N, root, rc, fi)  # for N.C only, also merging N, multiple: N.c_?

# draft
def C_form(_C, root, rc, fi=0):

    C = sum_N_(_C.N_, root=root, fC=1)  # merge rim,alt before cluster_NC_
    N_,_N__,o,Et,dvo = [],[],0, np.zeros(3),np.zeros(2)  # per C
    for n in _C._N_:  # core + surround
        if C in n.C_: continue  # clear/ loop?
        et = comp_C(C,n); v = val_(et,aw=rc,fi=fi)
        olp = np.sum([vo[0] / v for vo in n.vo_ if vo[0] > v])  # olp: rel n val in stronger Cs, not summed with C
        vo = np.array([v,olp])  # val, overlap per current root C
        if et[1-fi]/et[2] > ave*olp:
            n.C_ += [C]; n.vo_ += [vo]; Et += et; o+=olp  # _N_ += [n]?
            _N__ += flat(n.rim.N_ if fi else n.rim.L_)
            if C not in n._C_: dvo += vo
        elif C in n._C_:
            dvo += n._vo_[n._C_.index(C)]  # old vo_, or pack in _C_?

def cluster_NC_(_C_, rc):  # cluster centroids if pairwise Et + overlap Et

    C_,L_ = [],[]; Et = np.zeros(3)
    # form links:
    for _C,C in combinations(_C_,2):
        oN_ = list(set(C.N_) & set(_C.N_))  # exclude from comp?
        dy,dx = np.subtract(_C.yx,C.yx)
        link = comp_N(_C, C, ave, angle=np.array([dy,dx]), span=np.hypot(dy,dx))
        if val_(link.Et, aw=rc+contw) > 0:
            _C.N_ += C.N_  # refine _C if len(C.N_) / len(_C.N_) > ave?
            continue  # else Cs are different but may be connected:
        oV = val_(np.sum([n.Et for n in oN_],axis=0), rc)  # use rolp?
        _V = min(val_(C.Et, rc), val_(_C.Et, rc), 1e-7)
        link.Et[:2] *= intw
        link.Et[0] += oV/_V - arn  # or separate connect | centroid eval, re-centering?
        if val_(link.Et, rc) > 0:
            C_ += [_C,C]; L_+= [link]; Et += link.Et
    C_ = list(set(C_))
    if val_(Et, (len(C_)-1)*Lw, rc) > 0:
        G_ = []
        for C in C_:
            if C.fin: continue
            G = copy_(C, init=1); C.fin = 1
            for _C, Lt in zip(C.rim.N_,C.rim.L_):
                G.L_+= [Lt[0]]
                if _C.fin and _C.root is not G:
                    add_N(G,_C.root, fmerge=1)
                else: add_N(G,_C); _C.fin=1
            G_ += [G]
        Gt = sum_N_(G_); Gt.L_ = L_
        if val_(Et, (len(G_)-1)*Lw, rc+1) > 0:  # new len G_, aw
            Gt = cluster_NC_(G_, rc+1) or Gt  # agg recursion, also return top L_?
        return Gt
'''
    if V > 0:
        Nt = sum_N_(E_)
        if V * ((len(C_)-1)*Lw) > av*contw: Nt = cluster_NC_(C_, O+rc+contw) or Nt
'''
def cluster_N_(root, E_, rc, rng=1):  # connectivity cluster exemplar nodes via rim or links via nodet or rimt

    G_ = []  # flood-fill Gs, exclusive per fork,ave, only centroid-based can be fuzzy
    for n in root.N_: n.fin = 0
    for N in E_:
        if N.fin: continue
        # init N cluster with rim | root:
        if rng==1 or N.root.rng==1: # N is not rng-nested
            node_,link_,llink_,Et, olp = [N],[],[], N.Et+N.rim.Et, N.olp
            for l,_ in flat(N.rim.L_):  # +ve
                if l.rng==rng and val_(l.Et,aw=rc+contw)>0: link_ += [l]
                elif l.rng>rng: llink_ += [l]  # longer-rng rim
        else:
            n = N; R = n.root  # N.rng=1, R.rng > 1, cluster top-rng roots instead
            while R.root and R.root.rng > n.rng: n = R; R = R.root
            if R.fin: continue
            node_,link_,llink_,Et, olp = [R],R.L_,R.hL_,copy(R.Et),R.olp
            R.fin = 1
        nrc = rc+olp; N.fin = 1  # extend N cluster:
        for L in link_[:]:  # cluster nodes via links
            for _N in L.N_:
                if _N not in root.N_ or _N.fin: continue  # connectivity clusters don't overlap
                if rng==1 or _N.root.rng==1:  # not rng-nested
                    if rolp(N, [l for l,_ in flat(N.rim.L_)], fi=1) > ave*nrc:
                        node_ +=[_N]; Et += _N.Et+_N.rim.Et; olp+=_N.olp; _N.fin=1
                        for l,_ in flat(N.rim.L_):
                            if l not in link_ and l.rng == rng: link_ += [l]
                            elif l not in llink_ and l.rng>rng: llink_+= [l]  # longer-rng rim
                else:
                    _n =_N; _R=_n.root  # _N.rng=1, _R.rng > 1, cluster top-rng roots if rim intersect:
                    while _R.root and _R.root.rng > _n.rng: _n=_R; _R=_R.root
                    if not _R.fin and rolp(N,link_,fi=1, R=1) > ave*nrc:
                        # cluster by N_|L_ connectivity: oN_ = list(set(N.N_) & set(_N.N_))  # exclude from comp and merge?
                        link_ = list(set(link_+_R.link_)); llink_ = list(set(llink_+_R.hL_))
                        node_+= [_R]; Et +=_R.Et; olp += _R.olp; _R.fin = 1; _N.fin = 1
                nrc = rc+olp
        node_ = list(set(node_))
        alt = [L.root for L in link_ if L.root]  # get link clusters, individual rims are too weak for contour
        if alt:
            alt = sum_N_(list(set(alt)))
            if val_(alt.Et, (len(alt.N_)-1)*Lw, olp, _Et=Et, fi=0):
                alt = cross_comp(alt, olp) or alt
            _Et = alt.Et
        else: _Et = np.zeros(3)
        V = val_(Et, (len(node_)-1)*Lw, nrc, _Et)
        if V > 0:
            if V > ave*centw: Ct = cluster_C_(root, node_, rc+centw)  # form N.c_, doesn't affect clustering?
            else:             Ct = []   # option to keep as list, same for alt and rim?
            G_ += [sum2graph(root, node_, link_, llink_, Et, olp, rng, Ct, alt)]
        if G_:
            return sum_N_(G_, root)   # root N_|L_ replacement

def cluster_L_(root, N_, E_, rc):  # connectivity cluster links from exemplar node.rims

    G_ = []  # cluster by exemplar rim overlap
    fi = not E_[0].rim  # empty in base L_
    for n in N_: n.fin = 0
    for E in E_:
        if E.fin: continue
        node_,Et, olp = [E], E.rim.Et, E.olp
        link_ = [l for l,_ in flat(E.rim.L_) if l.Et[1] > avd*l.Et[2]*rc]  # L val is always diff?
        nrc = rc+olp; E.fin = 1  # extend N cluster:
        if fi:  # cluster Ls by L diff
            for _N in E.rim.N_:  # links here
                if _N.fin: continue
                _N.fin = 1
                for n in _N.N_:  # L.nodet
                    for L,_ in flat(n.rim.L_):
                        if L not in node_ and _N.Et[1] > avd * _N.Et[2] * nrc:  # diff
                            link_+= [_N]; node_+= [L]; Et += L.Et; olp+= L.olp  # /= len node_
        else:  # cluster Ls by rimt match
            for L in link_[:]: # snapshot
                for _N in L.N_:
                    if _N in N_ and not _N.fin and rolp(E, [l for l,_ in flat(_N.rim.L_)], fi=0) > ave * nrc:
                        node_ +=[_N]; Et += _N.Et+_N.rim.Et; olp+=_N.olp; _N.fin=1
                        for l,_ in flat(_N.rim.L_):
                            if l not in link_: link_ += [l]
        node_ = list(set(node_))
        V = val_(Et, (len(node_)-1)*Lw, rc+olp, _Et=root.Et, fi=fi)
        if V > 0:
            if V > ave*centw: Ct = cluster_C_(root, node_, rc+centw)  # form N.c_, doesn't affect clustering?
            else:             Ct = []   # option to keep as list, same for alt and rim?
            G_ += [sum2graph(root, node_, link_,[], Et, olp, 1, Ct)]
        if G_:
            return sum_N_(G_,root)
