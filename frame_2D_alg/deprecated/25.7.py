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
            if fi: node.rim.L_ += [(Link,rev)]; node.rim.N_ += [_node]
            else:  node.rim.L_[1-rev] += [(Link,rev)]; node.rim.N_[1-rev] += [_node]
            # add to rimt opposite to _N,N dir
            node.rim.Et += Et; node.rim.N_ += [_node]; node.rim.baseT += baseT
            node.rim.derTT += derTT  # simplified add_H(node.rim.derH, Link.derH, root=node, rev=rev)?

    for rev, n,_n in zip((0,1),(N,_N),(_N,N)):  # reverse Link dir in _N.rimt
        fn = isinstance(n.rim, CN)  # else rim is listt, simplified add_N_(rim), +|-:
        if fi:
            if fn: n.rim.L_ += [(Link,rev)]; n.rim.N_ += [_n]
            else:  n.rim[0] += [(Link,rev)]; n.rim[1] += [_n]
        else:
            if n.N_: n = CN(root=n, rim=[Link]); Link.rim = [n,_n]  # in der+: replace N with blank CN pseudo-nodet, keep N as root?
            if fn: n.rim.L_[1-rev] += [(Link,rev)]; n.rim.N_[1-rev] += [_n]  # rimt opposite to _N,N dir
            else:  n.rim[1-rev][0] += [(Link,rev)]; n.rim[1-rev][1] += [_n]
        # else:  n.rim[1 - rev]._rim += [(Link, rev, _n)]  # if n is link, nodet-mediated rim opposite to _N,N dir
        if fn:
            n.rim.Et += Et; n.rim.baseT += baseT; n.rim.derTT += derTT  # simplified add_H(node.rim.derH, Link.derH, root=node, rev=rev)?

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

def cluster0(root, N_, rc, fi):
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

    G_ = []
    for n in root.N_: n.fin = 0
    for N in E_:
        if N.fin: continue
        # init N cluster with rim | root:
        if rng==1 or N.root.rng==1:
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

def rim_(N, fi=None):  # unpack terminal rt_s, or med rt_s?
    Rim = []
    for r in (N if isinstance(N,list) else N.rim):  # n if link, [] if nested n.rim, else rt: (l,rev,_n)
        if r not in Rim:
            if isinstance(r,tuple):  # final rim: l,rev,_n
                Rim += [r if fi is None else r[2] if fi else r[0]]  # if val_(fi): always selective?
            elif isinstance(r, CN):
                Rim.extend(rim_(r,fi))  # r is nodet[i], may be nested
            else:
                Rim.extend(rim_(N.rim[-1],fi))  # get top layer | flatten all?
                break
    return Rim

def comp_link_(iL_, rc):  # comp CLs via directional node-mediated link tracing: der+'rng+ in root.link_ rim_t node rims

    for L in iL_:  # init mL_t: nodet-mediated Ls:
        for rev, N, mL_ in zip((0, 1), L.N_, L.mL_t):  # L.mL_t is empty
            for _L,_rev in rim_(N,0):
                if _L is not L and _L in iL_:
                    if val_(L.Et,0,aw=loopw) > 0:
                        mL_ += [(_L, rev ^ _rev)]  # direction of L relative to _L
    med = 1; _L_ = iL_
    L__,LL_,ET = [],[],np.zeros(3)
    while True:  # xcomp _L_
        L_, Et = [], np.zeros(3)
        for L in _L_:
            for mL_ in L.mL_t:
                for _L, rev in mL_:  # rev is relative to L
                    if _L in L.compared_: continue
                    dy,dx = np.subtract(_L.yx,L.yx)
                    Link = comp_N(_L,L, rc, np.array([dy,dx]), np.hypot(dy,dx), -1 if rev else 1)  # d = -d if L is reversed relative to _L
                    Link.med = med
                    LL_ += [Link]; Et += Link.Et  # include -ves, link order: nodet < L < rim, mN.rim || L
                    for l,_l in zip((_L,L),(L,_L)):
                        l.compared_ += [_l]
                        if l not in L_ and val_(l.et, aw=rc+med+loopw) > 0:  # cost+/ rng
                            L_ += [l]  # for med+ and exemplar eval
        if L_: L__ += L_; ET += Et
        # extend mL_t per last medL:
        if val_(Et,aw=rc+loopw+med*medw) > 0:  # project prior-loop value, med adds fixed cost
            _L_, ext_Et = set(), np.zeros(3)
            for L in L_:
                mL_t, lEt = [set(),set()], np.zeros(3)  # __Ls per L
                for mL_,_mL_ in zip(mL_t, L.mL_t):
                    for _L, rev in _mL_:
                        for _rev, N in zip((0,1), _L.N_):
                            rim = rim_(N,0)
                            if len(rim) == med:  # rim should not be nested?
                                for __L,__rev in rim:
                                    if __L in L.visited_ or __L not in iL_: continue
                                    L.visited_ += [__L]; __L.visited_ += [L]
                                    if val_(__L.Et,aw=loopw) > 0:
                                        mL_.add((__L, rev ^_rev ^__rev))  # combine reversals: 2 * 2 mLs, but 1st 2 are pre-combined
                                        lEt += __L.Et
                if lEt[0] > ave * lEt[2]:  # L rng+, vs. L comp above, add coef
                    L.mL_t = mL_t; _L_.add(L); ext_Et += lEt
            # refine eval by extension D:
            if val_(ext_Et, aw=rc+loopw+med*medw) > 0: med += 1
            else: break
        else: break
    return list(set(L__)), LL_, ET

def rim_1(N, fi=None):  # get max-med [(L,rev,_N)], rev: L dir relative to N

    if N.fi:                      rt_ = N.rim
    elif isinstance(N.rim[0],CN): rt_ = N.rim[0].rim+N.rim[1].rim  # L.nodet
    else:                         rt_ = N.rim[-1]  # med-nested L.rim
    return [r if fi is None else r[2] if fi else r[0] for r in rt_]

def L2N(link_):
    for L in link_: L.mL_t = [[],[]]; L.compared_,L.visited_ = [],[]; L.rim[0].med = 0; L.rim[1].med = 0; L._rim = [CN(rim=[[],[]]), CN(rim=[[],[]])]
    return link_

def comp_spec(_spec,spec, rc, LEt,Lspec, flist):

    if isinstance(_spec,CN) and isinstance(spec,CN):  # CN or list, may be CLay if include derH
        dspec = comp_N(_spec, spec, ave)
        Lspec += [dspec]; LEt += dspec.Et
        return
    elif flist:
        dspec_,dEt = [], np.zeros(3)  # dEt is not used here?
        for _e in _spec if isinstance(_spec,list) else _spec.N_:
            for e in spec if isinstance(spec,list) else spec.N_:
                if _e is e: LEt += e.Et  # not sure
                else: dspec_ += [comp_N(_e,e,rc)]  # comp_spec if e can be list?
        if len(dspec_) * Lw > ave:
            Dspec = sum_N_(dspec_)
            if isinstance(Lspec,list): Lspec = sum_N_(Lspec)
            add_N(Lspec,Dspec); LEt+=Dspec.Et
        else:
            for de in dspec_:
                Lspec += [de] if isinstance(Lspec,list) else add_N(Lspec,de)
                LEt += de.Et


def cluster(root, iN_, E_, rc, fi, rng=1):  # flood-fill node | link clusters

    G_ = []  # exclusive per fork,ave, only centroids can be fuzzy
    fln = isinstance(E_[0].rim[0], tuple)  # flat rim
    for n in (root.N_ if fi else iN_):
        n.fin = 0
    for E in E_:  # init
        if E.fin: continue
        E.fin = 1; seen_ = set()  # links
        if fi: # node clustering
            if rng==1 or E.root.rng==1:  # E is not rng-banded
                N_,link_,long_ = [E],[],[]
                for l in rim_(E,0):  # lrim
                    if l.rng==rng and val_(l.Et,aw=rc) > 0: link_+=[l]  # or select links here?
                    elif l.rng>rng: long_+=[l]
            else: # E is rng-banded, cluster top-rng roots
                n = E; R = n.root
                while R.root and R.root.rng > n.rng: n = R; R = R.root
                if R.fin: continue
                N_,link_,long_ = [R], R.L_, R.hL_; R.fin = 1
        else: # link clustering
            N_,long_ = [],[]; link_ = rim_(E, 1 if fln else 0)  # nodet or lrim
        L_ = []
        while link_:  # extend cluster N_
            L_ += link_; _link_ = link_
            link_ = set()
            for L in _link_:  # buffer
                if fi: # cluster nodes via rng (density) - specific links
                    for _N in L.rim if fln else L.rim[-1]:
                        if _N not in iN_ or _N.fin: continue
                        if rng==1 or _N.root.rng==1:  # not rng-banded
                            if rolp(E, set(rim_(E,0)), fi=1) > ave*rc:  # N_,L_ += full-rim connected _N,_L
                                N_ += [_N]; _N.fin = 1  # conditional
                                for l in rim_(_N,0):
                                    if l in seen_: continue
                                    if l.rng==rng and val_(l.Et, aw=rc) > 0: link_.add(l); seen_.add(l)
                                    elif l.rng>rng: long_ += [l]
                        else:  # cluster top-rng roots
                            _n = _N; _R = _n.root
                            while _R.root and _R.root.rng > _n.rng: _n = _R; _R = _R.root
                            if not _R.fin and rolp(E, link_, fi=1, R=1) > ave*rc:
                                N_ += [_R]; _R.fin = 1; _N.fin = 1
                                link_.update(set(_R.link_)-seen_); seen_.update(_R.link_); long_+=_R.hL_
                else:  # cluster links via rim
                    if fln:  # by L diff
                       for n in L.rim:  # in nodet, no eval
                            if n in iN_ and not n.fin and rolp(E, set(rim_(n,0)), fi) > ave*rc:
                                N_ += [l for l in rim_(n,0) if val_(l.Et,0,aw=rc) > 0]; n.fin = 1
                                if n not in seen_: link_.add(n); seen_.add(n)
                    else:  # by LL match
                        for _L in rim_(L,1):  # via E.rim if E.fi else E.rim[-1]
                            if _L.fin: continue
                            for LL in rim_(_L,0):
                                if val_(_L.Et,0,aw=rc) > 0:
                                    N_ += [_L]; _L.fin = 1
                                    if LL not in seen_: link_.add(LL); seen_.add(LL)
        if N_:
            N_,L_,long_ = list(set(N_)), list(L_), list(set(long_))
            Et, olp = np.zeros(3),0  # sum node_:
            for n in N_:
                Et += n.et; olp += n.olp
                # any fork
            if fi: altg_ = {L.root for L in L_ if L.root}  # contour if fi else cores, individual rims are too weak
            else:  altg_ = {n for N in N_ for n in (N.rim if fln else N.rim[0])}  # N_ is Ls, assign nodets
            _Et  = np.sum([i.Et for i in altg_], axis=0) if altg_ else np.zeros(3)

            if val_(Et,1, (len(N_)-1)*Lw, rc+olp, _Et) > 0:
                G_ += [sum2graph(root, E_,N_,L_,long_, Et, olp, rng, ([altg_,_Et] if fi else altg_) if altg_ else [])]
    if G_:
        return sum_N_(G_, root)

def rim_1(N, fi=None):  # get max-med [(L,rev,_N)], rev: L dir relative to N
    if N.fi:
        rt_ = N.rim
    elif isinstance(N.rim[0], list):
        rt_ = N.rim[-1]  # max-med layer in nested L.rim
    else:
        n_ = N.rim  # flat L.rim = nodet
        while isinstance(n_[0], CN) and not n_[0].fi:  # unpack n_: terminal L.[n,_n] tree branches
           n_ = [n for L in n_ for n in L.rim]  # L.rims stay flat
        rt_ = [rt for n in n_ for rt in n.rim]  # n.fi = 1
    return [r if fi is None else r[2] if fi else r[0] for r in rt_]

def Cluster(root, N_, rc, fi):  # clustering root

    Nflat_ = list(set([N for n_ in N_ for N in n_])) if fi else N_
    if fi:
        for n in Nflat_: n.sel=0
        E_, Et = get_exemplars(Nflat_, rc, fi)
    else: E_, Et = N_, root.Et  # fi=0?

    if val_(Et,fi, (len(Nflat_)-1)*Lw, rc+contw, root.Et) > 0:
        if fi:
            nG = []  # bottom-up rng-banded clustering:
            for rng, rN_ in enumerate(N_, start=1):
                rE_ = [n for n in rN_ if n.sel]
                aw = rc * rng + contw  # cluster Nf_ via rng exemplars:
                if rE_ and val_(np.sum([n.Et for n in rE_], axis=0),1,(len(rE_)-1)*Lw, aw) > 0:
                    nG = cluster(root, Nflat_, rE_, aw, 1, rng) or nG  # keep top-rng Gt
        else:
            nG = cluster(root, N_, E_, rc, fi=0)
        return nG

def rim_2(N, fi=None, fln=0):

    if N.fi: rt_ = N.rim
    # base rim = [(L,rev,_N)], rev: L dir to N
    elif isinstance(N.rim[0],list) and not fln:
        rt_ = [rt for lay in N.rim[1:] for rt in lay]  # all L.rim layers above nodet
    else:  # get nodet leaves
        n_ = N.rim if isinstance(N.rim[0],CN) else N.rim[0]
        return list(set([_n for n in n_ for _n in rim_(n, fi, fln=1)]))  # nodet tree leaves

    return [r if fi is None else r[2] if fi else r[0] for r in rt_]


def cross_comp1(root, rc, fi=1):  # rng+ and der+ cross-comp and clustering

    N_,L_,Et = comp_node_(root.N_,rc) if fi else comp_link_(root.N_,rc)  # rc: redundancy+olp, lG.N_ is Ls
    if len(L_) > 1:
        for n in [n for N in N_ for n in N] if fi else N_:
            for l in rim_(n,0): n.et+=l.Et  # ext olp
        mV,dV = val_(Et,2, (len(L_)-1)*Lw, rc+loopw)
        if dV > 0:
            if root.L_: root.lH += [sum_N_(root.L_)]  # replace L_ with agg+ L_:
            root.L_=L_; root.Et += Et
            if dV >avd: lG = cross_comp(CN(N_=L_), rc+contw, fi=0)  # -> Lt.nH, +2 der layers
            else:       lG = Cluster(root, L_, rc+contw, fi=0)  # cluster N.rim Ls, +1 layer
            if lG: root.lH+= [lG]+lG.nH; root.Et+=lG.Et; root.derH+=lG.derH  # new der lays
        if mV > 0:
            nG = Cluster(root, N_, rc+loopw, fi)   # get_exemplars, cluster_C_, rng-banded if fi
            if nG and val_(nG.Et,1, (len(nG.N_)-1)*Lw, rc+loopw+nG.rng, Et) > 0:
                nG = cross_comp(nG, rc+loopw) or nG  # agg+
            if nG:
                _H = root.nH; root.nH = []
                nG.nH = _H + [root] + nG.nH  # pack root in Nt.nH, has own L_,lH
                return nG  # recursive feedback

def comp_link_1(iL_, rc):  # comp links via directional node-mediated _link tracing with incr mediation

    for L in iL_:
        L._rim = []; L.rim = [L.rim, L.rim[0].rim+L.rim[1].rim]  # nest nodet, rim
    L__,LL_,ET,_L_ = [],[], np.zeros(3), iL_
    med = 1
    while _L_ and med < 4:
        L_, Et = [], np.zeros(3)
        for L in _L_:
            for _L,rev,_ in L.rim[-1]:  # top-med Ls, _rev^rev in comp_N
                if _L not in L.compared_ and _L in iL_ and val_(_L.Et,0, aw=rc+med) > 0:  # high-d, new links can't be outside iL_
                    dy, dx = np.subtract(_L.yx, L.yx)
                    Link = comp_N(_L,L, rc, med, LL_, np.array([dy,dx]), np.hypot(dy,dx), rev)  # d = -d if L is reversed relative to _L
                    if val_(Link.Et,1, aw=rc+med) > 0:  # recycle Ls if high m
                        L_ += [L,_L]; Et += Link.Et
        for L in _L_:
            if L._rim: L.rim += [L._rim]; L._rim = []  # merge new rim per med loop
        if L_:
            L_ = list(set(L_)); L__ += L_; _L_ = []
            for l in L_:
                if l not in _L_ and len(l.rim) > med and val_(l.Et,0, aw=rc+med) > 0:
                    _L_ += [l]  # high-d, rim extended in loop
        med += 1
    return list(set(L__)), LL_, ET
