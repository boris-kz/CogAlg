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


