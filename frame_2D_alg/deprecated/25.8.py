def comp_slice(edge, rV=1, ww_t=None):  # root function

    global ave, avd, wM, wD, wI, wG, wA, wL, ave_L, ave_PPm, ave_PPd, w_t
    ave, avd, ave_L, ave_PPm, ave_PPd = np.array([ave, avd, ave_L, ave_PPm, ave_PPd]) / rV  # projected value change
    if np.any(ww_t):
        w_t = [[wM, wD, wI, wG, wA, wL]] * ww_t
        # der weights
    for P in edge.P_:  # add higher links
        P.vertuple = np.zeros((2,6))
        P.rim = []; P.lrim = []; P.prim = []
    edge.dP_ = []
    comp_P_(edge)  # vertical P cross-comp -> PP clustering, if lateral overlap
    PPt, mvert, mEt = form_PP_(edge.P_, fd=0)  # all Ps are converted to PPs
    if PPt:
        edge.node_ = PPt
        comp_dP_(edge, mEt)
        edge.link_, dvert, dEt = form_PP_(edge.dP_, fd=1)
        edge.vert = mvert + dvert
        edge.Et = Et = mEt + dEt
    return PPt

def rroot(N):  # get top root
    R = N.root
    while R.root and R.root.rng > N.rng: N = R; R = R.root
    return R

def comb_cent_(nG, rc):

    cent_, node_, cM = [], [], 0
    for g in nG.N_:
        if g.cent_:
            node_.extend({n for c in g.cent_[0] for n in c.N_})
            cent_.extend(g.cent_[0]); cM += g.cent_[1]
    if cM > ave * rc * loopw:
        comp_(node_, rc, fC=1)  # extend rims of mo_-reinforced nodes, if not in seen_
        cluster_C_(nG, cent_,rc, fext=1)  # use combined cent_ as iN_

def cluster_C_(root, N_, rc, fdeep=0, fext=0):  # form centroids by clustering exemplar surround, drifting via rims of new member nodes

    def comp_C(C, n):
        _,et,_ = base_comp(C,n, rc, C.wTT)
        if fdeep:
            if val_(et,1,len(n.derH)-2,rc):
                comp_H(C.derH, n.derH, n.Et[2]/C.Et[2], et)
            for L in C.rim:
                if L in n.rim: et += L.Et  # overlap
        return et[0]/et[2] if et[0]!=0 else 1e-7

    _C_, _N_ = [], []
    for N in N_:
        if not N.exe: continue  # exemplars or all
        if fext:
            C = N  # extend centroid
            C._N_ = [n for _N in C.N_ for l in _N.rim for n in l.N_ if n is not _N and l not in C.L_]  # not tested links
        else:  # init centroid
            C = cent_attr(Copy_(N,root, init=2), rc); C.N_ = [N]; # C.wTT = np.ones((2,8)) why we are init wTT to 1 again?
            C._N_ = [n for l in N.rim for n in l.N_ if n is not N]  # core members + surround for comp to N_ mean
        _N_ += C._N_; _C_ += [C]
    N__ = copy(set(_N_))  # pack all possible ns, including those out of graph's ns
    # reset:
    for n in set(root.N_+_N_): n.C_,n.mo_, n._C_,n._mo_ = [],[], [],[]  # aligned pairs
    # reform C_, refine C.N_s, which may extend beyond root:
    while True:
        C_, cnt,mat,olp, Dm,Do = [],0,0,0,0,0; Ave = ave * rc * loopw
        _Ct_ = [[c, c.Et[0]/c.Et[2] if c.Et[0] !=0 else 1e-7, c.olp] for c in _C_]
        for _C,_m,_o in sorted(_Ct_, key=lambda t: t[1]/t[2], reverse=True):
            if _m > Ave*_o:
                C = sum_N_(_C.N_, root=root, fC=1, rc=rc)  # add olp in N.mo_ to C.olp?
                _N_,_N__,L_, mo_, M,O, dm,do = [],[],[],[],0,0,0,0  # per C
                for n in _C._N_:  # core+ surround
                    if C in n.C_: continue
                    m = comp_C(C,n)  # val,olp / C:
                    o = np.sum([mo[0]/ m for mo in n._mo_ if mo[0]>m/3])  # overlap = higher-C inclusion vals / current C val
                    cnt += 1  # count comps per loop
                    if m > Ave * o:
                        _N_+=[n]; L_+=n.rim; M+=m; O+=o; mo_ += [np.array([m,o])]  # n.o for convergence eval
                        _N__ += [_n for l in n.rim for _n in l.N_ if _n is not n]  # +|-ls for comp C
                        if _C not in n._C_: dm += m; do += o  # not in extended _N__
                    elif _C in n._C_:
                        __m,__o = n._mo_[n._C_.index(_C)]; dm +=__m; do +=__o
                if M > Ave * len(_N_) * O:
                    for n, mo in zip(_N_,mo_): n.mo_ += [mo]; n.C_ += [C]
                    # we need to init C._L_ in sum_N?  Since we pack need to pack it into C.L_
                    C.M = M; C.L_+= C._L_; C._L_= list(set(L_)); C.N_+= _N_; C._N_= list(set(_N__))  # core, surround elements
                    C_+=[C]; mat+=M; olp+=O; Dm+=dm; Do+=do   # new incl or excl
                    for n in _N_: add_N(C, n, fC=1)  # we need this to update C params? Else their derTT won't be updated too
                else:
                    for n in _C._N_:
                        for i, c in enumerate(n.C_):
                            if c is _C: n.mo_.pop(i); n.C_.pop(i); break  # remove mo mapping to culled _C
            else: break  # the rest is weaker
        if Dm > Ave * cnt * Do:  # dval vs. dolp, overlap increases as Cs may expand in each loop?
            _C_ = [cent_attr(C,rc) for C  in C_]
            for n in N__: n._C_ = n.C_; n._mo_= n.mo_; n.C_,n.mo_ = [],[]  # new n.C_s, combine with vo_ in Ct_?
        else:  # converged
            break
    if  mat > Ave * cnt * olp:
        root.cent_ = (set(C_),mat)
        # cross_comp, low overlap eval in comp_node_?