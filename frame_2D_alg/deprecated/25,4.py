'''                    # convert edge to CG
                    cross_comp(edge, rc=1, iN_=[PP2G(PP)for PP in edge.node_])  # restricted range and recursion, no comp PPd_ and alts?
                    edge_ += [edge]
    # unpack edges
    [nG0,lG0] = edge_[0].H.pop(0)  # lev0 = [PPm_,PPd_], H,derH layered in cross_comp
    H, derH, baseT, derTT = nG0.H, nG0.derH, nG0.baseT, nG0.derTT
    if lG0:  derH, baseT, derTT = add_H(derH, lG0.derH, root=frame), np.add(baseT, lG0.baseT), np.add(derTT, lG0.derTT)
    for edge in edge_:
        H, derH, baseT, derTT = add_GH(H, edge.H, root=frame), add_H(derH, edge.derH, root=frame), np.add(baseT, edge.baseT), np.add(derTT, edge.derTT)
    # no frame2G?
    frame.H, frame.derH, frame.baseT, frame.derTT = H, derH, baseT, derTT
    return frame
'''
def frame2G(G, **kwargs):
    blob2G(G, **kwargs)
    G.derH = kwargs.get('derH', [CLay(root=G, Et=np.zeros(4), derTT=[], node_=[],link_ =[])])
    G.Et = kwargs.get('Et', np.zeros(4))
    G.node_ = kwargs.get('node_', [])

def blob2G(G, **kwargs):
    # node_, Et stays the same:
    G.fi = 1  # fi=0 if cluster Ls|lGs
    G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
    G.H = kwargs.get('H',[])  # [cG, nG, lG]
    G.derH = []  # sum from nodes, then append from feedback, maps to node_tree
    G.extH = []  # sum from rims
    G.baseT = np.zeros(4)  # I,G,Dy,Dx
    G.derTT = kwargs.get('derTT', np.zeros((2,8)))  # m_,d_ base params
    G.derTTe = kwargs.get('derTTe', np.zeros((2,8)))
    G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
    G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = (y+Y)/2,(x+X)/2, then ave node yx
    G.rim = []  # flat links of any rng, may be nested in clustering
    G.maxL = 0  # nesting in nodes
    G.aRad = 0  # average distance between graph center and node center
    G.altG = []  # or altG? adjacent (contour) gap+overlap alt-fork graphs, converted to CG
    return G

def get_node_(G, fi): return G.H[-1][fi] if isinstance(G.H[0],list) else G.H  # node_ | nG.node_

def sum_C(node_):  # sum|subtract and average C-connected nodes

        C = copy_(node_[0]); C.H = node_  # add root and medoid / exemplar?
        C.M = 0
        sum_N_(node_[1:], root_G=C)  # no extH, extend_box
        alt_ = [n.altG for n in node_ if n.altG]
        if alt_:
            sum_N_(alt_, root_G=C.altG)  # no m, M, L in altG
        k = len(node_)
        for n in (C, C.altG):
            n.Et/=k; n.baseT/=k; n.derTT/=k; n.aRad/=k; n.yx /= k
            norm_H(n.derH, k)
        return C
'''
        lay = comb_H_(L_,root,fi=0)
        if fi: root.derH += [[lay]]  # [mfork] feedback, no eval?
        else:  root.derH[-1] += [lay]  # dfork feedback
'''
def feedback(root, ifi):  # root is frame if ifi else lev_lG
    # draft
    rv_t = np.ones((2,8))  # sum derTT coefs: m_,d_ [M,D,n,o, I,G,A,L] / Et, baseT, dimension
    rM, rD = 1, 1; hG = sum_N_(root.H[-1][0])  # top level, no  feedback (we need sum_N_ here? Since last layer is flat)

    for lev in reversed(root.H[:-1]):
        for fi, fork_G in lev[0], lev[2]:  # CG node_ if fi else, CL link_
            if fi:
                _m,_d,_n,_ = hG.Et; m,d,n,_ = fork_G.Et
                rM += (_m/_n) / (m/n)  # no o eval?
                rD += (_d/_n) / (d/n)
                rv_t += np.abs((hG.derTT/_n) / (fork_G.derTT/n))
                hG = fork_G
            else:
                # also for ddfork: not ifi?
                rMd, rDd, rv_td = feedback(root, fi)  # intra-level recursion in lG
                rv_t = rv_t + rv_td; rM += rMd; rD += rDd

    return rM,rD,rv_t

def get_rim(N,fi): return N.rim if fi else N.rimt[0] + N.rimt[1]  # add nesting in cluster_N_?

def cluster_C_(L_, rc):  # 0 nest gap from cluster_edge: same derH depth in root and top Gs

    def refine_C_(C_):  # refine weights in fuzzy C cluster around N, in root node_|link_
        '''
        comp mean-node pairs, node_cluster_sum weight = match / (ave * cluster_overlap | (dist/max_dist)**2)
        delete weak clusters, redefine node_ by proximity, recompute cluster_sums of mean_matches'''
        remove_ = []
        for C in C_:
            r = 0  # recursion count
            while True:
                C.M = 0; dM = 0  # pruned nodes and values, or comp all nodes again?
                for N in C.node_:
                    m = sum( base_comp(C,N)[0][0])  # derTT[0][0]
                    if C.altG and N.altG:
                        m += sum( base_comp(C.altG,N.altG)[0][0])
                    N.Ct_ = sorted(N.Ct_, key=lambda c: c[1], reverse=True)  # _N.M rdn = n stronger root Cts
                    for i, [_C,_m] in enumerate(N.Ct_):
                        if _C is C:
                            vm = m - ave * (ave_dist/np.hypot(*(C.yx-N.yx))) * (len(set(C.node_)&set(_C.node_))/(len(C.node_)+len(_C.node_)) * (i+1))
                            # ave * rel_distance * redundancy * rel_overlap between clusters
                            dm = (_m-vm) if r else vm  # replace init 0
                            dM += dm; _C.M += dm
                            if _C.M > ave: N.Ct_[i][1] = vm
                            else:          N.Ct_.pop(i)
                            break  # CCt update only
                if C.M < ave*C.Et[2]*clust_w:
                    for n in C.node_:
                        if C in n.Ct_: n.Ct_.remove(C)
                    remove_ += [C]  # delete weak | redundant cluster
                    break
                # separate drift eval
                if dM > ave:
                    C = sum_N_(C.node_)  # recompute centroid, or ave * iterations: cost increase?
                    C.M = 0; k = len(node_)
                    for n in (C, C.altG): n.Et /= k; n.baseT /= k; n.derTT /= k; n.aRad /= k; n.yx /= k; norm_H(n.derH, k)
                else: break
                r += 1
        # filter and assign root in rC_: flat, overlapping within node_ level:
        return [C for C in C_ if C not in remove_]

    ave = globals()['ave'] * rc  # recursion count
    C_ = []  # init centroid clusters for next cross_comp
    N_ = list(set([node for link in L_ for node in link.nodet]))
    N_ = sorted(N_, key=lambda n: n.Et[0], reverse=True)  # strong nodes initialize centroids
    for N in N_:
        if N.Et[0] < ave * N.Et[2]: break
        med = 1; node_,_n_ = [],[N]; N.Ct_ = []
        while med <= ave_med and _n_:  # init C.node_ is _Ns connected to N by <=3 mediation degrees
            n_ = []
            for n in [n for _n in _n_ for link,_ in _n.rim for n in link.nodet]:
                if hasattr(n,'Ct_'):
                    if med==1: continue  # skip directly connected nodes if in other CC
                else:
                    n.Ct_ = []; n_ += [n]  # add to CC
            node_ += n_; _n_ = n_; med += 1
            # add node_ margin for C drift?
        if node_:
            C = sum_N_(list(set(node_)), fw=1)  # weigh nodes by inverse dist to node_[0]
            C.M = 0; k = len(node_)
            for n in (C, C.altG): n.Et /= k; n.baseT /= k; n.derTT /= k; n.aRad /= k; n.yx /= k; norm_H(n.derH, k)
            for n in node_:  # node_ is flat?
                n.Ct_ += [[C,0]]  # empty m, may be in multiple Ns
            C_ += [C]

    return refine_C_(C_)  # refine centroid clusters

def cluster_C_1(L_, rc):  # select medoids for LC, next cross_comp

    def cluster_C(N, M, medoid_, medoid__):  # compute C per new medoid, compare C-node pairs, recompute C if max match is not medoid

        node_,_n_, med = [], [N], 1
        while med <= ave_med and _n_:  # node_ is _Ns connected to N by <=3 mediation degrees
            n_ = [n for _n in _n_ for link,_ in _n.rim for n in link.nodet  # +ve Ls
                  if med >1 or not n.C]  # skip directly connected medoids
                  # also test overlap?
            node_ += n_; _n_ = n_; med += 1
        node_ = list(set(node_))
        C = sum_N_(node_); k = len(node_)  # default
        for n in (C, C.altG): n.Et /= k; n.baseT /= k; n.derTT /= k; n.aRad /= k; n.yx /= k; norm_H(n.derH, k)
        maxM, m_ = 0,[]
        for n in node_:
            m = sum( base_comp(C,n)[0][0])  # derTT[0][0]
            if C.altG and N.altG: m += sum( base_comp(C.altG,N.altG)[0][0])
            m_ += [m]
            if m > maxM: maxN = n; maxM = m
        if maxN.C:
            return  # or compare m to mroott_'C'm and replace with local C if stronger?
        if N.mroott_:
            for _medoid,_m in N.mroott_:
                rel_olp = len(set(C.node_) & set(_medoid.C.node_)) / (len(C.node_) + len(_medoid.C.node_))
                if _m > m:
                    if m > ave * rel_olp:  # add medoid
                        for _N, __m in zip(C.node_, m_): _N.mroott_ += [[maxN, __m]]  # _N match to C
                        maxN.C = C; medoid_ += [maxN]; M += m
                    elif _m < ave * rel_olp:
                        medoid_.C = None; medoid_.remove(_medoid); M += _m  # abs change? remove mroots?
        elif m > ave:
            N.mroott_ = [[maxN, m]]; maxN.C = C; medoid_ += [maxN]; M += m

    N_ = iN_ = list(set([node for link in L_ for node in link.nodet]))
    ave = globals()['ave'] * rc
    medoid__ = []
    for N in N_:
        N.C = None; N.mroott_ = []  # init
    while N_:
        M, medoid_ = 0, []
        N_ = sorted(N_, key=lambda n: n.Et[0]/n.Et[2], reverse=True)  # strong nodes initialize centroids
        for N in N_:
            if N.Et[0] > ave * N.Et[2]:
                cluster_C(N, M, medoid_, medoid__)
            else:
                break  # rest of N_ is weaker
        medoid__ += medoid_
        if M < ave * len(medoid_):
            break  # converged
        else:
            N_ = medoid_

    return {n for N in medoid__ for n in N.C.node_ if sum([Ct[1] for Ct in n.mroott_]) > ave * clust_w}  # exemplars

def layer_C_(root, L_, rc):  # node-parallel cluster_C_ in mediation layers, prune Cs in top layer?
    # same nodes on all layers, hidden layers mediate links up and down, don't sum or comp anything?
    pass

def add_N(N,n, fi=1, fappend=0, fw=0):

    if fw:  # proximity weighting in centroid only, other params not used
        r = ave_dist / np.hypot(*(N.yx - n.yx))
        n.baseT*=r; n.derTT*=r; n.Et*=r
        # also scale yx drift contribution?

def add_node_H(H,h):

    for Lev, lev in zip(H, h):  # always aligned?
        for F, f in zip_longest(Lev, lev, fillvalue=None):
            if f:
                if F: add_N(F,f)  # nG|lG
                else: Lev += [f]  # if lG

    # if len(_P_)==1 and len(next(iter(_P_)).dert_)==1: continue
'''
        for Gp in _Gp_:
            _G,G, rn, dy,dx, radii, dist = Gp
            medG_ = _G._N_ & G._N_
            if medG_:
                _mL_,mL_ =[],[]; fcomp=0; fshort = 0  # compare indirectly connected Gs
                for g in medG_:
                    for mL in g.rim:
                        if mL in G.rim: mL_ += [mL]
                        elif mL in _G.rim: _mL_ += [mL]
                Lpt_ = [[_l,l,comp_angle(_l.baseT[2:],l.baseT[2:])[1]] for (_l,_),(l,_) in product(mL_,_mL_)]
                [_l,l,dA] = max(Lpt_, key=lambda x: x[2])  # links closest to the opposite from medG
                if dA > 0.4:  # we probably can include dA back?
                    G_ = set(_l.nodet) ^ set(l.nodet)  # we might get 4 Gs here now, _G, G and the other 2 different nodes from both of their rim
                    if len(G_) == 2: 
                        _G,G = list(G_)[:2]  # if there's 4 Gs, that's mean they are not directly mediated by link? So we should skip them?
                        fcomp=1
                # end nodes
            else:  # eval new Link, dist vs radii * induction, mainly / extH?
                (_m,_,_n,_),(m,_,n,_) = _G.Et,G.Et
                weighted_max = ave_dist * ((radii/aveR * int_w**3) * (_m/_n + m/n)/2 / (ave*(_n+n)))  # all ratios
                if dist < weighted_max:
                    fcomp=1; fshort = dist < weighted_max/2  # no density, ext V is not complete
                else: fcomp=0
            if fcomp:
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist, fshort=fshort)
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[2] * loop_w:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
'''
def merge_Lp(L, l, w_):  # combine med_Ls into Link

    L = copy_(L)
    L.nodet = list(set(L.nodet) ^ set(l.nodet))  # get end nodes and remove mediator
    L.box = extend_box(L.box, l.box)
    L.yx = (L.yx + l.yx) / 2
    L.baseT += l.baseT
    for Lay,lay in zip(L.derH, l.derH):
        L.derTT[1] += lay.derTT[1]
        L.Et[1] += lay.Et[1]  # D only, keep iL n,o?
    # get max comparands from L.nodet to compute M, L=A:
    _Et, Et = np.abs(L.nodet[0].Et), np.abs(L.nodet[1].Et)
    M,D,n,o = np.minimum(_Et,Et) / np.maximum(_Et,Et)
    _IG, IG = L.nodet[0].baseT[:2], L.nodet[1].baseT[:2]
    I,G = np.minimum(_IG) / np.maximum(IG)
    _y0,_x0,_yn,_xn = L.nodet[0].box; y0,x0,yn,xn = L.nodet[1].box
    _Len = (_yn-_y0) * (_xn-_x0); Len = (yn-y0) * (xn-x0)
    mL = min(_Len,Len) / max(_Len,Len)
    A = .5  # max dA
    # recompute match as max comparands - abs diff:
    L.derTT[0] = np.array([M,D,n,o,I,G,mL,mA]) * w_
    L.Et[0] = np.sum(L.derTT[0] - np.abs(L.derTT[1]))
    return L

def val_(Et, _Et, ave, mw=1, aw=1, fi=1):  # m+d cluster | cross_comp eval, including cross|root alt_Et projection

    m, d, n, o = Et; _m,_d,_n,_o = _Et  # cross-fork induction of root Et alt, same o (overlap)?
    m *= mw  # such as len*Lw

    d_loc = d * (_m - ave * aw * (_n/n))  # diff * co-projected m deviation, no bilateral deviation?
    d_ave = d - avd / ave  # d deviation, filter ave_d decreases with ave m, which lends value to d

    if fi: val = m + d_ave - d_loc  # match + proj surround val - blocking val, * decay?
    else:  val = d_ave + d_loc  # diff borrow val, generic + specific

    return val - ave * aw * n * o  # simplified: np.add(Et[:2]) > ave * np.multiply(Et[2:])

def val_1(Et, _Et=None, mw=1, aw=1, fi=1):  # m+d val per cluster|cross_comp

    m, d, n, o = Et  # m->d lend cancels-out in Et scope, not in higher-scope _Et?
    am = ave * aw
    ad = avd * aw
    if _Et:  # higher scope values
        _m,_d,_n,_o = _Et; rn = _n / n
        _m_dev = _m * rn - am
        _d_dev = _d * rn - ad
    else: _m_dev,_d_dev = 0,0

    if fi: return m*mw - am - (d - ad + _d_dev)  # m_dev -= lend to d_dev, local + global
    else:  return d*mw - ad + (m - am + _m_dev)  # d_dev += borrow from m_dev, local + global

def add_node_H(H, h, root):

    for Lev, lev in zip(H, h):  # always aligned?
        for i, (F, f) in enumerate(zip_longest(Lev, lev, fillvalue=None)):
            if f:  # nG | lG | dH
                if isinstance(F,CG): add_N(F,f)  # nG|lG
                elif isinstance(F,list): add_node_H(F,f, root=Lev[1])  # dH rooted in lG?
                elif F is None: Lev += [f]  # if lG, no empty layers?
                else:  Lev[i] = copy_(f, root=root)  # replace empty layer


