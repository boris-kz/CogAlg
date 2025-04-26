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

def get_exemplars(L_, ave):  # select for next cross_comp

    N_ = list(set([node for link in L_ for node in link.nodet]))
    exemplars, _N_ = [], set()
    for N in sorted(N_, key=lambda n: n.et[0]/n.et[2], reverse=True):
        M,_,n,_ = N.et  # sum from rim
        if eval(M, weights=[ave, n, clust_w, rolp_M(M, N,_N_)]):  # 1 + rolp_M to the intersect of inhibition zones
            exemplars += [N]; _N_.update(N._N_)
        else:
            break  # the rest of N_ is weaker
    return exemplars

def rolp_M(M, N, _N_):
    oL_ = [L for L,_ in N.rim if [n for n in L.nodet if n in _N_]]
    return 1 + sum([L.Et[0] for L in oL_]) / M  # ave weight in range 1:2

def comp_node_(_N_, ave, L=0):  # rng+ forms layer of rim and extH per N, appends N_,L_,Et, ~ graph CNN without backprop

    _Gp_ = [] # [G pair + co-positionals], for top-nested Ns, unless cross-nesting comp:
    if L: _N_ = filter(lambda N: len(N.derH)==L, _N_)
    # max len derH only
    for _G, G in combinations(_N_, r=2):
        if len(_G.H) != len(G.H):  # | root.H: comp top nodes only?
            continue
        _n, n = _G.Et[2], G.Et[2]; rn = _n/n if _n>n else n/_n
        radii = G.aRad + _G.aRad
        dy,dx = np.subtract(_G.yx,G.yx); dist = np.hypot(dy,dx)
        _G.add, G.add = 0, 0
        _Gp_ += [(_G,G, rn, dy,dx, radii, dist)]
    rng = 1
    N_,L_,ET = set(),[],np.zeros(4)
    _Gp_ = sorted(_Gp_, key=lambda x: x[-1])  # sort pairs by proximity
    while True:  # prior vM
        Gp_,Et = [],np.zeros(4)
        # add distance bands for nodes with incremented M?
        for Gp in _Gp_:
            _G, G, rn, dy,dx, radii, dist = Gp
            (_m,_,_n,_o),(m,_,n,o) = _G.Et, Et
            max_dist = ave_dist * (radii/aveR) * ((_m+m)/(ave*(_n+n+_o+o)) / int_w)  # ave_dist * radii * induction
            if max_dist > dist or _G._N_ & G._N_:
                # comp if close or share matching mediators
                Link = comp_N(_G,G, ave, fi=1, angle=[dy,dx], dist=dist, fshort = dist < max_dist/2)  # no density: incomplete ext V
                L_ += [Link]  # include -ve links
                if Link.Et[0] > ave * Link.Et[2] * loop_w:
                    N_.update({_G,G}); Et += Link.Et; _G.add,G.add = 1,1
            else:
                Gp_ += [Gp]  # re-evaluate not-compared pairs with one incremented N.M
        ET += Et
        if Et[0] > ave * Et[2] * loop_w:  # current-rng vM
            _Gp_ = [Gp for Gp in Gp_ if Gp[0].add or Gp[1].add]  # one incremented N.M
            rng += 1
        else:  # low projected rng+ vM
            break

    return  list(N_), L_, ET  # flat N__ and L__

def cluster_N_(root, L_, ave, rc):  # top-down segment L_ by >ave ratio of L.dists

    nG = CG(root=root)
    L_ = sorted(L_, key=lambda x: x.L)  # short links first
    min_dist = 0; Et = root.Et
    while True:
        # each loop forms G_ of L_ segment with contiguous distance values: L.L
        _L = L_[0]; N_, et = copy(_L.nodet), _L.Et
        for n in [n for l in L_ for n in l.nodet]:
            n.fin = 0
        for i, L in enumerate(L_[1:], start=1):
            if L.L/_L.L < 1.2: # >=1, no dist+ if weak?
                _G,G = L.nodet
                if val_((_G.Et+G.Et)*int_w + (_G.et+G.et), root.Et, aw= 2*rc*loop_w) > 0:  # _G.et+G.et: surround density
                    _L = L; N_ += L.nodet; et += L.Et  # else skip weak link inside segment
            else:
                i -= 1; break  # terminate contiguous-distance segment
        max_dist = _L.L
        G_ = []  # cluster current distance segment N_:
        if val_(et, Et, mw=(len(N_)-1)*Lw, aw=rc*clust_w) > 0:
            for N in {*N_}:
                if N.fin: continue  # clustered from prior _N_
                _eN_,node_,link_,et, = [N],[],[], np.zeros(4)
                while _eN_:
                    eN_ = []
                    for eN in _eN_:  # cluster rim-connected ext Ns, all in root Gt
                        node_+=[eN]; eN.fin = 1  # all rim
                        for L,_ in eN.rim:  # all +ve
                            if L not in link_:
                                eN_ += [n for n in L.nodet if not n.fin]
                                if L.L < max_dist:
                                    link_+=[L]; et+=L.Et  # current distance segment
                    _eN_ = {*eN_}
                # Gt:
                link_ = list({*link_})
                Lay = CLay(); [Lay.add_lay(lay) for lay in sum_H(link_, nG, fi=0)]
                derTT = Lay.derTT
                # weigh m_|d_ by similarity to mean m|d, weigh derTT:
                m_,M = centroid_M(derTT[0], ave=ave); d_,D = centroid_M(derTT[1], ave=ave)
                et[:2] = M,D; Lay.derTT = np.array([m_,d_])
                node_ = list({*node_})  # cluster roots:
                if val_(et, Et, mw=(len(node_)-1)*Lw, aw=rc*clust_w) > 0:
                    G_ += [sum2graph(nG, [node_,link_, et, Lay], 1, min_dist, max_dist)]
                else: G_ += [N]
        else:
            G_ += N_  # unclustered nodes
        G_ = list(set(G_)); L_ = L_[i + 1:] # longer links:
        if G_:
            [comb_alt_(G.alt_, ave, rc) for G in G_ if isinstance(G.alt_,list)]  # not nested in higher-dist Gs, but nodes have all-dist roots
        if L_:
            min_dist = max_dist  # next loop connects current-distance clusters via longer links
        else:
            break
    if G_:
        return sum_N_(G_, root_G=nG)  # highest dist segment, includes all nodes

def cluster_L_(root, L_, ave, rc, fnodet=1):  # CC links via nodet or rimt, no dist-nesting

    lG = CG(root=root)
    G_ = []  # flood-fill link clusters
    for L in L_: L.fin = 0
    for L in L_:
        if L.fin: continue
        L.fin = 1
        node_, link_, Et, Lay = [L], [], copy(L.Et), CLay()
        if fnodet:
            for _L in L.nodet[0].rim+L.nodet[1].rim if isinstance(L.nodet[0],CG) else [l for n in L.nodet for l,_ in n.rimt[0]+n.rimt[1]]:
                if _L in L_ and not _L.fin and _L.Et[1] > avd * _L.Et[2]:  # direct clustering by dval
                    link_ += L.nodet; Et += _L.Et; _L.fin = 1; node_ += [_L]
        else:
            for lL,_ in L.rimt[0] + L.rimt[1]:
                _L = lL.nodet[0] if lL.nodet[1] is L else lL.nodet[1]
                if _L in L_ and not _L.fin:  # +ve only, eval by directional density?
                    link_ += [lL]
                    Et += lL.Et; _L.fin = 1; node_ += [_L]
        node_ = list({*node_})
        if val_(Et, mw=(len(node_)-1)*Lw, aw=rc*clust_w) > 0:
            Lay = CLay()
            [Lay.add_lay(l) for l in sum_H(node_ if fnodet else link_, root=lG, fi=0)]
            G_ += [sum2graph(lG, [node_, link_, Et, Lay], fi=0)]
    if G_:
        [comb_alt_(G.alt_, ave, rc) for G in G_ if isinstance(G.alt_,list)]  # no alt_ in dgraph?
        sum_N_(G_, root_G = lG)
        # return lG

def cross_comp(root, rc, iN_, fi=1):  # rc: recursion count, fc: centroid phase, cross-comp, clustering, recursion

    N_,L_,Et = comp_node_(iN_, ave*rc) if fi else comp_link_(iN_, ave*rc)  # flat node_ or link_
    # N__,L__ if comp_node_
    if N_:
        mL_,dL_ = [],[]
        for l in [l for l_ in L_[1:] for l in l_] if fi else L_:  # exemplars in L__[1:] if comp_node_?
            if l.Et[0] > ave * l.Et[2]:
                mL_+= [l]
            if l.Et[1] > avd * l.Et[2]: dL_+= [l]
        nG, lG = [],[]
        # mfork:
        if val_(Et, mw=(len(mL_)-1)*Lw, aw=(rc+2)*clust_w) > 0:  # np.add(Et[:2]) > (ave+ave_d) * np.multiply(Et[2:])?
            if fi:                                               # cc_w if fc else lc_w? rc+=1 for all subseq ops?
                N__ = [n for n_ in N_[1:] for n in n_]
                if len(N__) > 1:  # N__ has >ave rim nodes
                    eN_ = get_exemplars(N__, ave*(rc+2))  # same as N__[1:]? (It should be different? Why it is the same?)
                    if eN_:  # exemplars: typical sparse nodes
                        eL_ = {l for n in eN_ for l,_ in n.rim}  # +connectivity: if l.L < ave_dist?
                        if val_(np.sum([l.Et for l in eL_],axis=0), Et, mw=((len(eL_)-1)*Lw), aw=(rc+3)*clust_w) > 0:
                            for n in iN_: n.fin=0
                            nG = cluster_exemplar_(root, eL_, ave*(rc+3), rc+3)  # cluster exemplars
            else:  # N_ = dfork L_
                nG = cluster_L_(root, N_, ave*(rc+2), rc=rc+2, fnodet=0)  # via llinks, no CC, no dist-nesting
            if nG:
                if val_(nG.Et, Et, mw=(len(nG.node_)-1)*Lw, aw=(rc+3)*loop_w) > 0:
                    cross_comp(nG, rc=rc+3, iN_=nG.node_) # recursive agglomeration -> root node_H
        # dfork:
        dval = val_(Et, mw=(len(dL_)-1)*Lw, aw=(rc+3)*clust_w, fi=0)
        if dval > 0:
            if isinstance(L_[0], list): L__ = [l for l_ in L_[1:] for l in l_]
            else:                       L__ = L_
            if dval > ave:  # recursive derivation -> lH within node_H level
                lG = cross_comp(sum_N_(L__),rc+3, L2N(L__), fi=0)  # comp_link_, no CC
            else:  # lower res, dL_ eval?
                lG = cluster_L_(sum_N_(dL_), L2N(L__), ave*(rc+3), rc=rc+3, fnodet=1)
                if nG: [comb_alt_(G.alt_, ave, rc) for G in nG.node_ if isinstance(G.alt_,list)]  # after mfork
        if nG or lG:
            lev = []
            for g in nG,lG:
                if g: lev += [g]; add_N(root, g)  # appends root.derH
                else: lev += [[]]
            root.H += [lev] + nG.H if nG else []  # nG.H from recursion, if any
            if lG:
                root.lH += lG.H+[sum_N_(lG.node_,root=lG)]  # lH: H within node_ level
        return nG