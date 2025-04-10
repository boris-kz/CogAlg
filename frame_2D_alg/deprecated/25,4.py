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

def add_N(N,n, fi=1, fappend=0, fw=0):

    if fw:  # proximity weighting in centroid only, other params not used
        r = ave_dist / np.hypot(*(N.yx - n.yx))
        n.baseT*=r; n.derTT*=r; n.Et*=r
        # also scale yx drift contribution?





