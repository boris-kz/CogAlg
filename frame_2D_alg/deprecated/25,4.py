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



