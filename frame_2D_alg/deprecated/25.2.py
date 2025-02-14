def weigh_m_(m_, M, ave):  # adjust weights on attr matches, also add cost attrs

    L = len(m_)
    M = np.sqrt(sum([m ** 2 for m in m_]) / L)
    _w_ = [1 for _ in m_]
    while True:
        w_ = [m/M for m in m_]  # rational deviations from mean
        Dw = sum([abs(w - _w) for w, _w in zip(w_, _w_)])  # weight update
        M = sum((m if m else 1e-7) * w for m, w in zip(m_, w_)) / L  # M update
        if Dw > ave:
            _w_ = w_
        else:
            break
    return w_, M
'''
L = comp_N(hG, lev_G, rn=_n / n if _n > n else n / _n)
if Val_(L.Et, _Et=L.Et) > 0:

# m/mag per attr:
rm_ = np.divide(hG.vert[0], hG.latuple)  # summed from all layers: no need for base vert?
rm_ = np.divide(hG.vert[0], hG.vert[0] + np.abs(hG.vert[1]))
for lay in hG.derH:
    for fork in lay:  # add rel CLay.m_d_t[0], or vert m_=rm_?
        rm_ += np.divide(fork.m_d_t[0], fork.m_d_t[0] + np.abs(fork.m_d_t[1]))
'''

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
    m_t = np.array([mL,mA], dtype=float); d_t = np.array([dL,dA], dtype=float)
    _o,o = _N.Et[3],N.Et[3]; olp = (_o+o) / 2  # inherit from comparands?
    Et = np.array([mL+mA, abs(dL)+abs(dA), .3, olp])  # n = compared vars / 6
    if fd:  # CL
        m_t = np.array([m_t, np.zeros(6), np.zeros(6)], dtype=object)  # empty lat, ver
        d_t = np.array([d_t, np.zeros(6), np.zeros(6)], dtype=object)
    else:   # CG
        (mLat, dLat), L_et = comp_latuple(_N.latuple, N.latuple, _o,o)
        (mVer, dVer), V_et = comp_md_(_N.vert[1], N.vert[1], dir)
        m_t = np.array([m_t, mLat, mVer], dtype=object)
        d_t = np.array([d_t, dLat, dVer], dtype=object)
        Et += np.array([L_et[0]+V_et[0], L_et[1]+V_et[1], 2, 0])
        # same olp?
    Link = CL(fd=fd, nodet=[_N,N], yx=np.add(_N.yx,N.yx)/2, angle=angle, dist=dist, box=extend_box(N.box,_N.box))
    lay0 = CLay(root=Link, Et=Et, m_d_t=[m_t,d_t], node_=[_N,N], link_=[Link])
    derH = comp_H(_N.derH, N.derH, rn,Link,Et,fd)  # comp shared layers, if any
    Link.derH = [lay0, *derH]
    # spec: comp_node_(node_|link_), combinatorial, node_ nested / rng-)agg+?
    if not fd and _N.altG and N.altG:  # if alt M?
        Link.altL = comp_N(_N.altG, N.altG, _N.altG.Et[2] / N.altG.Et[2])
        Et += Link.altL.Et
    Link.Et = Et
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            add_H(node.extH, Link.derH, root=node, rev=rev, fd=1)
            node.Et += Et
    return Link

def comp_md_t(_d_t,d_t, rn=.1, dir=1):  # dir may be -1

    m_t_, d_t_ = [], []
    M, D = 0,0
    for _d_, d_ in zip(_d_t, d_t):
        d_ = d_ * rn  # normalize by compared accum span
        dd_ = (_d_ - d_ * dir)  # np.arrays
        md_ = np.minimum(np.abs(_d_), np.abs(d_))
        md_[(_d_<0) != (d_<0)] *= -1  # negate if only one compared is negative
        md_ = np.divide(md_,d_)  # rms
        m_t_ += [md_]; M += np.sqrt(sum([m**2 for m in md_]) / len(d_))  # weighted sum
        d_t_ += [dd_]; D += dd_.sum()  # same weighting?

        M = sum([ np.sqrt( sum([m**2 for m in m_]) /len(m_)) for m_ in m_t])  # weigh M,D by ind_m / tot_M
        D = sum([ np.sqrt( sum([d**2 for d in d_]) /len(d_)) for d_ in d_t])

    return np.array([m_t_,d_t_],dtype=object), np.array([M,D])  # [m_,d_], Et

def comp_N1(_N,N, rn, angle=None, dist=None, dir=1):  # dir if fd, Link.derH=dH, comparand rim+=Link

    fd = isinstance(N,CL)  # compare links, relative N direction = 1|-1
    # comp externals, or baseT L,A? len(node_) is before comp_node_?
    if fd:
        _L, L = _N.dist, N.dist; L*=rn; dL = _L - L; mL = min(_L,L) / max(_L,L) - ave_L  # rm
        mA,dA = comp_angle(_N.angle, [d*dir *rn for d in N.angle])  # rev 2nd link in llink
        # comp med if LL: isinstance(>nodet[0],CL), higher-order version of comp dist?
    else:
        _L, L = len(_N.node_),len(N.node_); dL = _L- L*rn; mL = min(_L, L*rn) - ave_L
        mA,dA = comp_area(_N.box, N.box)  # compare area in CG vs angle in CL
        # or base externals in latuple
    # init
    dext = np.array( [np.array([mL,mA]), np.array([dL,dA])])
    M = mL+mA; D = abs(dL)+abs(dA); _o,o = _N.Et[3],N.Et[3]; olp=(_o+o)/2  # inherited
    Et = np.array([M,D, 2 if fd else 8, olp])  # n comp vars
    Link = CL(fd=fd, nodet=[_N,N], yx=np.add(_N.yx,N.yx)/2, angle=angle, dist=dist, box=extend_box(N.box,_N.box))
    if fd:
        vert = np.array([np.zeros(6),np.zeros(6)])  # sum from dderH
    else:  # default vert:
        vert, (Mv,Dv) = comp_vert(_N.vert[1], N.vert[1])
        if np.any(N.dext):
            ddext = comp_dext(_N.dext[1], N.dext[1], rn)  # combine der_ext across all layers, same as vert
            dext += ddext; M += np.sum(ddext[0]); D += np.sum(ddext[1])
        M+= Mv; D+= Dv; Et[:2] = M,D
        if M > ave:  # specification
            dLat,lEt = comp_latuple(_N.latuple, N.latuple, _o,o) # lower value
            Et += np.array([lEt[0], lEt[1], 2, 0])  # same olp?
            vert += dLat
    if M > ave and (len(N.derH) > 2 or isinstance(N,CL)):  # else derH is redundant to dext,vert
        dderH = comp_H(_N.derH, N.derH, rn, Link, Et, fd)  # comp shared layers, if any
        # sum in dext and vert
        # comp_node_(node_|link_)
    Link.derH = [CLay(root=Link,Et=Et,node_=[_N,N],link_=[Link], m_d_t=[[dext[0],vert[0]],[dext[1],vert[1]]]), *dderH]
    # spec:
    if not fd and _N.altG and N.altG:  # if alt M?
        Link.altL = comp_N(_N.altG, N.altG, _N.altG.Et[2] / N.altG.Et[2])
        Et += Link.altL.Et
    Link.Et = Et
    if val_(Et) > 0:
        for rev, node in zip((0,1), (N,_N)):  # reverse Link direction for _N
            if fd: node.rimt[1-rev] += [(Link,rev)]  # opposite to _N,N dir
            else:  node.rim += [(Link,dir)]
            add_H(node.extH, Link.derH, root=node, rev=rev, fd=1)
            node.Et += Et
    return Link

class CG(CBase):  # PP | graph | blob: params of single-fork node_ cluster
    # graph / node
    def __init__(G,  **kwargs):
        super().__init__()
        G.Et = kwargs.get('Et', np.zeros(4))  # sum all param Ets
        G.fd_ = kwargs.get('fd_',[])  # list of forks forming G, 1 if cluster of Ls | lGs, for feedback only?
        G.root = kwargs.get('root')  # may extend to list in cluster_N_, same nodes may be in multiple dist layers
        G.latuple = kwargs.get('latuple', np.array([.0,.0,.0,.0,.0, np.zeros(2)], dtype=object))  # lateral I,G,M,D,L,[Dy,Dx]
        G.dext = kwargs.get('dext',np.array([np.zeros(2),np.zeros(2)]))  # [mext,dext], sum across fork tree
        G.vert = kwargs.get('vert',np.array([np.zeros(6),np.zeros(6)]))  # vertical m_d_ of latuple, sum across fork tree
        G.derH = kwargs.get('derH',[])  # each lay is [m,d]: Clay(Et,node_,link_,m_d_t), sum|concat links across fork tree
        G.extH = kwargs.get('extH',[])  # sum from rims, single-fork
        G.box = kwargs.get('box', np.array([np.inf,np.inf,-np.inf,-np.inf]))  # y0,x0,yn,xn
        G.yx = kwargs.get('yx', np.zeros(2))  # init PP.yx = [(y0+yn)/2,(x0,xn)/2], then ave node yx
        G.maxL = kwargs.get('maxL', 0)  # if dist-nested in cluster_N_
        G.aRad = 0  # average distance between graph center and node center
        G.altG = []  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG
        # G.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge
        # G.fback_ = []  # node fb buffer, n in fb[-1]
        G.nnest = kwargs.get('nnest',0)  # node_H if > 0, node_[-1] is top G_
        G.lnest = kwargs.get('lnest',0)  # link_H if > 0, link_[-1] is top L_
        G.node_ = kwargs.get('node_',[])
        G.link_ = kwargs.get('link_',[])  # internal links
        G.rim = kwargs.get('rim',[])  # external links
    def __bool__(G): return bool(G.node_)  # never empty

class CL(CBase):  # link or edge, a product of comparison between two nodes or links
    name = "link"
    def __init__(l,  **kwargs):
        super().__init__()
        l.nodet = kwargs.get('nodet',[])  # e_ in kernels, else replaces _node,node: not used in kernels
        l.Et = kwargs.get('Et', np.zeros(4))
        l.fd = kwargs.get('fd',0)
        l.derH = kwargs.get('derH',[])  # list of single-fork CLays
        l.angle = kwargs.get('angle',[])  # dy,dx between nodet centers
        l.dist = kwargs.get('dist',0)  # distance between nodet centers
        l.box = kwargs.get('box',[])  # sum nodet, not needed?
        l.yx = kwargs.get('yx',[])
        # add med, rimt, extH in der+
    def __bool__(l): return bool(l.nodet)

def comp_area(_box, box, rn):
    _y0,_x0,_yn,_xn =_box; _A = (_yn - _y0) * (_xn - _x0)
    y0, x0, yn, xn = box;   A = (yn - y0) * (xn - x0)
    return _A-A*rn, min(_A,A) - ave_L**2  # mA, dA

def comp_dext(_dext, dext, rn, dir=1):
    (_dL, _dA), (dL, dA) = _dext,dext

    ddL = _dL - dL * rn * dir; mdL = min(_dL, dL*rn) / max(_dL, dL*rn) - ave_L  # m/mag
    if _dL < 0 != dL < 0: mdL = -mdL  # m is negative for comparands of opposite sign
    ddA = _dA - dA * rn * dir; mdA = min(_dA, dA*rn) / max(_dA, dA*rn) - 2
    if _dA < 0 != dA < 0: mdA = -mdA

    return np.array([np.array([mdL,mdA]),np.array([ddL,ddA])])
