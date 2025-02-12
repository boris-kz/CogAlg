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
