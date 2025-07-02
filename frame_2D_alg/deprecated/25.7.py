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

