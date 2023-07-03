def feedback(root):  # bottom-up update root.H, breadth-first

    fbV = aveG+1
    while root and fbV > aveG:
        if all([[node.fterm for node in root.node_]]):  # forward was terminated in all nodes
            root.fterm = 1
            fbval, fbrdn = 0,0
            for node in root.node_:
                # node.node_ may empty when node is converted graph
                if node.node_ and not node.node_[0].box:  # link_ feedback is redundant, params are already in node.derH
                    continue
                for sub_node in node.node_:
                    fd = sub_node.fds[-1] if sub_node.fds else 0
                    if not root.H: root.H = [CQ(H=[[],[]])]  # append bottom-up
                    if not root.H[0].H[fd]: root.H[0].H[fd] = Cgraph()
                    # sum nodes in root, sub_nodes in root.H:
                    sum_parH(root.H[0].H[fd].derH, sub_node.derH)
                    sum_H(root.H[1:], sub_node.H)  # sum_G(sub_node.H forks)?
            for Lev in root.H:
                fbval += Lev.val; fbrdn += Lev.rdn
            fbV = fbval/max(1, fbrdn)
            root = root.root
        else:
            break

def sub_recursion_eval(root, PP_, fd):  # fork PP_ in PP or blob, no derH in blob

    # add RVal=0, DVal=0 to return?
    term = 1
    for PP in PP_:
        if PP.valt[fd] > PP_aves[fd] * PP.rdnt[fd] and len(PP.P_) > ave_nsub:
            term = 0
            sub_recursion(PP)  # comp_der|rng in PP -> parLayer, sub_PPs
        elif isinstance(root, CPP):
            root.fback_ += [[PP.derH, PP.valt, PP.rdnt]]
    if term and isinstance(root, CPP):
        feedback(root, fd)  # upward recursive extend root.derT, forward eval only
        '''        
        for i, sG_ in enumerate(sG_t):
            val,rdn = 0,0
            for sub_G in sG_:
                val += sum(sub_G.valt); rdn += sum(sub_G.rdnt)
            Val = Valt[fd]+val; Rdn = Rdnt[fd]+rdn
        or
        Val = Valt[fd] + sum([sum(G.valt) for G in sG_])
        Rdn = Rdnt[fd] + sum([sum(G.rdnt) for G in sG_])
        '''

def slice_blob(blob, verbose=False):  # form blob slices nearest to slice Ga: Ps, ~1D Ps, in select smooth edge (high G, low Ga) blobs

    P_ = []
    height, width = blob.mask__.shape

    for y in range(1, height, 1):  # iterate through lines, each may have multiple slices -> Ps, y0 and yn are extended mask
        if verbose: print(f"\rConverting to image... Processing line {y + 1}/{height}", end=""); sys.stdout.flush()
        _mask = True  # mask -1st dert
        x = 1  # 0 is extended mask
        while x < width-1:  # iterate through pixels in a line (last index is extended mask)
            mask = blob.mask__[y, x]
            dert = [par__[y, x] for par__ in blob.der__t[1:]]   # exclude i
            g, ga, ri, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = dert
            if not mask:  # masks: if 0,_1: P initialization, if 0,_0: P accumulation, if 1,_0: P termination
                if _mask:  # ini P params with first unmasked dert
                    Pdert_ = [dert]
                    I = ri; M = ave_g - g; Ma = ave_ga - ga; Dy = dy; Dx = dx
                    Sin_da0, Cos_da0, Sin_da1, Cos_da1 = sin_da0, cos_da0, sin_da1, cos_da1
                else:
                    # dert and _dert are not masked, accumulate P params:
                    I +=ri; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx  # angle
                    Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1  # aangle
                    Pdert_ += [dert]
            elif not _mask:
                # _dert is not masked, dert is masked, pack P:
                P_ += [term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x-1, Pdert_)]
            _mask = mask
            x += 1
        if not _mask:  # pack last P:
            P_ += [term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x-1, Pdert_)]

    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")
    blob.P_ = P_
    return P_

def term_P(I, M, Ma, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1, y,x, Pdert_):

    G = np.hypot(Dy, Dx); Ga = (Cos_da0 + 1) + (Cos_da1 + 1)  # recompute G,Ga, it can't reconstruct M,Ma
    L = len(Pdert_)  # params.valt = [params.M+params.Ma, params.G+params.Ga]?
    return CP(ptuple=[I, G, Ga, M, Ma, [Dy, Dx], [Sin_da0, Cos_da0, Sin_da1, Cos_da1], L], dert_=Pdert_, y=y, x=x-(L+1)//2)

def pad1(mask__):  # pad blob.mask__ with rim of 1s:
    return np.pad(mask__,pad_width=[(1,1),(1,1)], mode='constant',constant_values=True)

def rotate_P_(blob, verbose=False):  # rotate each P to align it with direction of P or dert gradient

    iP_ = copy(blob.P_); der__t = blob.der__t; mask__= blob.mask__
    if verbose: i = 0
    P_ = []
    for P in iP_:
        G = P.ptuple[1]
        daxis = P.ptuple[5][0] / G  # dy: deviation from horizontal axis
        _daxis = 0
        if verbose: i += 1
        while abs(daxis) * G > ave_rotate:  # recursive reform P in blob.der__t along new G angle:
            if verbose: print(f"\rRotating... {i}/{len(P_)}: {round(np.degrees(np.arctan2(*P.axis)))}°", end=" " * 79); sys.stdout.flush()
            _axis = P.axis
            P = form_P(der__t, mask__, axis=np.divide(P.ptuple[5], np.hypot(*P.ptuple[5])), y=P.y, x=P.x)  # pivot to P angle
            maxis, daxis = comp_angle(_axis, P.axis)
            ddaxis = daxis +_daxis  # cancel-out if opposite-sign
            _daxis = daxis
            G = P.ptuple[1]  # rescan in the direction of ave_a, P.daxis if future reval:
            if ddaxis * G < ave_rotate:  # terminate if oscillation
                axis = np.add(_axis, P.axis)
                axis = np.divide(axis, np.hypot(*axis))  # normalize
                P = form_P(der__t, mask__, axis=axis, y=P.y, x=P.x)  # not pivoting to dert G
                break
        for _,y,x in P.dert_ext_:  # assign roots in der__t
            x0 = int(np.floor(x)); x1 = int(np.ceil(x)); y0 = int(np.floor(y)); y1 = int(np.ceil(y))
            kernel = []
            if not mask__[y0][x0]: kernel += [[[y0,x0], np.hypot((y-y0),(x-x0))]]
            if not mask__[y0][x1]: kernel += [[[y0,x1], np.hypot((y-y0),(x-x1))]]
            if not mask__[y1][x0]: kernel += [[[y1,x0], np.hypot((y-y1),(x-x0))]]
            if not mask__[y1][x1]: kernel += [[[y1,x1], np.hypot((y-y1),(x-x1))]]
            '''
            or, add mask checking:
            x0 = int(x); y0 = int(y)
            x1 = x0 + 1; y1 = y0 + 1
            y = y0 if y1 - y > y - y0 else y1  # nearest cell y
            x = x0 if x1 - x > x - x0 else x1  # nearest cell x
            '''
            y,x = sorted(kernel, key=lambda x: x[1])[0][0]  # nearest cell y,x
            blob.der__t_roots[y][x] += [P]  # final rotated P

        P_ += [P]
    blob.P_[:] = P_

    if verbose: print("\r", end=" " * 79); sys.stdout.flush(); print("\r", end="")

def form_P(der__t, mask__, axis, y,x):

    rdert_,dert_ext_ = [],[]
    P = CP()
    rdert_,dert_ext_, ry,rx = scan_direction(P, rdert_,dert_ext_, y,x, axis, der__t,mask__, fleft=1)  # scan left, include pivot
    x0=rx; yleft=ry  # up-right or down-right
    rdert_,dert_ext_, ry,rx = scan_direction(P, rdert_,dert_ext_, ry,rx, axis, der__t,mask__, fleft=0)  # scan right:
    # initialization:
    rdert = rdert_[0]
    G, Ga, I, Dy, Dx, Sin_da0, Cos_da0, Sin_da1, Cos_da1 = rdert; M=ave_g-G; Ma=ave_ga-Ga; dert_=[rdert]
    # accumulation:
    for rdert in rdert_[1:]:
        g, ga, i, dy, dx, sin_da0, cos_da0, sin_da1, cos_da1 = rdert
        I+=i; M+=ave_g-g; Ma+=ave_ga-ga; Dy+=dy; Dx+=dx; Sin_da0+=sin_da0; Cos_da0+=cos_da0; Sin_da1+=sin_da1; Cos_da1+=cos_da1
        dert_ += [rdert]
    L=len(dert_)
    P.dert_ = dert_
    P.dert_ext_ = dert_ext_
    P.y = yleft+axis[0]*((L+1)//2); P.x = x0+axis[1]*((L+1)//2)
    G = np.hypot(Dy,Dx); Ga =(Cos_da0+1)+(Cos_da1+1) # recompute G,Ga
    P.ptuple = [I,G,Ga,M,Ma, [Dy,Dx], [Sin_da0,Cos_da0,Sin_da1,Cos_da1], L]
    P.axis = axis
    return P

def scan_direction(P, rdert_,dert_ext_, y,x, axis, der__t,mask__, fleft):  # leftward or rightward from y,x

    sin,cos = axis
    while True:
        x0, y0 = int(x), int(y)  # floor
        x1, y1 = x0+1, y0+1  # ceiling
        if all([mask__[y0,x0], mask__[y0,x1], mask__[y1,x0], mask__[y1,x1]]):
            break  # need at least one unmasked cell to continue direction
        kernel = [  # cell weighing by inverse distance from float y,x:
            # https://www.researchgate.net/publication/241293868_A_study_of_sub-pixel_interpolation_algorithm_in_digital_speckle_correlation_method
            (y0,x0, (y1-y) * (x1-x)),
            (y0,x1, (y1-y) * (x-x0)),
            (y1,x0, (y-y0) * (x1-x)),
            (y1,x1, (y-y0) * (x-x0))]
        ptuple = [
            sum((par__[cy,cx] * dist for cy,cx, dist in kernel))
            for par__ in der__t[1:]]
        if fleft:
            y -= sin; x -= cos  # next y,x
            rdert_ = [ptuple] + rdert_ # append left
            dert_ext_ = [[[P],y,x]] + dert_ext_  # append left external params: roots and coords per dert
        else:
            y += sin; x += cos  # next y,x
            rdert_ = rdert_ + [ptuple]  # append right
            dert_ext_ = dert_ext_ + [[[P],y,x]]

    return rdert_,dert_ext_, y,x
