def comp_G_(G_, fder):  # cross-comp Gs (patterns of patterns): Gs, derGs, or segs inside PP

    for i, _G in enumerate(G_):

        for G in G_[i+1:]:  # compare each G to other Gs in rng, bilateral link assign
            if G in [node for link in _G.link_ for node in link.node_]:  # G,_G was compared in prior rng+, add frng to skip?
                continue
            # comp external params:
            _x = (_G.xn +_G.x0)/2; _y = (_G.yn +_G.y0)/2; x = (G.xn + G.x0)/2; y = (G.yn + G.y0)/2
            dx = _x - x; dy = _y - y
            distance = np.hypot(dy, dx)  # Euclidean distance between centroids, sum in G.sparsity
            proximity = ave_rng-distance  # coord match

            mang, dang = comp_angle(_G.angle, G.angle)  # dy,dx for derG or high-aspect Gs, both *= aspect?
            # n orders of len and sparsity, -= fill: sparsity inside nodes?
            if fder:
                mlen,dlen = 0,0; _sparsity = _G.sparsity; sparsity = G.sparsity  # single-link derGs
            else:
                _L = len(_G.node_); L = len(G.node_)
                dlen = _L - L; mlen = min(_L, L)
                if isinstance(G.node_[0][0], CP): _sparsity, sparsity = 1, 1
                else:
                    _sparsity = sum(node.sparsity for node in _G.node_) / _L
                    sparsity = sum(node.sparsity for node in G.node_) / L
            dspar = _sparsity-sparsity; mspar = min(_sparsity,sparsity)
            # draft:
            mext = [proximity, mang, mlen, mspar]; mVal = proximity + mang + mlen + mspar
            dext = [distance, dang, dlen, dspar];  dVal = distance + dang + dlen + dspar
            derext = [mext,dext,mVal,dVal]

            if mVal > ave_ext * ((sum(_G.valt)+sum(G.valt)) / (2*sum(G_aves))):  # max depends on combined G value
                mplevel, dplevel = comp_plevels(_G.plevels, G.plevels, _G.fds, G.fds, derext)
                valt = [mplevel[1] - ave_Gm, dplevel[1] - ave_Gd]  # norm valt: *= link rdn?
                derG = Cgraph(  # or mean x0=_x+dx/2, y0=_y+dy/2:
                    plevels=[mplevel,dplevel], y0=min(G.y0,_G.y0), yn=max(G.yn,_G.yn), x0=min(G.x0,_G.x0), xn=max(G.xn,_G.xn),
                    sparsity=distance, angle=[dy,dx], valt=valt, node_=[_G,G])
                _G.link_ += [derG]; G.link_ += [derG]  # any val
                for fd in 0,1:
                    if valt[fd] > 0:  # alt fork is redundant, no support?
                        for node, (graph, meds_, gvalt) in zip([_G, G], [G.roott[fd], _G.roott[fd]]):  # bilateral inclusion
                            if node not in graph:
                                for mderG in node.link_:
                                    mnode = mderG.node_[0] if mderG.node_[1] is node else mderG.node_[1]
                                    if mderG not in meds_:  # combined meds per node
                                        meds_ += [[derG.node_[0] if derG.node_[1] is node else derG.node_[1] for derG in node.link_]]
                                gvalt[0] += node.valt[0]; gvalt[1] += node.valt[1]
                                graph += [node, meds_, valt]  # meds per node


def eval_med_layer(graph_, graph, fd):   # recursive eval of reciprocal links from increasingly mediated nodes

    node_, meds_, valt = graph
    save_node_, save_meds_ = [], []
    adj_Val = 0  # adjust connect val in graph

    for G, med_node_ in zip(node_, meds_):  # G: node or sub-graph
        mmed_node_ = []  # __Gs that mediate between Gs and _Gs
        for _G in med_node_:
            for derG in _G.link_:
                if derG not in G.link_:  # link_ includes all unique evaluated mediated links, flat or in layers?
                    # med_PP.link_:
                    med_link_ = derG.node_[0].link_ if derG.node_[0] is not _G else derG.node_[1].link_
                    for _derG in med_link_:
                        if G in _derG.node_ and _derG not in G.link_:  # __G mediates between _G and G
                            G.link_ += [_derG]
                            adj_val = _derG.valt[fd] - ave_agg  # or increase ave per mediation depth
                            # adjust nodes:
                            G.valt[fd] += adj_val; _G.valt[fd] += adj_val  # valts not updated
                            valt[fd] += adj_val; _G.roott[fd][2][fd] += adj_val  # root is not graph yet
                            __G = _derG.node_[0] if _derG.node_[0] is not _G else _derG.node_[1]
                            if __G not in mmed_node_:  # not saved via prior _G
                                mmed_node_ += [__G]
                                adj_Val += adj_val
        if G.valt[fd]>0:
            # G remains in graph
            save_node_ += [G]; save_meds_ += [mmed_node_]  # mmed_node_ may be empty

    for G, mmed_ in zip(save_node_, save_meds_):  # eval graph merge after adjusting graph by mediating node layer
        add_mmed_= []
        for _G in mmed_:
            _graph = _G.roott[fd]
            if _graph in graph_ and _graph is not graph:  # was not removed or merged via prior _G
                _node_, _meds_, _valt = _graph
                for _node, _meds in zip(_node_, _meds_):  # merge graphs, ignore _med_? add direct links:
                    for derG in _node.link_:
                        __G = derG.node_[0] if derG.node_[0] is not _G else derG.node_[1]
                        if __G not in add_mmed_ + mmed_:  # not saved via prior _G
                            add_mmed_ += [__G]
                            adj_Val += derG.valt[fd] - ave_agg
                valt[fd] += _valt[fd]
                graph_.remove(_graph)
        mmed_ += add_mmed_

    graph[:] = [save_node_,save_meds_,valt]
    if adj_Val > ave_med:  # positive adj_Val from eval mmed_
        eval_med_layer(graph_, graph, fd)  # eval next med layer in reformed graph

'''
- Segment input blob into dir_blobs by primary direction of kernel gradient: dy>dx
- Merge weakly directional dir_blobs, with dir_val < cost of comp_slice_
- Evaluate merged blobs for comp_slice_: if blob.M > ave_M
'''
def segment_by_direction(iblob, **kwargs):

    dert__ = list(iblob.dert__)
    mask__ = iblob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]
    verbose = kwargs.get('verbose')
    render = kwargs.get('render')

    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:
    dir_blob_, idmap, adj_pairs = flood_fill(dert__, abs(dy__) > abs(dx__), verbose=verbose, mask__=mask__, fseg=True)
    assign_adjacents(adj_pairs)  # fseg=True: skip adding the pose

    if render: _dir_blob_ = deepcopy(dir_blob_) # get a copy for dir blob before merging, for visualization purpose
    merged_ids = []  # ids of merged adjacent blobs, to skip in the rest of dir_blobs

    for i, blob in enumerate(dir_blob_):
        if blob.id not in merged_ids:
            blob = merge_adjacents_recursive(blob, merged_ids, blob.adj_blobs[0], strong_adj_blobs=[])  # no pose

            if (blob.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
                blob.fsliced = True
                slice_blob(blob, verbose)  # slice and comp_slice_ across directional sub-blob
            iblob.dir_blobs.append(blob)

        for dir_blob in iblob.dir_blobs:
            if dir_blob.id in merged_ids:  # strong blob was merged to another blob, remove it
                iblob.dir_blobs.remove(dir_blob)

        if render: visualize_merging_process(iblob, dir_blob_, _dir_blob_,mask__, i)
    if render:
        # for debugging: visualize adjacents of merged blob to see that adjacents are assigned correctly after the merging:
        if len(dir_blob_)>50 and len(dir_blob_)<500:
            new_idmap = (np.zeros_like(idmap).astype('int'))-2
            for blob in iblob.dir_blobs:
                y0,yn,x0,xn = blob.box
                new_idmap[y0:yn,x0:xn] += (~blob.mask__)*(blob.id + 2)

            visualize_merging_process(iblob, dir_blob_, _dir_blob_, mask__, 0)
            from draw_frame_blobs import visualize_blobs
            visualize_blobs(new_idmap, iblob.dir_blobs)


def merge_adjacents_recursive(blob, merged_ids, adj_blobs, strong_adj_blobs):

    if dir_eval(blob.Dy, blob.Dx):  # directionally weak blob, merge with all adjacent weak blobs

        if blob in adj_blobs: adj_blobs.remove(blob)  # remove current blob from adj adj blobs (assigned bilaterally)
        merged_adj_blobs = []  # weak adj_blobs
        for adj_blob in adj_blobs:

            if dir_eval(adj_blob.Dy, adj_blob.Dx):  # also directionally weak, merge adj blob in blob:
                if adj_blob.id not in merged_ids:
                    merged_ids.append(adj_blob.id)
                    blob = merge_blobs(blob, adj_blob, strong_adj_blobs)
                    # recursively add adj_adj_blobs to merged_adj_blobs:
                    for adj_adj_blob in adj_blob.adj_blobs[0]:
                        # not included in merged_adj_blobs via prior adj_blob.adj_blobs, or is adj_blobs' blob:
                        if adj_adj_blob not in merged_adj_blobs and adj_adj_blob is not blob \
                                and adj_adj_blob.id not in merged_ids: # not merged to prior blob in dir_blobs
                            merged_adj_blobs.append(adj_adj_blob)
            else:
                strong_adj_blobs.append(adj_blob)

        if merged_adj_blobs:
            blob = merge_adjacents_recursive(blob, merged_ids, merged_adj_blobs, strong_adj_blobs)

        # all weak adj_blobs should now be merged, resulting blob may be weak or strong, vertical or lateral
        blob.adj_blobs = [[],[]]  # replace with same-dir strong_adj_blobs.adj_blobs + opposite-dir strong_adj_blobs:

        for adj_blob in strong_adj_blobs:
            # merge with same-direction strong adj_blobs:
            if (adj_blob.sign == blob.sign) and adj_blob.id not in merged_ids and adj_blob is not blob:
                merged_ids.append(adj_blob.id)
                blob = merge_blobs(blob, adj_blob, strong_adj_blobs)
            # append opposite-direction strong_adj_blobs:
            elif adj_blob not in blob.adj_blobs[0] and adj_blob.id not in merged_ids:
                blob.adj_blobs[0].append(adj_blob)
                blob.adj_blobs[1].append(2)  # assuming adjacents are open, just to visualize the adjacent blobs

    return blob
'''
    mulptiple (x,y)s are overlaid by (rx,ry), we need to compute overlaid_area for each (x,y)
    for (overlaid_y, overlaid_x), overlaid_area in overlaid_pixels:
        rdert__t[i][ry, rx] += dert__t[i][overlaid_y, overlaid_x] * overlaid_area # overlaid_area is per pixel, fractional.
    or distance between (rx,ry) and (x,y): max=1,   
'''
def rotate(P, dert__t, mask__):

    angle = np.arctan2(P.ptuple.angle[0], P.ptuple.angle[1])  # angle of rotation, in rad
    nparams = len(dert__t)  # number of params in dert, it should be 10 here
    yn, xn = dert__t[0].shape[:2]
    xcenter, ycenter = int(P.x0 + P.ptuple.L / 2), P.y  # center of P
    mask_ = mask__[ycenter,:]
    # r = rotated
    # initialization
    x_, rxs_, rys_ = [xcenter], [[xcenter]], [[ycenter]]
    min_rx, min_ry, max_rx, max_ry = 0, 0, 0, 0
    xdistances_, ydistances_ = [[0]], [[0]]

    cx = xcenter
    while cx-1>0 and not mask_[cx-1] :  # scan left
        cx -= 1
        rx = (np.cos(angle) * (cx-xcenter)) + xcenter  # y part of equation is removed bcause y and ycenter is same line
        ry = (np.sin(angle) * (cx-xcenter)) + ycenter
        if rx % 1 !=  0:  # rotated x coordinate has decimal, get overlapping rxs
            rxs = [int(np.floor(rx)),int(np.ceil(rx))]  # we need int for indexing, float can't be used for indexing
            xdistances = [abs(rxs[0]-rx),abs(rxs[1]-rx)]
        else:
            rxs = [int(rx)]
            xdistances = [0]
        if ry % 1 !=  0:  # rotated y coordinate has decimal, get overlapping rys
            rys = [int(np.floor(ry)),int(np.ceil(ry))]
            ydistances = [abs(rys[0]-ry),abs(rys[1]-ry)]
        else:
            rys = [int(ry)]
            ydistances = [0]
        # pack rotated xs and distances
        x_.insert(0, cx); rxs_.insert(0, rxs); rys_.insert(0, rys)
        xdistances_.insert(0, xdistances); ydistances_.insert(0, ydistances)
        min_rx = min(rxs+[min_rx]); min_ry = min(rys+[min_ry])  # get min to scale negative to positive
        max_rx = max(rxs+[max_rx]); max_ry = max(rys+[max_ry])  # get max to init rdert_

    cx = xcenter
    while cx+1<xn and not mask_[cx-1] :  # scan right
        cx += 1;
        rx = (np.cos(angle) * (cx-xcenter)) + xcenter  # y part of equation is removed bcause y and ycenter is same line
        ry = (np.sin(angle) * (cx-xcenter)) + ycenter

        if rx % 1 !=  0:  # rotated x coordinate has decimal, get overlapping rxs
            rxs = [int(np.floor(rx)),int(np.ceil(rx))]
            xdistances = [abs(rxs[0]-rx),abs(rxs[1]-rx)]
        else:
            rxs = [int(rx)]
            xdistances = [0]
        if ry % 1 !=  0:  # rotated y coordinate has decimal, get overlapping rys
            rys = [int(np.floor(ry)),int(np.ceil(ry))]
            ydistances = [abs(rys[0]-ry),abs(rys[1]-ry)]
        else:
            rys = [int(ry)]
            ydistances = [0]
        # pack rotated xs and distances
        x_.append(cx); rxs_.append(rxs); rys_.append(rys)
        xdistances_.append(xdistances); ydistances_.append(ydistances)
        min_rx = min(rxs+[min_rx]); min_ry = min(rys+[min_ry])  # get min to scale negative to positive
        max_rx = max(rxs+[max_rx]); max_ry = max(rys+[max_ry])  # get max to init rdert_

    # we need 2 steps here because we need get all rxs and rys first before we can get the min and scale coordinate from negative to positive
    # scale coordinate from negative to positive
    for i, rxs in enumerate(rxs_):
        rxs_[i] = [rx-min_rx for rx in rxs]
    for i, rys in enumerate(rys_):
        rys_[i] = [ry-min_ry for ry in rys]

    max_rx -= min_rx; max_ry -= min_ry  # scale size of rdert__ and rmask__ too
    rxcenter = xcenter - min_rx; rycenter = ycenter - min_ry  # scale center coordinate
    ryn, rxn = max_ry+1, max_rx+1

    # init rotated rdert__ and rmask__
    rmask__ = np.full((ryn, rxn), fill_value=True, dtype="bool")
    rdert__t = [ np.zeros((ryn, rxn), dtype="uint8") for _ in range(nparams)]

    for x, rxs, rys, xdistances, ydistances in zip(x_, rxs_, rys_, xdistances_, ydistances_):
        # all mapped area's mask is False
        for ry, ydistance in zip(rys, ydistances):
            for rx, xdistance in zip(rxs, xdistances):
                # get rotated mask__
                rmask__[ry,rx] = False

                # distance ratio, using average of x and y distance
                # the higher distance, the lower ratio
                dratio = 1- ((xdistance + ydistance) / 2)

                # get rotated dert__
                for i in range(nparams):
                    rdert__t[i][ry, rx] += dert__t[i][ycenter, x] * dratio  # each value = dert value * dratio

    # below not fully updated
    rmask_ = rmask__[rycenter,:]
    rdert_ = []
    dy, dx = P.ptuple.angle[:]
    cx = rxcenter; cy = rycenter
    while cx-dx>0 and not rmask_[cx-dx]:  # scan left
        cx -= dx
        # compute next rx,ry, then mapped xs,ys, then fill params and mask
        rdert_.insert(0, [rdert__[cy,cx] for rdert__ in rdert__t] )  # pack left

    cx = rxcenter
    while cx+dx<rxn and not rmask_[cx+dx]:  # scan right
        cx += dx
        # compute next rx,ry, then mapped xs,ys, then fill params and mask
        rdert_.append([rdert__[cy,cx] for rdert__ in rdert__t])  # pack right

    # form P with new_dert_ and new_mask_ here, reuse from comp_slice?

def comp_plevels(_plevels, plevels, _fds, fds, derext):

    mplevel, dplevel = [],[]  # fd plevels, each cis+alt, same as new_caT
    mval, dval = 0,0  # m,d in new plevel, else c,a
    iVal = ave_G  # to start loop:

    for _plevel, plevel, _fd, fd in zip(reversed(_plevels), reversed(plevels), _fds, fds):  # caForks (caTree)
        if iVal < ave_G or _fd != fd:  # top-down, comp if higher plevels and fds match, same agg+
            break
        mTree, dTree = [],[]; mtval, dtval = 0,0  # Fork trees
        _caForks, _valt = _plevel; caForks, valt = plevel

        for _caFork, caFork in zip(_caForks, caForks):  # bottom-up alt+, pass-through fds
            mplayers, dplayers = [],[]; mlval, dlval = 0,0
            if _caFork and caFork:
                mplayer, dplayer = comp_players(_caFork, caFork, derext)
                mplayers += [mplayer]; dplayers += [dplayer]
                mlval += mplayer[1]; dlval += dplayer[1]
            else:
                mplayers += [[]]; dplayers += [[]]  # to align same-length trees for comp and sum
            # pack fds:
            mTree += [[mplayers, mlval, caFork[2]]]; dTree += [[dplayers, dlval, caFork[2]]]
            mtval += mlval; dtval += dlval
        # merge trees:
        mplevel += mTree; dplevel += mTree  # merge Trees in candidate plevels
        mval += mtval; dval += dtval
        iVal = mval+dval  # after 1st loop

    return [mplevel,mval], [dplevel,dval]  # always single new plevel


def comp_players(_caFork, caFork, derext):  # unpack and compare layers from der+

    mplayer, dplayer = [], []  # flat lists of ptuples, nesting decoded by mapping to lower levels
    mVal, dVal = 0,0  # m,d in new player, else c,a
    _players, _val, _fds =_caFork; players, val, fds = caFork

    for _player, player in zip(_players, players):
        mTree, dTree = [],[]; mtval, dtval = 0,0
        _caforks,_ = _player; caforks,_ = player

        for _cafork, cafork in zip(_caforks, caforks):  # bottom-up alt+, pass-through fds
            if _cafork and cafork:
                _ptuples,_ = _cafork; ptuples,_ = cafork  # no comp valt?
                if _ptuples and ptuples:
                    mtuples, dtuples = comp_ptuples(_ptuples, ptuples, _fds, fds, derext)
                    mTree += [mtuples]; dTree += [dtuples]
                    mtval += mtuples[1]; dtval += dtuples[1]
                else:
                    mTree += [[]]; dTree += [[]]
            else:
                mTree += [[]]; dTree += [[]]
        # merge Trees:
        mplayer += mTree; dplayer += dTree
        mVal += mtval; dVal += dtval

    return [mplayer,mVal], [dplayer,dVal]  # single new lplayer


def comp_ptuples(_Ptuples, Ptuples, _fds, fds, derext):  # unpack and compare der layers, if any from der+

    mPtuples, dPtuples = [[],0], [[],0]  # [list, val] each

    for _Ptuple, Ptuple, _fd, fd in zip(_Ptuples, Ptuples, _fds, fds):  # bottom-up der+, Ptuples per player, pass-through fds
        if _fd == fd:

            mtuple, dtuple = comp_ptuple(_Ptuple[0], Ptuple[0], daxis=derext[1][1])  # init Ptuple = ptuple, [[[[[[]]]]]]
            mext___, dext___ = [[],0], [[],0]
            for _ext__, ext__ in zip(_Ptuple[1][0], Ptuple[1][0]):  # ext__: extuple level
                mext__, dext__ = [[],0], [[],0]
                for _ext_, ext_ in zip(_ext__[0], ext__[0]):  # ext_: extuple layer
                    mext_, dext_ = [[],0], [[],0]
                    for _extuple, extuple in zip(_ext_[0], ext_[0]):  # loop ders from prior comps in each lower ext_
                        # + der extlayer:
                        mextuple, dextuple, meval, deval = comp_extuple(_extuple, extuple)
                        mext_[0] += [mextuple]; mext_[1] += meval
                        dext_[0] += [dextuple]; dext_[1] += deval
                    # + der extlevel:
                    mext__[0] += [mext_]; mext__[1] += mext_[1]
                    dext__[0] += [dext_]; mext__[1] += dext_[1]
                # + der inplayer:
                mext___[0] += [mext__]; mext___[1] += mext__[1]
                dext___[0] += [dext__]; dext___[1] += dext__[1]
            # + der Ptuple:
            mPtuples[0] += [[mtuple, mext___]]; mPtuples[1] += mtuple.val+mext___[1]
            dPtuples[0] += [[dtuple, dext___]]; dPtuples[1] += dtuple.val+dext___[1]
        else:
            break  # comp same fds

    mV = derext[2]; dV = derext[3]  # add der extset to all last elements:
    mext_[0] += [derext[0]]; mext_[1] += mV; mext__[1] += mV; mext___[1] += mV; mPtuples[1] += mV
    dext_[0] += [derext[1]]; dext_[1] += dV; dext__[1] += dV; dext___[1] += dV; dPtuples[1] += dV

    return mPtuples, dPtuples


def comp_extuple(_extuple, extuple):

    mval, dval = 0,0
    dextuple, mextuple = [],[]

    for _param, param, ave in zip(_extuple, extuple, (ave_distance, ave_dangle, ave_len, ave_sparsity)):
        # params: ddistance, dangle, dlen, dsparsity: all derived scalars
        d = _param-param;          dextuple += [d]; dval += d - ave
        m = min(_param,param)-ave; mextuple += [m]; mval += m

    return mextuple, dextuple, mval, dval
