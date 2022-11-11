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
