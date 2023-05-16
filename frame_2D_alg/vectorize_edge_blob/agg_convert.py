'''
Agg_recursion eval and PP->graph conversion
'''
from .agg_recursion import Cgraph, agg_recursion, op_parH
from copy import copy, deepcopy
from .classes import CQ, Cptuple, CP, CderP, CPP
from .filters import PP_vars, PP_aves, ave_nsub, ave_agg

# move here temporary, for debug purpose
# not fully updated
def agg_recursion_eval(blob, PP_, fd):

    fseg = isinstance(blob, CPP)
    for i, PP in enumerate(PP_):
        converted_graph  = PP2graph(PP, fseg=fseg, ifd=fd)  # convert PP to graph
        PP_[i] = converted_graph
    if fseg:
        converted_blob = PP2graph(blob, fseg=fseg, ifd=fd)  # convert root to graph (root default fd = 1?)
        for PP in PP_: PP.root = converted_blob
    else:
        converted_blob = blob2graph(blob, fseg=fseg)  # convert root to graph

    Val = converted_blob.valt[fd]
    fork_rdnt = [1+(converted_blob.valt[fd] > converted_blob.valt[1-fd]), 1+(converted_blob.valt[1-fd] > converted_blob.valt[fd])]

    if (Val > PP_aves[fd] * ave_agg * (converted_blob.rdnt[fd]+1) * fork_rdnt[fd]) \
        and len(PP_) > ave_nsub : # and converted_blob[0].alt_rdn < ave_overlap:
        converted_blob.rdnt[fd] += 1  # estimate
        agg_recursion(converted_blob, fseg=fseg)

# old
def frame2graph(frame, fseg, Cgraph):  # for frame_recursive

    mblob_ = frame.PPm_; dblob_ = frame.PPd_  # PPs are blobs here
    x0, xn, y0, yn = frame.box
    gframe = Cgraph(alt_plevels=CpH, rng=mblob_[0].rng, rdn=frame.rdn, x0=(x0+xn)/2, xn=(xn-x0)/2, y0=(y0+yn)/2, yn=(yn-y0)/2)
    for fd, blob_, plevels in enumerate(zip([mblob_,dblob_], [gframe.plevels, gframe.alt_plevels])):
        graph_ = []
        for blob in blob_:
            graph = PP2graph(blob, fseg, Cgraph, fd)
            sum_pH(plevels, graph.plevels)
            graph_ += [graph]
        [gframe.node_.Q, gframe.alt_graph_][fd][:] = graph_  # mblob_|dblob_, [:] to enable to left hand assignment, not valid for object

    return gframe

# tentative, will be finalized when structure in agg+ is finalized
def blob2graph(blob, fseg):

    PPm_ = blob.PPm_; PPd_ = blob.PPd_
    x0, xn, y0, yn = blob.box

    alt_mblob = Cgraph(fds=copy(PPm_[0].fds), aggH=CQ(Qd=[CQ(Qd=[])]), rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    alt_dblob = Cgraph(fds=copy(PPd_[0].fds), aggH=CQ(Qd=[CQ(Qd=[])]), rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])

    mblob = Cgraph(fds=copy(PPm_[0].fds), aggH=CQ(Qd=[CQ(Qd=[])]), alt_Graph=alt_mblob, rng=PPm_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    dblob = Cgraph(fds=copy(PPd_[0].fds), aggH=CQ(Qd=[CQ(Qd=[])]), alt_Graph=alt_dblob, rng=PPd_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])

    blob.mgraph = mblob  # update graph reference
    blob.dgraph = dblob  # update graph reference
    blobs= [mblob, dblob]

    for fd, PP_ in enumerate([PPm_,PPd_]):  # if any
        for i, PP in enumerate(PP_):
            graph = PP2graph(PP, fseg, fd)
            if i: op_parH(blobs[fd].pH, graph.pH, fcomp=0)
            else: blobs[fd].pH = deepcopy(graph.pH)
            graph.root = blobs[fd]
            blobs[fd].node_ += [graph]

    for alt_blob in blob.adj_blobs[0]:  # adj_blobs = [blobs, pose]
        if not alt_blob.mgraph:
            blob2graph(alt_blob, fseg)  # convert alt_blob to graph
        if alt_mblob.pH.Q: op_parH(alt_mblob.pH, alt_blob.mgraph.pH, fcomp=0)
        else:              alt_mblob.pH = deepcopy(alt_blob.mgraph.pH)
        if alt_dblob.pH.Q: op_parH(alt_dblob.pH, alt_blob.dgraph.pH, fcomp=0)
        else:              alt_dblob.pH = deepcopy(alt_blob.dgraph.pH)

    return mblob, dblob


# tentative, will be finalized when structure in agg+ is finalized
def PP2graph(PP, fseg, ifd=1):

    alt_derH = CQ()
    alt_subH = CQ(Qd=[alt_derH],Q=[0], fds=[0]); alt_aggH = CQ(Qd=[alt_subH], Q=[0], fds=[1]); alt_valt = [0,0]; alt_rdnt = [0,0]; alt_box = [0,0,0,0]
    if not fseg and PP.alt_PP_:  # seg doesn't have alt_PP_
        pQd = deepcopy(PP.alt_PP_[0].derH); alt_valt = copy(PP.alt_PP_[0].valt)
        alt_box = copy(PP.alt_PP_[0].box); alt_rdnt = copy(PP.alt_PP_[0].rdnt)
        for altPP in PP.alt_PP_[1:]:  # get fd sequence common for all altPPs:
            sum_derH(pQd, altPP.derH)
            Y0,Yn,X0,Xn = alt_box; y0,yn,x0,xn = altPP.box
            alt_box[:] = min(Y0,y0),max(Yn,yn),min(X0,x0),max(Xn,xn)
            for i in 0,1:
                alt_valt[i] += altPP.valt[i]
                alt_rdnt[i] += altPP.rdnt[i]
        for dderH in pQd:
            if isinstance(dderH, Cptuple):  # convert ptuple to CQ
                pQ = CQ(n=dderH.n)
                for pname in pnames:
                    par = getattr(dderH, pname)
                    if pname != "x":  # x is in box
                        pQ.Qd += [par]; pQ.Q += [0]
                        if pname not in ["I", "angle", "aangle", "axis"]:
                            pQ.valt[1] += par
                alt_derH.Qd += [pQ]
            else:  # vertuple
                QdderH = deepcopy(dderH)
                QdderH.Qm.pop(-2); QdderH.Qd.pop(-2); QdderH.Q.pop(-2)  # remove x from existing vertuple
                alt_derH.Qd += [QdderH]
            alt_derH.Q += [0]; alt_derH.fds += [1]
    alt_box = [(alt_box[0]+alt_box[1]) /2, (alt_box[2]+alt_box[3]) /2] + alt_box
    alt_Graph = Cgraph(pH=alt_aggH, valt=alt_valt, rdnt=alt_rdnt, box=alt_box)

    Qd = []; Q = []; fds = []
    for dderH in PP.derH:
        if isinstance(dderH, Cptuple):  # convert ptuple to CQ
            pQ = CQ(n=dderH.n)
            for pname in pnames:
                par = getattr(dderH, pname)
                if pname != "x":  # x is in box
                    pQ.Qd += [par]; pQ.Q += [0]  # Qm is just filler, else we need to check if they are empty before summing them
                    if pname not in ["I", "angle", "aangle", "axis"]:
                        pQ.valt[1] += par
            Qd += [pQ]
        else:  # vertuple
            QdderH = deepcopy(dderH)
            QdderH.Qm.pop(-2); QdderH.Qd.pop(-2); QdderH.Q.pop(-2)  # remove x from existing vertuple
            Qd += [QdderH]
        Q += [0]; fds += [1]

    derH = CQ(Qd=Qd, Q=Q, fds=fds)
    subH = CQ(Qd=[derH],Q=[0], fds=[0]); aggH = CQ(Qd=[subH], Q=[0], fds=[1])

    box = [(PP.box[0]+PP.box[1]) /2, (PP.box[2]+PP.box[3]) /2] + PP.box
    graph = Cgraph(pH=aggH, valt=copy(PP.valt), rndt=copy(PP.rdnt), box=box, alt_Graph=alt_Graph)

    return graph

# drafts:
def inpack_derH(pPP, ptuples, idx_=[]):  # pack ptuple vars in derH of pPP vars, converting macro derH -> micro derH
    # idx_: indices per lev order in derH, lenlev: 1, 1, 2, 4, 8...

    repack(pPP, ptuples[0], idx_+[0])  # single-element 1st lev
    if len(ptuples)>1:
        repack(pPP, ptuples[1], idx_+[1])  # single-element 2nd lev
        i=2; last=4
        idx = 2  # init incremental elevation = i
        while last<=len(ptuples):
            lev = ptuples[i:last]  # lev is nested, max len_sublev = lenlev-1, etc.
            inpack_derH(pPP, lev, idx_+[idx])  # add idx per sublev
            i=last; last+=i  # last=i*2
            idx+=1  # elevation in derH

def repack(pPP, ptuple, idx_):  # pack derH in elements of iderH

    for i, param_name in enumerate(PP_vars):
        par = getattr(ptuple, param_name)
        Par = pPP[i]
        if len(Par) > len(idx_):  # Par is derH of pars
            Par[-1] += [par]  # pack par in top lev of Par, added per inpack_derH recursion
        else:
            Par += [[par]]  # add new Par lev, implicitly nested in ptuples?