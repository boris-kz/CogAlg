'''
Agg_recursion eval and PP->graph conversion
'''
from .comp_slice import sum_derH
from .agg_recursion import Cgraph, agg_recursion, op_parH
from copy import copy, deepcopy
from .classes import CQ, Cptuple, CP, CderP, CPP
from .filters import PP_vars, PP_aves, ave_nsub, ave_agg

# move here temporary, for debug purpose
# not fully updated
def agg_recursion_eval(blob, PP_, fd):

    for i, PP in enumerate(PP_):
        converted_graph  = PP2graph(PP, fd=fd)  # convert PP to graph
        PP_[i] = converted_graph
        converted_blob = blob2graph(blob, fd=fd)  # convert root to graph

    Val = converted_blob.valt[fd]
    fork_rdnt = [1+(converted_blob.valt[fd] > converted_blob.valt[1-fd]), 1+(converted_blob.valt[1-fd] > converted_blob.valt[fd])]

    if (Val > PP_aves[fd] * ave_agg * (converted_blob.rdnt[fd]+1) * fork_rdnt[fd]) \
        and len(PP_) > ave_nsub : # and converted_blob[0].alt_rdn < ave_overlap:
        converted_blob.rdnt[fd] += 1  # estimate
        agg_recursion(converted_blob)

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
def blob2graph(blob, fseg, fd):

    PP_ = [blob.PPm_, blob.PPd_][fd]
    x0, xn, y0, yn = blob.box
    Graph = Cgraph(fds=copy(PP_[0].fds), rng=PP_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])

    [blob.mgraph, blob.dgraph][fd] = Graph  # update graph reference

    for i, PP in enumerate(PP_):
        graph = PP2graph(PP, fseg, fd)
        sum_derH(Graph.pH, graph.pH)
        graph.root = Graph
        Graph.node_ += [graph]


    return Graph


# tentative, will be finalized when structure in agg+ is finalized
def PP2graph(PP, fseg, ifd=1):


    box = [(PP.box[0]+PP.box[1]) /2, (PP.box[2]+PP.box[3]) /2] + PP.box
    graph = Cgraph(pH=deepcopy(PP.derH), valt=copy(PP.valt), rndt=copy(PP.rdnt), box=box)

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