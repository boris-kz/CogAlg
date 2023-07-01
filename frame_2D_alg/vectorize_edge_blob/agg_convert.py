'''
Agg_recursion eval and PP->graph conversion
'''

import numpy as np
from .agg_recursion import Cgraph, agg_recursion, op_parT
from copy import copy, deepcopy
from .classes import CP, CderP, CPP
from .filters import PP_vars, PP_aves, ave_nsub, ave_agg, med_decay
# move here temporary, for debug purpose
# not fully updated
def agg_recursion_eval(blob, PP_, fd):

    for i, PP in enumerate(PP_):
        converted_graph  = PP2graph(PP, 0, fd)  # convert PP to graph
        PP_[i] = converted_graph
        converted_blob = blob2graph(blob, 0, fd)  # convert root to graph

    Valt = [np.sum(converted_blob.valT[0]), np.sum(converted_blob.valT[1])]
    Rdnt = [np.sum(converted_blob.rdnT[0]), np.sum(converted_blob.rdnT[1])]
    fork_rdnt = [1+(Valt[fd] > Valt[1-fd]), 1+(Valt[1-fd] > Valt[fd])]

    if (Valt[fd] > PP_aves[fd] * ave_agg * (Rdnt[fd]+1) * fork_rdnt[fd]) \
        and len(PP_) > ave_nsub : # and converted_blob[0].alt_rdn < ave_overlap:
        rdn = converted_blob.rdnT[fd]
        while isinstance(rdn,list): rdn = rdn[0]
        rdn += 1
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
    Graph = Cgraph(fd=PP_[0].fd, rng=PP_[0].rng, box=[(y0+yn)/2,(x0+xn)/2, y0,yn, x0,xn])
    [blob.mgraph, blob.dgraph][fd] = Graph  # update graph reference

    for i, PP in enumerate(PP_):
        graph = PP2graph(PP, fseg, fd)
        op_parT(Graph, graph, fcomp=0)
        graph.root = Graph
        Graph.node_ += [graph]

    return Graph

# tentative, will be finalized when structure in agg+ is finalized
def PP2graph(PP, fseg, ifd=1):

    box = [(PP.box[0]+PP.box[1]) /2, (PP.box[2]+PP.box[3]) /2] + list(PP.box)
    # add nesting for subH and aggH:
    graph = Cgraph( parT=[[[copy(PP.derT[0])]],[[copy(PP.derT[1])]]],
                    valT=[[[copy(PP.valT[0])]],[[copy(PP.valT[1])]]], rdnT=[[[copy(PP.rdnT[0])]],[[copy(PP.rdnT[1])]]], box=box)
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

# temporary, not used here:
# _,_,med_valH = med_eval(link._P.link_t[fd], old_link_=[], med_valH=[], fd=fd)  # recursive += mediated link layer

def med_eval(last_link_, old_link_, med_valH, fd):  # recursive eval of mediated link layers, in form_graph only?

    curr_link_ = []; med_val = 0
    # compute med_valH, layer= val of links mediated by incremental number of nodes:

    for llink in last_link_:
        for _link in llink._P.link_t[fd]:
            if _link not in old_link_:  # not-circular link
                old_link_ += [_link]  # evaluated mediated links
                curr_link_ += [_link]  # current link layer,-> last_link_ in recursion
                med_val += np.sum(_link.valT[fd])
    med_val *= med_decay ** (len(med_valH) + 1)
    med_valH += [med_val]
    if med_val > aveB:
        # last med layer val-> likely next med layer val
        curr_link_, old_link_, med_valH = med_eval(curr_link_, old_link_, med_valH, fd)  # eval next med layer

    return curr_link_, old_link_, med_valH

# currently not used:

def sum_unpack(Q,q):  # recursive unpack of two pairs of nested sequences, to sum final ptuples

    Que,Val_,Rdn_ = Q; que,val_,rdn_ = q  # alternating rngH( derH( rngH... nesting, down to ptuple|val|rdn
    for i, (Ele,Val,Rdn, ele,val,rdn) in enumerate(zip_longest(Que,Val_,Rdn_, que,val_,rdn_, fillvalue=[])):
        if ele:
            if Ele:
                if isinstance(val,list):  # element is layer or fork
                    sum_unpack([Ele,Val,Rdn], [ele,val,rdn])
                else:  # ptuple
                    Val_[i] += val; Rdn_[i] += rdn
                    sum_ptuple(Ele, ele)
            else:
                Que += [deepcopy(ele)]; Val_+= [deepcopy(val)]; Rdn_+= [deepcopy(rdn)]

def comp_unpack(Que,que, rn):  # recursive unpack nested sequence to compare final ptuples

    DerT,ValT,RdnT = [[],[]],[[],[]],[[],[]]  # alternating rngH( derH( rngH.. nesting,-> ptuple|val|rdn

    for Ele,ele in zip_longest(Que,que, fillvalue=[]):
        if Ele and ele:
            if isinstance(Ele[0],list):
                derT,valT,rdnT = comp_unpack(Ele, ele, rn)
            else:
                # elements are ptuples
                mtuple, dtuple = comp_dtuple(Ele, ele, rn)  # accum rn across higher composition orders
                mval=sum(mtuple); dval=sum(dtuple)
                derT = [mtuple, dtuple]
                valT = [mval, dval]
                rdnT = [int(mval<dval),int(mval>=dval)]  # to use np.sum

            for i in 0,1:  # adds nesting per recursion
                DerT[i]+=[derT[i]]; ValT[i]+=[valT[i]]; RdnT[i]+=[rdnT[i]]

    return DerT,ValT,RdnT

def add_unpack(H, incr):  # recursive unpack hierarchy of unknown nesting to add input
    # new_H = []
    for i, e in enumerate(H):
        if isinstance(e,list):
            add_unpack(e,incr)
        else: H[i] += incr
    return H

def last_add(H, i):  # recursive unpack hierarchy of unknown nesting to add input
    while isinstance(H,list):
        H=H[-1]
    H+=i

def unpack(H):  # recursive unpack hierarchy of unknown nesting
    while isinstance(H,list):
        last_H = H
        H=H[-1]
    return last_H

