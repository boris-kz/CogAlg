
from __future__ import annotations
import sys
from collections import deque
from numbers import Real
from typing import Any, NamedTuple, Tuple
from time import time
from copy import copy
import numpy as np
from visualization.draw_frame import visualize
from class_cluster import CBase, init_param as z
from utils import kernel_slice_3x3 as ks    # use in comp_pixel
# from vectorize_edge.classes import Ct
# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED = -2

class Cder__t(NamedTuple):
    dy: Any
    dx: Any
    g: Any
    def get_pixel(self, y: Real, x: Real) -> Tuple[Real, Real, Real]:
        return self.dy[y, x], self.dx[y, x], self.g[y, x]

class Cbox(NamedTuple):
    n: Real
    w: Real
    s: Real
    e: Real
    # properties
    @property
    def cy(self) -> Real: return (self.n + self.s) / 2
    @property
    def cx(self) -> Real: return (self.w + self.e) / 2
    @property
    def slice(self) -> Tuple[slice, slice]: return slice(self.n, self.s), slice(self.w, self.e)
    # operators:
    def extend(self, other: Cbox) -> Cbox:
        """Add 2 boxes."""
        return Cbox(min(self.n, other.n), min(self.w, other.w), max(self.s, other.s), max(self.e, other.e))

    # methods
    def __add__(self, other): return [a+b for a,b in zip(self, other)] if self else copy(other)
    def __sub__(self, other): return [a-b for a,b in zip(self, other)] if self else copy(other)

    def accumulate(self, y: Real, x: Real) -> Cbox:
        """Box coordinate accumulation."""
        return Cbox(min(self.n, y), min(self.w, x), max(self.s, y + 1), max(self.e, x + 1))

    def expand(self, r: int, h: Real, w: Real) -> Cbox:
        """Box expansion by margin r."""
        return Cbox(max(0, self.n - r), max(0, self.w - r), min(h, self.s + r), min(w, self.e + r))

    def shrink(self, r: int) -> Cbox:
        """Box shrink by margin r."""
        return Cbox(self.n + r, self.w + r, self.s - r, self.e - r)

    def sub_box2box(self, sb: Cbox) -> Cbox:
        """sub_box to box transform."""
        return Cbox(self.n + sb.n, self.w + sb.w, sb.s + self.n, sb.e + self.w)

    def box2sub_box(self, b: Cbox) -> Cbox:
        """box to sub_box transform."""
        return Cbox(b.n - self.n, b.w - self.w, b.s - self.n, b.e - self.w)

class CBlob(CBase):
    # comp_pixel:
    sign : bool = None
    I : float = 0.0
    Dy : float = 0.0
    Dx : float = 0.0
    G : float = 0.0
    A : float = 0.0 # blob area
    # composite params:
    M : float = 0.0 # summed PP.M, for both types of recursion?
    box : Cbox = Cbox(0, 0, 0, 0)  # n,w,s,e
    ibox : Cbox = Cbox(0, 0, 0, 0) # box for i__
    mask__ : object = None
    i__ : object = None     # reference to original input (no shrinking)
    der__t : Cder__t = None   # tuple of derivatives arrays, consistent in shape
    adj_blobs : list = z([])  # adjacent blobs
    fopen : bool = False
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_dx:
    Mdx : float = 0.0
    Ddx : float = 0.0
    # derivation hierarchy:
    root_ibox : Cbox = Cbox(0, 0, 0, 0)  # from root blob
    root_der__t : list = z([])  # from root blob
    prior_forks : str = ''
    fBa : bool = False  # in root_blob: next fork is comp angle, else comp_r
    rdn : float = 1.0  # redundancy to higher blob layers, or combined?
    rng : int = 1  # comp range, set before intra_comp
    rlayers : list = z([])  # list of layers across sub_blob derivation tree, deeper layers are nested with both forks
    dlayers : list = z([])  # separate for range and angle forks per blob
    PPm_ : list = z([])  # mblobs in frame
    PPd_ : list = z([])  # dblobs in frame
    valt : list = z([])  # PPm_ val, PPd_ val, += M,G?
    fsliced : bool = False  # from comp_slice
    root : object = None  # frame or from frame_bblob
    mgraph : object = None  # reference to converted blob
    dgraph : object = None  # reference to converted blob

'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

def frame_blobs_root(i__, intra=False, render=False, verbose=False):

    if verbose: start_time = time()
    Y, X = i__.shape[:2]
    der__t = comp_pixel(i__)
    sign__ = ave - der__t.g > 0   # sign is positive for below-average g
    frame = CBlob(i__=i__, der__t=der__t, box=Cbox(0, 0, Y, X), rlayers=[[]])
    fork_data = '', Cbox(1, 1, Y - 1, X - 1), der__t, sign__, None  # fork, fork_ibox, der__t, sign__, mask__
    # https://en.wikipedia.org/wiki/Flood_fill:
    frame.rlayers[0], idmap, adj_pairs = flood_fill(frame, fork_data, verbose=verbose)
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    for blob in frame.rlayers[0]:
        frame.accumulate(I=blob.I, Dy=blob.Dy, Dx=blob.Dx)
    # dlayers = []: no comp_a yet
    if verbose: print(f"{len(frame.rlayers[0])} blobs formed in {time() - start_time} seconds")

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frame.rlayers += intra_blob_root(frame, render, verbose)  # recursive eval cross-comp range| angle| slice per blob
        if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks

    if render: visualize(frame)
    return frame


def comp_pixel(pi__):
    # compute directional derivatives:
    dy__ = (
        (pi__[ks.bl] - pi__[ks.tr]) * 0.25 +
        (pi__[ks.bc] - pi__[ks.tc]) * 0.50 +
        (pi__[ks.br] - pi__[ks.tl]) * 0.25
    )
    dx__ = (
        (pi__[ks.tr] - pi__[ks.bl]) * 0.25 +
        (pi__[ks.mr] - pi__[ks.mc]) * 0.50 +
        (pi__[ks.br] - pi__[ks.tl]) * 0.25
    )
    G__ = np.hypot(dy__, dx__)                          # compute gradient magnitude

    return Cder__t(dy__, dx__, G__)


def flood_fill(root_blob, fork_data, verbose=False):
    # unpack and derive required fork data
    fork, fork_ibox, der__t, sign__, mask__ = fork_data
    height, width = der__t.g.shape  # = der__t shape
    fork_i__ = root_blob.i__[fork_ibox.slice]
    assert height, width == fork_i__.shape  # same shape as der__t

    idmap = np.full((height, width), UNFILLED, 'int32')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None: idmap[~mask__] = EXCLUDED
    if verbose:
        n_pixels = (height*width) if mask__ is None else mask__.sum()
        step = 100 / n_pixels  # progress % percent per pixel
        progress = 0.0; print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()
    blob_ = []
    adj_pairs = set()

    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts
                blob = CBlob(
                    i__=root_blob.i__, sign=sign__[y, x], root_ibox=fork_ibox, root_der__t=der__t,
                    box=Cbox(y, x, y + 1, x + 1), rng=root_blob.rng, prior_forks=root_blob.prior_forks + fork)
                blob_ += [blob]
                idmap[y, x] = blob.id
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()
                    # add dert to blob
                    blob.accumulate(I  = fork_i__[y1][x1],
                                    Dy = der__t.dy[y1][x1],
                                    Dx = der__t.dx[y1][x1])
                    blob.A += 1
                    blob.box = blob.box.accumulate(y1, x1)
                    # neighbors coordinates, 4 for -, 8 for +
                    if blob.sign:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]
                    # search neighboring derts:
                    for y2, x2 in adj_dert_coords:
                        # image boundary is reached:
                        if (y2 < 0 or y2 >= height or x2 < 0 or x2 >= width or
                                idmap[y2, x2] == EXCLUDED):
                            blob.fopen = True
                        # pixel is filled:
                        elif idmap[y2, x2] == UNFILLED:
                            # same-sign dert:
                            if blob.sign == sign__[y2, x2]:
                                idmap[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts += [(y2, x2)]
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))  # blob.id always increases
                # terminate blob
                blob.ibox = fork_ibox.sub_box2box(blob.box)
                blob.der__t = Cder__t(
                    *(par__[blob.box.slice] for par__ in der__t))
                blob.mask__ = (idmap[blob.box.slice] == blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = np.hypot(blob.Dy, blob.Dx)
                if verbose:
                    progress += blob.A * step; print(f"\rClustering... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    return blob_, idmap, adj_pairs


def assign_adjacents(adj_pairs):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in adj_pairs:
        blob1 = CBlob.get_instance(blob_id1)
        blob2 = CBlob.get_instance(blob_id2)

        y01, yn1, x01, xn1 = blob1.box
        y02, yn2, x02, xn2 = blob2.box

        if blob1.fopen and blob2.fopen:
            pose1 = pose2 = 2
        elif y01 < y02 and x01 < x02 and yn1 > yn2 and xn1 > xn2:
            pose1, pose2 = 0, 1  # 0: internal, 1: external
        elif y01 > y02 and x01 > x02 and yn1 < yn2 and xn1 < xn2:
            pose1, pose2 = 1, 0  # 1: external, 0: internal
        else:
            if blob2.A > blob1.A:
                pose1, pose2 = 0, 1  # 0: internal, 1: external
            else:
                pose1, pose2 = 1, 0  # 1: external, 0: internal
        # bilateral assignments
        '''
        if f_segment_by_direction:  # pose is not needed
            blob1.adj_blobs += [blob2]
            blob2.adj_blobs += [blob1]
        '''
        blob1.adj_blobs[0] += [blob2]
        blob1.adj_blobs[1] += [pose2]
        blob2.adj_blobs[0] += [blob1]
        blob2.adj_blobs[1] += [pose1]


if __name__ == "__main__":
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=0)
    argument_parser.add_argument('-c', '--clib', help='use C shared library', type=int, default=0)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=1)
    argument_parser.add_argument('-e', '--extra', help='run frame_recursive after frame_blobs', type=int, default=0)
    args = argument_parser.parse_args()
    image = imread(args.image)
    verbose = args.verbose
    intra = args.intra
    render = args.render

    start_time = time()
    if args.extra:  # not functional yet
        from frame_recursive import frame_recursive
        frame = frame_recursive(image, intra, render, verbose)
    else:
        frame = frame_blobs_root(image, intra, render, verbose)

    end_time = time() - start_time
    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)


def comp_G(link, Et):

    _G, G = link._G, link.G
    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0, 1,1, 0,0
    # keep separate P ptuple and PP derH, empty derH in single-P G, + empty aggH in single-PP G:

    # / P:
    mtuple, dtuple, Mtuple, Dtuple = comp_ptuple(_G.ptuple, G.ptuple, rn=1, fagg=1)
    mval, dval = sum(mtuple), sum(abs(d) for d in dtuple)  # mval is signed, m=-min in comp x sign
    mrdn = dval>mval; drdn = dval<=mval
    dect = [0,0]
    for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
        for i, (par, max, ave) in enumerate(zip(ptuple, Ptuple, aves)):
            # compute link decay coef: par/ max(self/same)
            if fd: dect[fd] += abs(par)/ abs(max) if max else 1
            else:  dect[fd] += (par+ave)/ (max+ave) if max else 1
    dertv = [[mtuple,dtuple], [mval,dval],[mrdn,drdn],[dect[0]/6,dect[1]/6]]
    Mval+=mval; Dval+=dval; Mrdn+=mrdn; Drdn+=drdn; Mdec+=dect[0]/6; Ddec+=dect[1]/6  # ave of 6 params

    # / PP:
    dderH = []
    if _G.derH and _G.derH:  # empty in single-P Gs?
        for _lay, lay in zip(_G.derH,_G.derH):
            mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay[1], lay[1], rn=1, fagg=1)
            mval = sum(mtuple); dval = sum(abs(d) for d in dtuple)
            mrdn = dval > mval; drdn = dval < mval
            mdec, ddec = 0, 0
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: ddec += abs(par)/ abs(max) if max else 1
                    else:   mdec += (par+ave)/ (max+ave) if max else 1
            mdec /= 6; ddec /= 6
            Mval+=dval; Dval+=mval; Mdec=(Mdec+mdec)/2; Ddec=(Ddec+ddec)/2
            dderH += [[[mtuple,dtuple], [mval,dval],[mrdn,drdn],[mdec,ddec]]]

    # / G:
    der_ext = comp_ext([_G.L,_G.S,_G.A],[G.L,G.S,G.A], [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec])
    dderH = [[dertv]+dderH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec], der_ext]

    if _G.aggH and G.aggH:
        daggH, valt,rdnt,dect = comp_aggHv(_G.aggH, G.aggH, rn=1) if G.fHH else comp_subHv(_G.aggH, G.aggH, rn=1)
        # aggH is subH before agg+
        mval,dval = valt; Mval+=dval; Dval+=mval
        Mrdn += rdnt[0] + dval>mval; Drdn += rdnt[1] + dval<=mval
        Mdec = (Mdec+dect[0])/2; Ddec = (Ddec+dect[1])/2
        # flat, appendleft:
        daggH = [[dderH]+daggH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]]
    else:
        daggH = dderH
    link.daggH += [daggH]

    link.Vt,link.Rt,link.Dt = Valt,Rdnt,Dect = [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # reset per comp_G

    for fd, (Val,Rdn,Dec) in enumerate(zip(Valt,Rdnt,Dect)):
        if Val > G_aves[fd] * Rdn:
            # to eval fork grapht in form_graph_t:
            Et[0][fd] += Val; Et[1][fd] += Rdn; Et[2][fd] += Dec
            G.Vt[fd] += Val;  G.Rt[fd] += Rdn;  G.Dt[fd] += Dec
            if not fd:
                for G in link.G, link._G:
                    rimH = G.rimH
                    if rimH and isinstance(rimH[0],list):  # rim is converted to rimH in 1st sub+
                        if len(rimH) == len(G.RimH): rimH += [[]]  # no new rim layer yet
                        rimH[-1] += [link]  # rimH
                    else:
                        rimH += [link]  # rim


def comp_aggHv(_aggH, aggH, rn):  # no separate ext

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    SubH = []

    for _lev, lev in zip(_aggH, aggH):  # compare common subHs, if lower-der match?
        if _lev and lev:

            if len(lev) == 5:  # derHv with additional ext
                dderH, valt,rdnt,dect, dextt = comp_derHv(_lev,lev, rn)
                SubH += [[dderH, valt,rdnt,dect,dextt]]
            else:
                dsubH, valt,rdnt,dect = comp_subHv(_lev[0],lev[0], rn)
                SubH += dsubH  # concat
            Mdec += dect[0]; Ddec += dect[1]
            mval,dval = valt; Mval += mval; Dval += dval
            Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + mval <= dval
    if SubH:
        S = min(len(_aggH),len(aggH)); Mdec/= S; Ddec /= S  # normalize

    return SubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]


def comp_subHv(_subH, subH, rn):

    Mval,Dval, Mrdn,Drdn, Mdec,Ddec = 0,0,1,1,0,0
    dsubH =[]

    for _lay, lay in zip(_subH, subH):  # compare common lower layer|sublayer derHs, if prior match?

        dderH, valt,rdnt,dect, dextt = comp_derHv(_lay,lay, rn)  # derHv: [derH, valt, rdnt, dect, extt]:
        dsubH += [[dderH, valt,rdnt,dect,dextt]]  # flat
        Mdec += dect[0]; Ddec += dect[1]
        mval,dval = valt; Mval += mval; Dval += dval
        Mrdn += rdnt[0] + dval > mval; Drdn += rdnt[1] + dval <= mval
    if dsubH:
        S = min(len(_subH),len(subH)); Mdec/= S; Ddec /= S  # normalize

    return dsubH, [Mval,Dval],[Mrdn,Drdn],[Mdec,Ddec]  # new layer,= 1/2 combined derH


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_ = T
            derH, valt, rdnt, dect, extt_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)],
                ]
                for [Tuplet,Valt,Rdnt,Dect], [tuplet,valt,rdnt,dect]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0)])
            ]
        else:
            T[:] = deepcopy(t)

def sum_ext(Extt, extt):

    if isinstance(Extt[0], list):
        for Ext,ext in zip(Extt,extt):  # ext: m,d tuple
            for i,(Par,par) in enumerate(zip(Ext,ext)):
                Ext[i] = Par+par
    else:  # single ext
        for i in range(3): Extt[i]+=extt[i]  # sum L,S,A


def comp_ext(_ext, ext, Valt, Rdnt, Dect):  # comp ds:

    (_L,_S,_A),(L,S,A) = _ext,ext
    dL = _L-L

    if isinstance(A,list):
        if any(A) and any(_A):
            mA,dA = comp_angle(_A,A); adA=dA
        else:
            mA,dA,adA = 0,0,0
        max_mA = max_dA = .5  # = ave_dangle
        dS = _S/_L - S/L  # S is summed over L, dS is not summed over dL
    else:
        mA = get_match(_A,A)- ave_dangle; dA = _A-A; adA = abs(dA); _aA=abs(_A); aA=abs(A)
        max_dA = _aA + aA; max_mA = max(_aA, aA)
        dS = _S - S
    mL = get_match(_L,L) - ave_mL
    mS = get_match(_S,S) - ave_mL

    m = mL+mS+mA; d = abs(dL)+ abs(dS)+ adA
    Valt[0] += m; Valt[1] += d
    Rdnt[0] += d>m; Rdnt[1] += d<=m
    _aL = abs(_L); aL = abs(L)
    _aS = abs(_S); aS = abs(S)

    # ave dec = ave (ave dec, ave L,S,A dec):
    Dect[0] = ((mL / max(aL,_aL) if aL or _aL else 1 +
                mS / max(aS,_aS) if aS or _aS else 1 +
                mA / max_mA if max_mA else 1) /3
                + Dect[0]) / 2
    Dect[1] = ((dL / (_aL+aL) if aL+_aL else 1 +
                dS / (_aS+aS) if aS+_aS else 1 +
                dA / max_dA if max_mA else 1) /3
                + Dect[1]) / 2

    return [[mL,mS,mA], [dL,dS,dA]]

'''
Each call to comp_rng | comp_der forms dderH: new layer of derH. Both forks are merged in feedback to contain complexity
(deeper layers are appended by feedback, if nested we need fback_tree: last_layer_nforks = 2^n_higher_layers)
'''
def sub_recursion(root, PP, fd):  # called in form_PP_, evaluate PP for rng+ and der+, add layers to select sub_PPs
    rng = PP.rng+(1-fd)

    if fd: link_ = comp_der(PP.link_)  # der+: reform old links
    else:  link_ = comp_rng(PP.link_, rng)  # rng+: form new links, should be recursive?

    rdn = (PP.valt[fd] - PP_aves[fd] * PP.rdnt[fd]) > (PP.valt[1-fd] - PP_aves[1-fd] * PP.rdnt[1-fd])
    PP.rdnt += (0, rdn) if fd else (rdn, 0)     # a little cumbersome, will be revised later
    for P in PP.P_: P.root = [None,None]  # fill with sub_PPm_,sub_PPd_ between nodes and PP:

    form_PP_t(PP, link_, base_rdn=PP.rdnt[fd])
    root.fback_t[fd] += [[PP.derH, PP.valt, PP.rdnt]]  # merge in root.fback_t fork, else need fback_tree


def comp_P_(edge: Cgraph, adj_Pt_: List[Tuple[CP, CP]]):  # cross-comp P_ in edge: high-gradient blob, sliced in Ps in the direction of G

    for _P, P in adj_Pt_:  # scan, comp contiguously uplinked Ps, rn: relative weight of comparand
        # initial comp is rng+
        distance = np.hypot(_P.yx[1]-P.yx[1],_P.yx[0]-P.yx[0])
        comp_P(edge.link_, _P, P, rn=len(_P.dert_)/len(P.dert_), derP=distance, fd=0)

    form_PP_t(edge, edge.link_, base_rdn=2)

def comp_P(link_, _P, P, rn, derP=None, S=None, V=0):  #  derP if der+, S if rng+

    if derP is not None:  # der+: extend in-link derH, in sub+ only
        dderH, valt, rdnt = comp_derH(_P.derH, P.derH, rn=rn)  # += fork rdn
        derH = derP.derH | dderH; S = derP.S  # dderH valt,rdnt for new link
        aveP = P_aves[1]

    elif S is not None:  # rng+: add derH
        mtuple, dtuple = comp_ptuple(_P.ptuple, P.ptuple, rn, fagg=0)
        valt = Cmd(sum(mtuple), sum(abs(d) for d in dtuple))
        rdnt = Cmd(1+(valt.d>valt.m), 1+(1-(valt.d>valt.m)))   # or rdn = Dval/Mval?
        derH = CderH([Cmd(mtuple, dtuple)])
        aveP = P_aves[0]
        V += valt[0] + sum(mtuple)
    else:
        raise ValueError("either derP (der+) or S (rng+) should be specified")

    A = Cangle(*(_P.yx - P.yx))
    derP = CderP(derH=derH, valt=valt, rdnt=rdnt, P=P,_P=_P, S=S, A=A)

    if valt.m > aveP*rdnt.m or valt.d > aveP*rdnt.d:
        link_ += [derP]

    return V


def form_PP_t(root, root_link_, base_rdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = defaultdict(list)
        derP_ = []
        for derP in root_link_:
            if derP.valt[fd] > P_aves[fd] * derP.rdnt[fd]:
                P_Ps[derP.P] += [derP._P]  # key:P, vals:linked _Ps, up and down
                P_Ps[derP._P] += [derP.P]
                derP_ += [derP]  # filtered derP
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(P_Ps[P])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                cP_ += [_P]
                perimeter += P_Ps[_P]  # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, derP_, base_rdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for fd, PP_ in enumerate(PP_t):  # after form_PP_t -> P.roott
        for PP in PP_:
            if PP.valt[1] * len(PP.link_) > PP_aves[1] * PP.rdnt[1]:  # val*len*rng: sum ave matches - fixed PP cost
                der_recursion(root, PP, fd)  # eval rng+/PPm or der+/PPd
        if root.fback_t and root.fback_t[fd]:
            feedback(root, fd)  # after sub+ in all nodes, no single node feedback up multiple layers

    root.node_ = PP_t  # nested in sub+, add_alt_PPs_?

def sum_aggH(AggH, aggH, base_rdn):

    if aggH:
        if AggH:
            for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                if layer[-1] == 1:  # derHv with depth == 1
                    sum_derHv(Layer, layer, base_rdn)
                else:  # subHv
                    sum_subHv(Layer, layer, base_rdn)
        else:
            AggH[:] = deepcopy(aggH)


def sum_aggHv(T, t, base_rdn):

    if t:
        if T:
            AggH,Valt,Rdnt,Dect,_ = T; aggH,valt,rdnt,dect,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if AggH:
                for Layer, layer in zip_longest(AggH,aggH, fillvalue=[]):
                    if layer[-1] == 1:  # derHv with depth == 1
                        sum_derHv(Layer, layer, base_rdn)
                    else:  # subHv
                        sum_subHv(Layer, layer, base_rdn)
            else:
                AggH[:] = deepcopy(aggH)
            sum_ext(Ext,ext)
        else:
           T[:] = deepcopy(t)


def sum_subHv(T, t, base_rdn, fneg=0):

    if t:
        if T:
            SubH,Valt,Rdnt,Dect,Ext,_ = T; subH,valt,rdnt,dect,ext,_ = t
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i]+dect[i])/2
            if SubH:
                for Layer, layer in zip_longest(SubH,subH, fillvalue=[]):
                    sum_derHv(Layer, layer, base_rdn, fneg)  # _lay[0][0] is mL
                    sum_ext(Layer[-2], layer[-2])
            else:
                SubH[:] = deepcopy(subH)
            sum_ext(Ext,ext)
        else:
            T[:] = deepcopy(t)


def sum_derHv(T,t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    if t:
        if T:
            DerH, Valt, Rdnt, Dect, Extt_,_ = T; derH, valt, rdnt, dect, extt_,_ = t
            for Extt, extt in zip(Extt_,extt_):
                sum_ext(Extt, extt)
            for i in 0,1:
                Valt[i] += valt[i]; Rdnt[i] += rdnt[i]+base_rdn; Dect[i] = (Dect[i] + dect[i])/2
            DerH[:] = [
                [[sum_dertuple(Dertuple,dertuple, fneg*i) for i,(Dertuple,dertuple) in enumerate(zip(Tuplet,tuplet))],
                  [V+v for V,v in zip(Valt,valt)], [R+r+base_rdn for R,r in zip(Rdnt,rdnt)], [(D+d)/2 for D,d in zip(Dect,dect)], 0
                ]
                for [Tuplet,Valt,Rdnt,Dect,_], [tuplet,valt,rdnt,dect,_]  # ptuple_tv
                in zip_longest(DerH, derH, fillvalue=[([0,0,0,0,0,0],[0,0,0,0,0,0]), (0,0),(0,0),(0,0),0])
            ]
            sum_ext(Extt_, extt_)

        else:
            T[:] = deepcopy(t)

def rng_recursion(PP, rng=1):  # similar to agg+ rng_recursion, but contiguously link mediated

    _link_ = PP.link_  # init proto-links = [_P,P] in der0

    while True:  # form new links with recursive rng+ in edge|PP, secondary pair comp eval
        link_ = []
        V = 0
        for link in _link_:
            V = comp_P(link_,link, V=V) # link layer match
        PP.link_ += link_  # rng+/ last der+, or link_ should be mlink_ only?

        if V >= ave * len(link_) * 6:  # len mtuple
            rng += 1
            for _derP, derP in combinations(link_, 2):  # scan new link pairs
                # or trace through P.uplink_?
                _P = _derP.P; P = derP.P
                if _derP.P is not derP._P:  # same as derP._P is _derP._P or derP.P is _derP.P
                    continue
                __P = _derP._P  # next layer of Ps
                if len(__P.derH) < len(P.derH):  # for call from der+: compare same der layers only
                    continue
                distance = np.hypot(*(__P.yx - P.yx))  # distance between P midpoints, /= L for eval?
                if rng-1 < distance <= rng:
                    if P.valt[0]+__P.valt[0] > ave * (P.rdnt[0]+_P.rdnt[0]):
                        link_ += [CderP(_P=__P, P=P, S=distance, A=Cangle(*(_P.yx-P.yx)))]
            _link_ = link_
        else:
            break
    PP.rng=rng

class Ct(list):     # tuple operations

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.extend(kwargs.values())

    def __abs__(self): return hypot(self.p1, self.p1)
    def __pos__(self): return self
    def __neg__(self): return self.__class__(-self.p1, -self.p2, -self.p3)

    def __add__(self, other): return Ct([a+b for a,b in zip(self, other)]) if self else copy(other)
    def __sub__(self, other): return Ct([a-b for a,b in zip(self, other)]) if self else copy(other)

    def normalize(self):
        dist = abs(self)
        return self.__class__(self.p1 / dist, self.p2 / dist)

class Cangle(NamedTuple):  # not sure it's needed

    dy: Real
    dx: Real
    # operators:
    def __abs__(self): return hypot(self.dy, self.dx)
    def __pos__(self): return self
    def __neg__(self): return self.__class__(-self.dy, -self.dx)
    def __add__(self, other): return self + other  # .__class__(self.dy + other.dy, self.dx + other.dx) ?
    def __sub__(self, other): return self + (-other)


def sum_derH(T, t, base_rdn, fneg=0):  # derH is a list of layers or sub-layers, each = [mtuple,dtuple, mval,dval, mrdn,drdn]

    DerH, Valt, Rdnt = T; derH, valt, rdnt = t
    for i in 0,1:
        Valt[i] += valt[i]
        Rdnt[i] += rdnt[i] + base_rdn
    DerH[:] = [
        # sum der layers, dertuple is mtuple | dtuple, fneg*i: for dtuple only:
        [ sum_dertuple(Mtuple, mtuple, fneg=0), sum_dertuple(Dtuple, dtuple, fneg=fneg) ]
        for [Mtuple, Dtuple], [mtuple, dtuple]
        in zip_longest(DerH, derH, fillvalue=[[0,0,0,0,0,0],[0,0,0,0,0,0]])  # mtuple,dtuple
    ]

def comp_derH(_derH, derH, rn=1, fagg=0):  # derH is a list of der layers or sub-layers, each = ptuple_tv

    Ht = []
    for derH in [_derH, derH]:  # init H is dertuplet, local convert to dertv_ (permanent conversion in sum2PP):
        Ht += [derH.H] if isinstance(derH.H[0],CderH) else [CderH(H=derH.H, valt=copy(derH.valt), rdnt=copy(derH.rdnt), dect=copy(derH.dect), depth=0)]
    derLay = []; Vt,Rt,Dt = [0,0],[0,0],[0,0]

    for _lay, lay in zip(Ht):
        mtuple,dtuple, Mtuple,Dtuple = comp_dtuple(_lay.H[1], lay.H[1], rn=rn, fagg=1)
        valt = [sum(mtuple),sum(abs(d) for d in dtuple)]
        rdnt = [valt[1] > valt[0], valt[1] <= valt[0]]
        if fagg:
            dect = [0,0]
            for fd, (ptuple,Ptuple) in enumerate(zip((mtuple,dtuple),(Mtuple,Dtuple))):
                for (par, max, ave) in zip(ptuple, Ptuple, aves):  # different ave for comp_dtuple
                    if fd: dect[1] += abs(par)/ abs(max) if max else 1
                    else:  dect[0] += (par+ave)/ (max+ave) if max else 1
            dect[0] = dect[0]/6; dect[1] = dect[1]/6  # ave of 6 params

        Vt = np.add(Vt,valt); Rt = np.add(Rt,rdnt)
        if fagg: Dt = np.divide(np.add(Dt,dect),2)
        derLay += [CderH(H=[mtuple,dtuple], valt=valt,rdnt=rdnt,dect=dect, depth=0)]  # dertvs

    return derLay, Vt,Rt,Dt  # to sum in each G Et

def comp_dtuple(_dT, dT, aves, rn):  # compare dtuples, include comp_ext and comp_ptuple?

    # same as numerical comp in comp_
    mtuple, dtuple, max_tuple = [],[],[]
    for _par, par, ave in zip(_dT, dT, aves):
        par *= rn
        match = min(abs(_par), abs(par))
        if (_par<0) != (par<0): match = -match
        mtuple += match - ave
        dtuple += [_par - par]
        max_tuple += max(abs(_par),abs(par))  # to compute dec: relative match if comp d

    tuplet = [mtuple,dtuple]
    valt = [sum(mtuple), sum(abs(d) for d in dtuple)]
    rdnt = [valt.d > valt.m, valt.d < valt.m]

    return tuplet, max_tuple, valt, rdnt


def form_PP_t(root, P_, irdn):  # form PPs of derP.valt[fd] + connected Ps val

    PP_t = [[],[]]
    for fd in 0,1:
        P_Ps = defaultdict(list)  # key: P, val: _Ps
        Link_ = []
        for P in P_:  # not PP.link_: P uplinks are unique, only G links overlap
            for derP in unpack_last_link_(P.link_):
                P_Ps[P] += [derP._P]; Link_ += [derP]  # not needed for PPs?
        inP_ = []  # clustered Ps and their val,rdn s for all Ps
        for P in root.P_:
            if P in inP_: continue  # already packed in some PP
            cP_ = [P]  # clustered Ps and their val,rdn s
            perimeter = deque(P_Ps[P])  # recycle with breadth-first search, up and down:
            while perimeter:
                _P = perimeter.popleft()
                if _P in cP_: continue
                cP_ += [_P]
                perimeter += P_Ps[_P]  # append linked __Ps to extended perimeter of P
            PP = sum2PP(root, cP_, Link_, irdn, fd)
            PP_t[fd] += [PP]  # no if Val > PP_aves[fd] * Rdn:
            inP_ += cP_  # update clustered Ps

    for PP in PP_t[1]:  # eval der+ / PPd only, after form_PP_t -> P.root
        if PP.Vt[1] * len(PP.link_) > PP_aves[1] * PP.Rt[1]:
            # node-mediated correlation clustering:
            der_recursion(root, PP, fd=1)
        if root.fback_:
            feedback(root)  # after der+ in all nodes, no single node feedback

    root.node_ = PP_t  # nested in der+, add_alt_PPs_?


def sum2PP(root, P_, derP_, irdn, fd):  # sum links in Ps and Ps in PP

    PP = Cgraph(fd=fd, root=root, P_=P_, rng=root.rng+1)  # initial PP.box = (inf,inf,-inf,-inf)
    # += uplinks:
    S,A = 0, [0,0]
    for derP in derP_:
        if derP.P not in P_ or derP._P not in P_: continue
        derP.P.derH.rdn[fd] += irdn; derP.P.derH += derP.derH
        derP._P.derH.rdn[fd]+= irdn; derP._P.derH -= derP.derH  # reverse d signs downlink
        PP.link_ += [derP]; derP.roott[fd] = PP
        PP.Vt = np.add(PP.Vt,derP.vt)
        PP.Rt = np.add(np.add(PP.Rt,derP.rt), [irdn,irdn])
        derP.A = np.add(A,derP.A); S += derP.S
    PP.ext = [len(P_), S, A]  # all from links
    depth = root.derH.depth or fd  # =1 at 1st der+
    PP.derH.depth = depth
    # += Ps:
    celly_,cellx_ = [],[]
    for P in P_:
        P.derH.depth = depth  # or copy from links
        PP.ptuple += P.ptuple
        PP.derH += P.derH
        for y,x in P.cells:
            PP.box = PP.box.accumulate(y,x); celly_+=[y]; cellx_+=[x]
    # pixmap:
    y0,x0,yn,xn = PP.box
    PP.mask__ = np.zeros((yn-y0, xn-x0), bool)
    celly_ = np.array(celly_); cellx_ = np.array(cellx_)
    PP.mask__[(celly_-y0, cellx_-x0)] = True

    return PP
'''
        if T.typ == 'derH':
            if t.H:
                if T.H:
                    H, Valt, Rdnt, Dect, Extt, Depth = T.H, T.valt, T.rdnt, T.dect, T.ext, t.depth
                    h, valt, rdnt, dect, extt, depth = t.H, t.valt, t.rdnt, t.dect, t.ext, t.depth
                    Valt[:] = np.add(Valt,valt)
                    Rdnt[:] = np.add( np.add(Rdnt,rdnt), [T.irdn,t.irdn])
                    Rdnt[0] += Valt[1] > Valt[0]
                    Rdnt[1] += Valt[0] > Valt[1]
                    if T.fagg:
                        Dect[:] = np.divide( np.add(Dect,dect), 2)
                    fC=0
                    if isinstance(H[0], z):
                        fC=1
                        if isinstance(h[0], list):  # convert dertv to derH:
                            h = [CderH(H=h, valt=copy(t.valt), rdnt=copy(t.rdnt), dect=copy(t.dect), ext=copy(t.ext), depth=0)]
                    elif isinstance(h[0], z):
                        fC=1; H = [CderH(H=H, valt=copy(T.valt), rdnt=copy(T.rdnt), dect=copy(T.dect), ext=copy(T.ext), depth=0)]

                    if fC:  # both derH_:
                        add_(H, h)
                        # H[:(len(h))] = [DerH + derH for DerH, derH in zip_longest(H,h)]  # if different length or always same? (could be different length)
                    else:  # both dertuplets:
                        H[:] = [list(np.add(Dertuple,dertuple)) for Dertuple, dertuple in zip(H,h)]  # mtuple,dtuple
                else:
                    T.H[:] = deepcopy(t.H)
'''