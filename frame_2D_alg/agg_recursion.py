from comp_slice import *

class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    # draft
    params = list  # PP derivation layer, flat, decoded by mapping each m,d to lower-layer param
    x0 = int  # redundant to params:
    x = float  # median x
    L = int  # pack in params?
    sign = NoneType  # g-ave + ave-ga sign
    y = int  # for vertical gaps in PP.P__, replace with derP.P.y?
    PP = object  # lower comparand
    _PP = object  # higher comparand
    root = lambda:None  # segment in sub_recursion
    # higher derivatives
    rdn = int  # mrdn, + uprdn if branch overlap?
    uplink_layers = lambda: [[],[]]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[],[]]
   # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):

    # draft
    params = list  # derivation layers += derP params per der+, param L is actually Area
    sign = bool
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[],[]]
    downlink_layers = lambda: [[],[]]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # for visualization only, original box before flipping
    mask__ = bool
    P__ = list  # input  # derP__ = list  # redundant to P__
    seg_levels = lambda: [[[]],[[]]]  # from 1st agg_recursion, seg_levels[0] is seg_t, higher seg_levels are segP_t s
    PPP_levels = list  # from 2nd agg_recursion, PP_t = levels[0], from form_PP, before recursion
    layers = list  # from sub_recursion, each is derP_t
    root = lambda:None  # higher-order segP or PPP


def agg_recursion(blob, fseg):  # compositional recursion per blob.Plevel. P, PP, PPP are relative terms, each may be of any composition order

    if fseg: PP_t = [blob.seg_levels[0][-1], blob.seg_levels[1][-1]]   # blob is actually PP, recursion forms segP_t, seg_PP_t, etc.
    else:    PP_t = [blob.levels[0][-1], blob.levels[1][-1]]  # input-level composition Ps, initially PPs

    PPP_t = []  # next-level composition Ps, initially PPPs  # for fiPd, PP_ in enumerate(PP_t): fiPd = fiPd % 2  # dir_blob.M += PP.M += derP.m
    n_extended = 0

    for i, PP_ in enumerate(PP_t):   # fiPd = fiPd % 2
        fiPd = i % 2
        if fiPd: ave_PP = ave_dPP
        else:    ave_PP = ave_mPP

        if fseg: M = ave- np.hypot(blob.params[0][5], blob.params[0][6])  # hypot(dy, dx)
        else: M = ave-abs(blob.G)
        if M > ave_PP * blob.rdn and len(PP_)>1:  # >=2 comparands
            n_extended += 1

            PPm__ = [comp_PP_root(deepcopy(PP_))]  # PP is generic for lower-level composition
            PPd__ = [comp_PP_root(deepcopy(PP_))]
            # to be updated:
            segm_ = form_seg_root(PPm__, root_rdn=2, fPd=0)  # forms segments: parameterized stacks of (P,derP)s
            segd_ = form_seg_root(PPd__, root_rdn=2, fPd=1)  # seg is a stack of (P,derP)s

            PPPm_, PPPd_ = form_PPP_root([segm_, segd_], root_rdn=2)  # PPP is generic next-level composition
            splice_PPs(PPPm_, frng=1)
            splice_PPs(PPPd_, frng=0)
            PPP_t += [PPPm_, PPPd_]  # flat version

            if PPPm_: sub_recursion([], PPPm_, frng=1)  # rng+
            if PPPd_: sub_recursion([], PPPd_, frng=0)  # der+
        else:
            PPP_t += [[], []]  # replace with neg PPPs?

    if fseg:
        blob.seg_levels += [PPP_t]  # new level of segPs
    else:
        blob.levels.append(PPP_t)  # levels of dir_blob are Plevels

    if n_extended/len(PP_t) > 0.5:  # mean ratio of extended PPs
        agg_recursion(blob, fseg)

# draft:
def comp_PP_root(PP__):  # PP__ is 2D, PP can also be PPP, etc.

    for PP_ in PP__:
        for PP in PP_:
            for _PP in PP.uplink_layers[-1]:
                comp_PP(_PP, PP)  # variable rng, comp cross-sign if PPd?
    '''
    # not sure these are needed:
    uplink_layers = [[] for PP_ in PP__]
    downlink_layers = deepcopy(uplink_layers)
    
        uplink_layers[i] += [derPP]  # add derPP
        if _PP in PP_: downlink_layers[PP_.index(_PP)] += [derPP]

    for PP, uplink_layer, downlink_layer in zip_longest(PP_, uplink_layers, downlink_layers, fillvalue=[]):
        PP.uplink_layers += [uplink_layer]
        PP.downlink_layers += [downlink_layer]
    '''

def comp_PP(_PP, PP):  # draft

    derPP = CderPP

    for _param_layer, param_layer in zip(_PP.params, PP.params):
        derPP += [comp_derP(_param_layer, param_layer)]

    PP.uplink_layers[-1] += [derPP]  # each layer has Mlayer, Dlayer
    _PP.downlink_layers[-1] += [derPP]

def form_segPPP_root():  # not sure about form_seg_root
    pass

def form_PPP_root(seg_t, root_rdn=2):
    '''
    if match params[-1]: form PPP
    elif match params[:-1]: splice PPs and their segs?
    '''
    pass

# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass

# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # we need this because we may add new seg during splicing process, and those new seg need to check their link for splicing too
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__])/len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # this may occur after the merging where multiple links are having same PP
                    avg_y = sum([P.y for P in seg.P__])/len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_: PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_: spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_

# to be updated
def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_CPP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]


def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Ddx = Ddx
    P.Mdx = Mdx

