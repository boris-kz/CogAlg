import sys
import numpy as np
from itertools import zip_longest
from copy import deepcopy, copy
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
import math as math
from comp_slice import *

'''
Blob edges may be represented by higher-composition PPPs, etc., if top param-layer match,
in combination with spliced lower-composition PPs, etc, if only lower param-layers match.
This may form closed edge patterns around flat blobs, which defines stable objects.   
'''

class CderPP(ClusterStructure):  # tuple of derivatives in PP uplink_ or downlink_, PP can also be PPP, etc.

    player = lambda: [None], [None]  # lists of ptuples in current derivation layer per fork
    mval = float  # summed player vals, both are signed, PP sign by fPds[-1]
    dval = float
    # 5 below are not needed?
    box = list  # 2nd level, stay in PP?
    rdn = int  # mrdn, + uprdn if branch overlap?
    _PP = object  # prior comparand  or always in PP_?
    PP = object  # next comparand
    root = lambda: None  # PP, not segment in sub_recursion
    # not needed?
    uplink_layers = lambda: [[], []]  # init a layer of dderPs and a layer of match_dderPs
    downlink_layers = lambda: [[], []]
    # from comp_dx
    fdx = NoneType

class CPPP(CPP, CderPP):
    players = list  # max n ptuples in layer = n ptuples in all lower layers: 1, 1, 2, 4, 8...
    fPds = list  # prior fork sequence, map to mplayer or dplayer in lower (not parallel) player, none for players[0]
    rng = lambda: 1  # rng starts with 1
    rdn = int  # for PP evaluation, recursion count + Rdn / nderPs
    Rdn = int  # for accumulation only
    nP = int  # len 2D derP__ in levels[0][fPd]?  ly = len(derP__), also x, y?
    uplink_layers = lambda: [[], []]  # likely not needed
    downlink_layers = lambda: [[], []]
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_
    fdiv = NoneType
    box = list  # x0, xn, y0, yn
    mask__ = bool
    PP_ = list  # (input,derPP,fin)s, common root of layers and levels
    cPP_ = list  # co-refs in other PPPs
    rlayers = list  # | mlayers
    dlayers = list  # | alayers
    levels = lambda: [[]]  # agg_PPs ) agg_PPPs ) agg_PPPPs..
    root = lambda: None  # higher-order segP or PPP


def agg_recursion(dir_blob, PP_, fPd, fseg=0):  # compositional recursion per blob.Plevel; P, PP, PPP are relative to each other

    comb_levels = []
    ave_PP = ave_dPP if fPd else ave_mPP
    V = sum([PP.dval for PP in PP_]) if fPd else sum([PP.mval for PP in PP_])
    if V > ave_PP:
        rng = V/ave_PP  # cross-comp, bilateral match assign list per PP, re-clustering by match to PPP centroids

        PPP_ = comp_PP_(PP_, rng)  # cross-comp all PPs within rng, same PPP_ for both forks
        PPPm_, PPPd_ = comp_centroid(PPP_)  # if top level miss, lower levels match: splice PPs vs form PPPs
        # add separate centroid clustering forks
        sub_recursion_agg(PPPm_, fPd=0)  # reform PP_ per PPP for rng+
        sub_recursion_agg(PPPd_, fPd=1)  # reform PP_ per PPP for der+
        agg_recursion_eval(PPPm_, dir_blob, fPd=0)
        agg_recursion_eval(PPPd_, dir_blob, fPd=1)

        for PPP_ in PPPm_, PPPd_:
            for PPP in PPPm_:
                for i, (comb_level, level) in enumerate(zip_longest(comb_levels, PPP.agg_levels, fillvalue=[])):
                    if level:
                        if i > len(comb_levels) - 1:
                            comb_levels += [[level]]  # add new level
                        else:
                            comb_levels[i] += [level]  # append existing layer
            sub_comb_levels = [PPP_] + comb_levels
        comb_levels += [sub_comb_levels]  # not sure

    return comb_levels

# draft
def sub_recursion_agg(PPP_, fPd):  # rng+: extend PP_ per PPP, der+: replace PP with derPP in PPt

    comb_layers = []
    for PPP in PPP_:
        val = PPP.dval if fPd else PPP.mval
        ave = ave_dPP if fPd else ave_mPP

        if val > ave and len(PPP.PP_) > ave_nsub:
            PP_ = [PPt[0] for PPt in PPP.PP_]

            sub_PPP_ = comp_PP_(PP_, rng=val/ave)  # cross-comp all PPs within rng
            comp_centroid(sub_PPP_)  # may splice PPs instead of forming PPPs

            for i, (comb_layer, PPP_layer) in enumerate(zip_longest(comb_layers, PPP.dlayers if fPd else PPP.rlayers, fillvalue=[])):
                if PPP_layer:
                    if i > len(comb_layers) - 1:
                        comb_layers += [PPP_layer]  # add new r|d layer
                    else:
                        comb_layers[i] += PPP_layer  # splice r|d PP layer into existing layer

    return comb_layers

def agg_recursion_eval(PP_, root, fPd, fseg=0):  # from agg_recursion per fork, adds agg_level to agg_PP or dir_blob

    if isinstance(root, CPP): dval = root.dval; mval = root.mval
    else:                     dval = root.G; mval = root.M
    if fPd: ave_PP = ave_dPP; val = dval; alt_val = mval
    else:   ave_PP = ave_mPP; val = mval; alt_val = dval
    ave = ave_PP * (3 + root.rdn + 1 + (alt_val > val))  # fork rdn per PP, 3: agg_coef

    if val > ave and len(PP_) > ave_nsub:
        root.levels += [agg_recursion(root, root.levels[-1], fPd, fseg)]


def comp_PP_(PP_, rng):  # rng cross-comp, draft

    PPP_ = []
    iPPP_ = [CPPP(PP=PP, players=deepcopy(PP.players), fPds=deepcopy(PP.fPds), x0=PP.x0, xn=PP.xn, y0=PP.y0, yn=PP.yn) for PP in PP_]

    while PP_:  # compare _PP to all other PPs within rng
        _PP, _PPP = PP_.pop(), iPPP_.pop()
        _PP.root = _PPP
        for PPP, PP in zip(iPPP_[:rng], PP_[:rng]):  # need to revise for partial rng

            # all possible comparands in dy<rng, with incremental y, accum derPPs in PPPs
            area = PP.players[0][0].L; _area = _PP.players[0][0].L  # not sure
            dx = ((_PP.xn - _PP.x0) / 2) / _area - ((PP.xn - PP.x0) / 2) / area
            dy = _PP.y / _area - PP.y / area
            distance = np.hypot(dy, dx)  # Euclidean distance between PP centroids

            if distance * ((_PP.mval + PP.mval) / 2 / ave_mPP) <= 3:  # ave_rng
                # comp PPs:
                mplayer, dplayer = comp_players(_PP.players, PP.players, _PP.fPds, PP.fPds)
                mval = sum([mtuple.val for mtuple in mplayer])
                dval = sum([dtuple.val for dtuple in dplayer])
                derPP = CderPP(player=[mplayer, dplayer], mval=mval, dval=dval)  # derPP is single-layer, PP stays as is
                if mval > ave_mPP:
                    fin = 1  # PPs match, sum derPP in both PPP and _PPP, m fork:
                    sum_players(_PPP.players, PP.players)
                    sum_players(PPP.players, PP.players)  # same fin for both in comp_PP_
                else:
                    fin = 0
                _PPP.PP_ += [[PP, derPP, fin]]
                _PP.cPP_ += [[PP, derPP, 1]]  # rdn refs, initial fin=1, derPP is reversed
                PPP.PP_ += [[_PP, derPP, fin]]
                PP.cPP_ += [[_PP, derPP, 1]]  # bilateral assign to eval in centroid clustering, derPP is reversed
                '''
                if derPP.match params[-1]: form PPP
                elif derPP.match params[:-1]: splice PPs and their segs? 
                '''
        PPP_.append(_PPP)
    return PPP_


def comp_centroid(PPP_):  # comp PP to average PP in PPP, sum >ave PPs into new centroid, recursion while update>ave

    update_val = 0  # update val, terminate recursion if low

    for PPP in PPP_:
        PPP_val = 0  # new total, may delete PPP
        PPP_rdn = 0  # rdn of PPs to cPPs in other PPPs
        PPP_players = []
        for i, (PP, _, fin) in enumerate(PPP.PP_):  # comp PP to PPP centroid, derPP is replaced, use comp_plevels?

            mplayer, dplayer = comp_players(PPP.players, PP.players, PPP.fPds, PP.fPds)  # norm params in comp_ptuple
            mval = sum([mtuple.val for mtuple in mplayer])
            dval = sum([dtuple.val for dtuple in dplayer])
            derPP = CderPP(player=[mplayer, dplayer], mval=mval, dval=dval)  # derPP is recomputed at each call
            # compute rdn:
            cPP_ = PP.cPP_
            cPP_ = sorted(cPP_, key=lambda cPP: cPP[1].mval, reverse=True)  # sort by derPP.val per recursion call
            rdn = 1
            for (cPP, cderPP, cfin) in cPP_:
                if cderPP.mval > derPP.mval:  # cPP is instance of PP, eval derPP.mval only
                    if cfin: PPP_rdn += 1  # n of cPPs redundant to PP, if included and >val
                else:
                    break
            fneg = mval < ave_mPP * rdn  # rdn per PP

            if (fneg and fin) or (not fneg and not fin):  # re-clustering: exclude included or include excluded PP
                PPP.PP_[i][2] = not fin
                update_val += abs(mval)  # or sum abs mparams?
            if not fneg:  # include PP in PPP:
                PPP_val += mval
                PPP_rdn += rdn
                if PPP_players:
                    sum_players(PPP_players, PP.players, fneg)  # no fneg now?
                else:
                    PPP_players = PP.players  # initialization is simpler
                PPP.PP_[i][1] = derPP  # replace not sum

                for i, cPPt in enumerate(PP.cPP_):
                    if isinstance(cPPt[0], CPP):  # why is this needed?
                        cPPP = cPPt[0].root
                        for j, PPt in enumerate(cPPP.PP_):  # get PPP and replace their cPP with derPP
                            if PPt[0] is PP:
                                cPPP.PP_[j][0] = derPP
                        PPP.cPPt_[i][0] = derPP

        if PPP_players: PPP.players = PPP_players
        if PPP_val < ave_mPP * PPP_rdn:  # ave rdn-adjusted value per cost of PPP

            update_val += abs(PPP_val)  # or sum abs mparams?
            PPP_.remove(PPP)  # PPPs are hugely redundant, need to be pruned

            for (PP, derPP, fin) in PPP.PP_:  # remove refs to local copy of PP in other PPPs
                for (cPP, _, _) in PP.cPP_:
                    for i, (ccPP, _, _) in enumerate(cPP.cPP_):  # ref of ref
                        if ccPP is PP:
                            cPP.cPP_.pop(i)  # remove ccPP tuple
                            break

    if update_val > ave_mPP:
        comp_centroid(PPP_)  # recursion while min update value


# for deeper agg_recursion:
def comp_levels(_levels, levels, der_levels, fsubder=0):  # only for agg_recursion, each param layer may consist of sub_layers

    # recursive unpack of nested param layers, each layer is ptuple pair_layers if from der+
    der_levels += [comp_players(_levels[0], levels[0])]

    # recursive unpack of deeper layers, nested in 3rd and higher layers, if any from agg+, down to nested tuple pairs
    for _level, level in zip(_levels[1:], levels[1:]):  # level = deeper sub_levels, stop if none
        der_levels += [comp_levels(_level, level, der_levels=[], fsubder=fsubder)]

    return der_levels  # possibly nested param layers


'''
    1st and 2nd layers are single sublayers, the 2nd adds tuple pair nesting. Both are unpacked by func_pairs, not func_layers.  
    Multiple sublayers start on the 3rd layer, because it's derived from comparison between two (not one) lower layers. 
    4th layer is derived from comparison between 3 lower layers, where the 3rd layer is already nested, etc:
    initial 3-layer nesting: https://github.com/assets/52521979/ea6d436a-6c5e-429f-a152-ec89e715ebd6
    '''


# the below is out of date:

def sum_levels(Params, params):  # Capitalized names for sums, as comp_levels but no separate der_layers to return

    if Params:
        sum_players(Params[0], params[0])  # recursive unpack of nested ptuple layers, if any from der+
    else:
        Params.append(deepcopy(params[0]))  # no need to sum

    for Level, level in zip_longest(Params[1:], params[1:], fillvalue=[]):
        if Level and level:
            sum_levels(Level, level)  # recursive unpack of higher levels, if any from agg+ and nested with sub_levels
        elif level:
            Params.append(deepcopy(level))  # no need to sum


def form_PPP_t(iPPP_t):  # form PPs from match-connected segs
    PPP_t = []

    for fPd, iPPP_ in enumerate(iPPP_t):
        # sort by value of last layer: derivatives of all lower layers:
        if fPd:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.dval, reverse=True)  # descending order
        else:
            iPPP_ = sorted(iPPP_, key=lambda iPPP: iPPP.mval, reverse=True)  # descending order

        PPP_ = []
        for i, iPPP in enumerate(iPPP_):
            if fPd:
                iPPP_val = iPPP.dval
            else:
                iPPP_val = iPPP.mval

            for mptuple, dptuple in zip(iPPP.mplayer, iPPP.dplayer):
                if mptuple and dptuple:  # could be None
                    if fPd:
                        iPPP.rdn += dptuple.val > mptuple.val
                    else:
                        iPPP.rdn += mptuple.val > dptuple.val
            '''
            for param_layer in iPPP.params:  # may need recursive unpack here
                iPPP.rdn += sum_named_param(param_layer, 'val', fPd=fPd)> sum_named_param(param_layer, 'val', fPd=1-fPd)
            '''

            ave = vaves[fPd] * iPPP.rdn * (i + 1)  # derPP is redundant to higher-value previous derPPs in derPP_
            if iPPP_val > ave:
                PPP_ += [iPPP]  # base derPP and PPP is CPP
                if iPPP_val > ave * 10:
                    comp_PP_(iPPP, fPd)  # replace with reclustering?
            else:
                break  # ignore below-ave PPs
        PPP_t.append(PPP_)
    return PPP_t


def form_segPPP_root(PP_, root_rdn, fPd):  # not sure about form_seg_root

    for PP in PP_:
        link_eval(PP.uplink_layers, fPd)
        link_eval(PP.downlink_layers, fPd)

    for PP in PP_:
        form_segPPP_(PP)


def form_segPPP_(PP):
    pass


# pending update
def splice_segs(seg_):  # in 1st run of agg_recursion
    pass


# draft, splice 2 PPs for now
def splice_PPs(PP_, frng):  # splice select PP pairs if der+ or triplets if rng+

    spliced_PP_ = []
    while PP_:
        _PP = PP_.pop(0)  # pop PP, so that we can differentiate between tested and untested PPs
        tested_segs = []  # new segs may be added during splicing, their links also need to be checked for splicing
        _segs = _PP.seg_levels[0]

        while _segs:
            _seg = _segs.pop(0)
            _avg_y = sum([P.y for P in _seg.P__]) / len(_seg.P__)  # y centroid for _seg

            for link in _seg.uplink_layers[1] + _seg.downlink_layers[1]:
                seg = link.P.root  # missing link of current seg

                if seg.root is not _PP:  # after merging multiple links may have the same PP
                    avg_y = sum([P.y for P in seg.P__]) / len(seg.P__)  # y centroid for seg

                    # test for y distance (temporary)
                    if (_avg_y - avg_y) < ave_splice:
                        if seg.root in PP_:
                            PP_.remove(seg.root)  # remove merged PP
                        elif seg.root in spliced_PP_:
                            spliced_PP_.remove(seg.root)
                        # splice _seg's PP with seg's PP
                        merge_PP(_PP, seg.root)

            tested_segs += [_seg]  # pack tested _seg
        _PP.seg_levels[0] = tested_segs
        spliced_PP_ += [_PP]

    return spliced_PP_


def merge_PP(_PP, PP, fPd):  # only for PP splicing

    for seg in PP.seg_levels[fPd][-1]:  # merge PP_segs into _PP:
        accum_PP(_PP, seg, fPd)
        _PP.seg_levels[fPd][-1] += [seg]

    # merge uplinks and downlinks
    for uplink in PP.uplink_layers:
        if uplink not in _PP.uplink_layers:
            _PP.uplink_layers += [uplink]
    for downlink in PP.downlink_layers:
        if downlink not in _PP.downlink_layers:
            _PP.downlink_layers += [downlink]