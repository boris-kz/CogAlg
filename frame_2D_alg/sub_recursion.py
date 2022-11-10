'''
comp_slice_ sub_recursion + utilities
'''
import cv2
from itertools import zip_longest
from copy import copy, deepcopy
import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice import CPP,CP, CderP, form_seg_root,form_PP_root, comp_P, sum_players, slice_blob, ave_nsub, ave_splice, PP_aves
from agg_recursion import agg_recursion_eval

flip_ave = 10
ave_dir_val = 50
ave_M = -500  # high negative ave M for high G blobs


def sub_recursion_eval(root):  # for PP or dir_blob

    if isinstance(root, CPP): root_PPm_, root_PPd_ = root.rlayers[0], root.dlayers[0]
    else:                     root_PPm_, root_PPd_ = root.PPm_, root.PPd_

    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:
            if fd:  # add root to derP for der+:
                for P_ in PP.P__[1:-1]:  # skip 1st and last row
                    for P in P_:
                        for derP in P.uplink_layers[-1][fd]:
                            derP.roott[fd] = PP
                comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:
                comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]

            val = PP.valt[fd]; alt_val = PP.valt[1-fd]  # for fork rdn:
            ave = PP_aves[fd] * (PP.rdn + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # 1+PP.rdn incr
                # splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer

            # segs agg_recursion:
            agg_recursion_eval(PP, [copy(PP.mseg_levels[-1]), copy(PP.dseg_levels[-1])])
            # include empty comb_layers:
            if fd: root.dlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]
            else:  root.rlayers = [[[PPm_] + mcomb_layers], [[PPd_] + dcomb_layers]]

            # or higher der val?
            if isinstance(root, CPP):  # root is CPP
                root.valt[fd] += PP.valt[fd]
            else:  # root is CBlob
                if fd: root.G += PP.valt[1]
                else:  root.M += PP.valt[0]

def sub_recursion(PP):  # evaluate each PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert to top down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdn += 2  # two-fork rdn, priority is not known?

    sub_segm_ = form_seg_root([copy(P_) for P_ in P__], fd=0, fds=PP.fds)
    sub_segd_ = form_seg_root([copy(P_) for P_ in P__], fd=1, fds=PP.fds)  # returns bottom-up
    # sub_PPm_, sub_PPd_:
    PP.rlayers[0], PP.dlayers[0] = form_PP_root((sub_segm_, sub_segd_), PP.rdn + 1)
    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs


def comp_P_rng(P__, rng):  # rng+ sub_recursion in PP.P__, switch to rng+n to skip clustering?

    for P_ in P__:
        for P in P_:  # add 2 link layers: rng_derP_ and match_rng_derP_:
            P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]

    for y, _P_ in enumerate(P__[:-rng]):  # higher compared row, skip last rng: no lower comparand rows
        for x, _P in enumerate(_P_):
            # get linked Ps at dy = rng-1:
            for pri_derP in _P.downlink_layers[-3][0]:  # fd always = 0 here
                pri_P = pri_derP.P
                # compare linked Ps at dy = rng:
                for ini_derP in pri_P.downlink_layers[0]:
                    P = ini_derP.P
                    # add new Ps, their link layers and reset their roots:
                    if P not in [P for P_ in P__ for P in P_]:
                        append_P(P__, P)
                        P.uplink_layers += [[],[[],[]]]; P.downlink_layers += [[],[[],[]]]
                    derP = comp_P(_P, P)
                    P.uplink_layers[-2] += [derP]
                    _P.downlink_layers[-2] += [derP]

    return P__

def comp_P_der(P__):  # der+ sub_recursion in PP.P__, compare P.uplinks to P.downlinks

    dderPs__ = []  # derP__ = [[] for P_ in P__[:-1]]  # init derP rows, exclude bottom P row

    for P_ in P__[1:-1]:  # higher compared row, exclude 1st: no +ve uplinks, and last: no +ve downlinks
        # not revised:
        dderPs_ = []  # row of dderPs
        for P in P_:
            dderPs = []  # dderP for each _derP, derP pair in P links
            for _derP in P.uplink_layers[-1][1]:  # fd=1
                for derP in P.downlink_layers[-1][1]:
                    # there maybe no x overlap between recomputed Ls of _derP and derP, compare anyway,
                    # mderP * (ave_olp_L / olp_L)? or olp(_derP._P.L, derP.P.L)?
                    # gap: neg_olp, ave = olp-neg_olp?
                    dderP = comp_P(_derP, derP)  # form higher vertical derivatives of derP.players,
                    # or comp derP.players[1] only: it's already diffs of all lower players?
                    derP.uplink_layers[0] += [dderP]  # pre-init layer per derP
                    _derP.downlink_layers[0] += [dderP]
                    dderPs_ += [dderP]  # actually it could be dderPs_ ++ [derPP]
                # compute x overlap between dderP'__P and P, in form_seg_ or comp_layer?
            dderPs_ += dderPs  # row of dderPs
        dderPs__ += [dderPs_]

    return dderPs__

# not revised, should be rotate():

def rotate_dert_(idert_, mask_):

    # method from here:
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/rotate.htm

    idert_ = list(idert_)  # idert_ is zipped, need to convert it
    nparams = len(idert_[0])
    ysize = 1
    xsize = len(idert_)

    dert_ = [[] for _ in range(nparams)]
    for dert in list(idert_):
        for i in range(nparams):
            dert_[i].append(dert[i])

    ave_ga = sum(dert_[2]) / xsize  # average of ga in rad
    ave_x = int(xsize/2)  # local ave_x to dert_ only

    x_ = []; new_x_ = []
    y_ = []; new_y_ = []
    # rotate by ga
    for x in range(xsize):
        x_ += [x]; y_ += [0]
        new_x_ += [int(np.round((np.cos(ave_ga) * (x-ave_x)) + ave_x))]
        new_y_ += [int(np.round((np.sin(ave_ga) * (x-ave_x)) + 0))]

    # remove negative coordinates by scaling it to positive
    min_x = min(new_x_)
    min_y = min(new_y_)
    new_x_ = [ new_x - min_x for new_x in new_x_]
    new_y_ = [ new_y - min_y for new_y in new_y_]

    # initialize rotate dert and mask
    new_xsize = max(new_x_) + 1
    new_ysize = max(new_y_) + 1
    rotated_dert_ = [np.zeros((new_ysize, new_xsize), dtype="uint8") for _ in range(nparams)]
    rotated_mask = np.full((new_ysize, new_xsize), fill_value=True, dtype="bool")

    # fill value
    for x,y,new_x, new_y in zip(x_,y_,new_x_, new_y_):
        for i in range(nparams):
            rotated_dert_[i][new_y, new_x] = dert_[i][x]
            rotated_mask[new_y, new_x] = mask_[x]

    return rotated_dert_, rotated_mask


def rotate_dert2(dert, angle):

    # using opencv method
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    ysize, xsize = dert.shape[:2]
    center = (xsize/2, ysize/ 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # get new xsize and new ysize
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_xsize = int(ysize * abs_sin + xsize * abs_cos)
    new_ysize = int(ysize * abs_cos + xsize * abs_sin)

    rotation_matrix[0, 2] += new_xsize / 2 - center[0]
    rotation_matrix[1, 2] += new_ysize / 2 - center[1]

    # rotate image with the new xysize and translated rotation matrix
    rotated_dert = cv2.warpAffine(dert, rotation_matrix, (new_xsize, new_ysize))

    return rotated_dert.astype("uint8")

'''
P.axis: 0|1|2|3, which maps to angle. P x0,xn,y0,yn, they can be computed from x,y + (axis_angle * len_dert_/2)
P.x = (P.x0+ len P.dert_/2) and med_dert is P.dert_[len P.dert_/2]  Same for P.y. 
New: dert_= [med_dert], dert_.appendleft(left_derts), dert_.append(right_derts).
New P.x = ave_x + (len(right_derts) - len(left_derts))/2 * axis_angle. Something like that, similar for P.y.
'''
def dir_eval(Dy, Dx):  # blob direction strength eval

    G = np.hypot(Dy,Dx)
    rD = Dy / Dx if Dx else 2 * Dy
    if abs(G * rD) < ave_dir_val:
        return True
    else: return False

def merge_blobs(blob, adj_blob, strong_adj_blobs):  # merge blob and adj_blob by summing their params and combining dert__ and mask__

    # accumulate blob Dert
    blob.accum_from(blob)

    _y0, _yn, _x0, _xn = blob.box
    y0, yn, x0, xn = adj_blob.box
    cy0 = min(_y0, y0); cyn = max(_yn, yn); cx0 = min(_x0, x0); cxn = max(_xn, xn)

    if (y0<=_y0) and (yn>=_yn) and (x0<=_x0) and (xn>=_xn): # blob is inside adj blob
        # y0, yn, x0, xn for blob within adj blob
        ay0 = (_y0 - y0); ayn = (_yn - y0); ax0 = (_x0 - x0); axn = (_xn - x0)
        extended_mask__ = adj_blob.mask__ # extended mask is adj blob's mask, AND extended mask with blob mask
        extended_mask__[ay0:ayn, ax0:axn] = np.logical_and(blob.mask__, extended_mask__[ay0:ayn, ax0:axn])
        extended_dert__ = adj_blob.dert__ # if blob is inside adj blob, blob derts should be already in adj blob derts

    elif (_y0<=y0) and (_yn>=yn) and (_x0<=x0) and (_xn>=xn): # adj blob is inside blob
        # y0, yn, x0, xn for adj blob within blob
        by0  = (y0 - _y0); byn  = (yn - _y0); bx0  = (x0 - _x0); bxn  = (xn - _x0)
        extended_mask__ = blob.mask__ # extended mask is blob's mask, AND extended mask with adj blob mask
        extended_mask__[by0:byn, bx0:bxn] = np.logical_and(adj_blob.mask__, extended_mask__[by0:byn, bx0:bxn])
        extended_dert__ = blob.dert__ # if adj blob is inside blob, adj blob derts should be already in blob derts

    else:
        # y0, yn, x0, xn for combined blob and adj blob box
        cay0 = _y0-cy0; cayn = _yn-cy0; cax0 = _x0-cx0; caxn = _xn-cx0
        cby0 =  y0-cy0; cbyn =  yn-cy0; cbx0 = x0-cx0;  cbxn = xn-cx0
        # create extended mask from combined box
        extended_mask__ = np.ones((cyn-cy0,cxn-cx0)).astype('bool')
        extended_mask__[cay0:cayn, cax0:caxn] = np.logical_and(blob.mask__, extended_mask__[cay0:cayn, cax0:caxn])
        extended_mask__[cby0:cbyn, cbx0:cbxn] = np.logical_and(adj_blob.mask__, extended_mask__[cby0:cbyn, cbx0:cbxn])
        # create extended derts from combined box
        extended_dert__ = [np.zeros((cyn-cy0,cxn-cx0)) for _ in range(len(blob.dert__))]
        for i in range(len(blob.dert__)):
            extended_dert__[i][cay0:cayn, cax0:caxn] = blob.dert__[i]
            extended_dert__[i][cby0:cbyn, cbx0:cbxn] = adj_blob.dert__[i]

    # update dert, mask , box and sign
    blob.dert__ = extended_dert__
    blob.mask__ = extended_mask__
    blob.box = [cy0,cyn,cx0,cxn]
    blob.sign = abs(blob.Dy)>abs(blob.Dx)

    # add adj_blob's adj blobs to strong_adj_blobs to merge or add them as adj_blob later
    for adj_adj_blob,pose in zip(*adj_blob.adj_blobs):
        if adj_adj_blob not in blob.adj_blobs[0] and adj_adj_blob is not blob:
            strong_adj_blobs.append(adj_adj_blob)

    # update adj blob 'adj blobs' adj_blobs reference from pointing adj blob into the merged blob
    for i, adj_adj_blob1 in enumerate(adj_blob.adj_blobs[0]):            # loop adj blobs of adj blob
        for j, adj_adj_blob2 in enumerate(adj_adj_blob1.adj_blobs[0]):   # loop adj blobs from adj blobs of adj blob
            if adj_adj_blob2 is adj_blob and adj_adj_blob1 is not blob : # update reference to the merged blob
                adj_adj_blob1.adj_blobs[0][j] = blob

    return blob


def visualize_merging_process(iblob, dir_blob_, _dir_blob_, mask__, i):

    cv2.namedWindow('(1-3)Before merging, (4-6)After merging - weak blobs,strong blobs,strong+weak blobs', cv2.WINDOW_NORMAL)

    # masks - before merging
    img_mask_strong = np.ones_like(mask__).astype('bool')
    img_mask_weak = np.ones_like(mask__).astype('bool')
    # get mask of dir blobs
    for dir_blob in _dir_blob_:
        y0, yn, x0, xn = dir_blob.box

        rD = dir_blob.Dy / dir_blob.Dx if dir_blob.Dx else 2 * dir_blob.Dy
        # direction eval on the blob
        if abs(dir_blob.G * rD)  < ave_dir_val: # weak blob
            img_mask_weak[y0:yn, x0:xn] = np.logical_and(img_mask_weak[y0:yn, x0:xn], dir_blob.mask__)
        else:
            img_mask_strong[y0:yn, x0:xn] = np.logical_and(img_mask_strong[y0:yn, x0:xn], dir_blob.mask__)

    # masks - after merging
    img_mask_strong_merged = np.ones_like(mask__).astype('bool')
    img_mask_weak_merged = np.ones_like(mask__).astype('bool')
    # get mask of merged blobs
    for dir_blob in iblob.dir_blobs:
        y0, yn, x0, xn = dir_blob.box

        rD = dir_blob.Dy / dir_blob.Dx if dir_blob.Dx else 2 * dir_blob.Dy
        # direction eval on the blob
        if abs(dir_blob.G * rD)  < ave_dir_val: # weak blob
            img_mask_weak_merged[y0:yn, x0:xn] = np.logical_and(img_mask_weak_merged[y0:yn, x0:xn], dir_blob.mask__)
        else:  # strong blob
            img_mask_strong_merged[y0:yn, x0:xn] = np.logical_and(img_mask_strong_merged[y0:yn, x0:xn], dir_blob.mask__)

    # assign value to masks
    img_separator = np.ones((mask__.shape[0],2)) * 20         # separator
    # before merging
    img_weak = ((~img_mask_weak)*90).astype('uint8')                    # weak blobs before merging process
    img_strong = ((~img_mask_strong)*255).astype('uint8')               # strong blobs before merging process
    img_combined = img_weak + img_strong                                # merge weak and strong blobs
    # img_overlap = np.logical_and(~img_mask_weak, ~img_mask_strong)*255
    # after merging
    img_weak_merged = ((~img_mask_weak_merged)*90).astype('uint8')          # weak blobs after merging process
    img_strong_merged = ((~img_mask_strong_merged)*255).astype('uint8')     # strong blobs after merging process
    img_combined_merged = img_weak_merged + img_strong_merged               # merge weak and strong blobs
    # img_overlap_merged = np.logical_and(~img_mask_weak_merged, ~img_mask_strong_merged)*255
    # overlapping area (between blobs) to check if we merge blob twice

    img_concat = np.concatenate((img_weak, img_separator,
                                 img_strong, img_separator,
                                 img_combined, img_separator,
                                 img_weak_merged, img_separator,
                                 img_strong_merged, img_separator,
                                 img_combined_merged, img_separator), axis=1).astype('uint8')
    # plot image
    cv2.imshow('(1-3)Before merging, (4-6)After merging - weak blobs,strong blobs,strong+weak blobs', img_concat)
    cv2.resizeWindow('(1-3)Before merging, (4-6)After merging - weak blobs,strong blobs,strong+weak blobs', 1920, 720)
    cv2.waitKey(50)
    if i == len(dir_blob_) - 1:
        cv2.destroyAllWindows()


# draft
def splice_dir_blob_(dir_blobs):
    for i, _dir_blob in enumerate(dir_blobs):  # it may be redundant to loop all blobs here, use pop should be better here
        for fd in 0, 1:
            if fd: PP_ = _dir_blob.PPd_
            else:  PP_ = _dir_blob.PPm_
            PP_val = sum([PP.valt[fd] for PP in PP_])

            if PP_val - ave_splice > 0:  # high mPP pr dPP
                _top_P_ = _dir_blob.P__[0]
                _bottom_P_ = _dir_blob.P__[-1]
                for j, dir_blob in enumerate(dir_blobs):
                    if _dir_blob is not dir_blob:

                        top_P_ = dir_blob.P__[0]
                        bottom_P_ = dir_blob.P__[-1]
                        # test y adjacency:
                        if (_top_P_[0].y - 1 == bottom_P_[0].y) or (top_P_[0].y - 1 == _bottom_P_[0].y):
                            # test x overlap:
                            if (dir_blob.x0 - 1 < _dir_blob.xn and dir_blob.xn + 1 > _dir_blob.x0) \
                                    or (_dir_blob.x0 - 1 < dir_blob.xn and _dir_blob.xn + 1 > dir_blob.x0):
                                splice_2dir_blobs(_dir_blob, dir_blob)  # splice dir_blob into _dir_blob
                                dir_blobs[j] = _dir_blob

def splice_2dir_blobs(_blob, blob):
    # merge blob into _blob here

    # P__, box, dert and mask
    # not sure how to merge dert__ and mask__ yet
    x0 = min(_blob.box[0], blob.box[0])
    xn = max(_blob.box[1], blob.box[1])
    y0 = min(_blob.box[2], blob.box[2])
    yn = max(_blob.box[3], blob.box[3])
    _blob.box = (x0, xn, y0, yn)

    if (_blob.P__[0][0].y - 1 == blob.P__[-1][0].y):  # blob at top
        _blob.P__ = blob.P__ + _blob.P__
    else:  # _blob at top
        _blob.P__ = _blob.P__ + blob.P__

    # accumulate blob numeric params:
    # 'I', 'Dy', 'Dx', 'G', 'A', 'M', 'Sin_da0', 'Cos_da0', 'Sin_da1', 'Cos_da1', 'Ga', 'Mdx', 'Ddx', 'rdn', 'rng'
    for param_name in blob.numeric_params:
        _param = getattr(_blob,param_name)
        param = getattr(blob,param_name)
        setattr(_blob, param_name, _param+param)

    # accumulate blob list params:
    _blob.adj_blobs += blob.adj_blobs
    _blob.rlayers += blob.rlayers
    _blob.dlayers += blob.dlayers
    _blob.PPm_ += blob.PPm_
    _blob.PPd_ += blob.PPd_
    # _blob.valt[0] += blob.valt[0]; _blob.valt[1] += blob.valt[1] (we didnt assign valt yet)
    _blob.dir_blobs += blob.dir_blobs
    _blob.mlevels += blob.mlevels
    _blob.dlevels += blob.dlevels


def append_P(P__, P):  # pack P into P__ in top down sequence

    current_ys = [P_[0].y for P_ in P__]  # list of current-layer seg rows
    if P.y in current_ys:
        P__[current_ys.index(P.y)].append(P)  # append P row
    elif P.y > current_ys[0]:  # P.y > largest y in ys
        P__.insert(0, [P])
    elif P.y < current_ys[-1]:  # P.y < smallest y in ys
        P__.append([P])
    elif P.y < current_ys[0] and P.y > current_ys[-1]:  # P.y in between largest and smallest value
        for i, y in enumerate(current_ys):  # insert y if > next y
            if P.y > y: P__.insert(i, [P])  # PP.P__.insert(P.y - current_ys[-1], [P])


def copy_P(P, iPtype=None):  # Ptype =0: P is CP | =1: P is CderP | =2: P is CPP | =3: P is CderPP | =4: P is CaggPP

    if not iPtype:  # assign Ptype based on instance type if no input type is provided
        if isinstance(P, CPP):
            Ptype = 2
        elif isinstance(P, CderP):
            Ptype = 1
        elif isinstance(P, CP):
            Ptype = 0
    else:
        Ptype = iPtype

    uplink_layers, downlink_layers = P.uplink_layers, P.downlink_layers  # local copy of link layers
    P.uplink_layers, P.downlink_layers = [], []  # reset link layers
    roott = P.roott  # local copy
    P.roott = [None, None]
    if Ptype == 1:
        P_derP, _P_derP = P.P, P._P  # local copy of derP.P and derP._P
        P.P, P._P = None, None  # reset
    elif Ptype == 2:
        mseg_levels, dseg_levels = P.mseg_levels, P.dseg_levels
        rlayers, dlayers = P.rlayers, P.dlayers
        P__ = P.P__
        P.mseg_levels, P.dseg_levels, P.rlayers, P.dlayers, P.P__ = [], [], [], [], []  # reset
    elif Ptype == 3:
        PP_derP, _PP_derP = P.PP, P._PP  # local copy of derP.P and derP._P
        P.PP, P._PP = None, None  # reset
    elif Ptype == 4:
        gPP_, cPP_ = P.gPP_, P.cPP_
        rlayers, dlayers = P.rlayers, P.dlayers
        mlevels, dlevels = P.mlevels, P.dlevels
        P.gPP_, P.cPP_, P.rlayers, P.dlayers, P.mlevels, P.dlevels = [], [], [], [], [], []  # reset

    new_P = deepcopy(P)  # copy P with empty root and link layers, reassign link layers:
    new_P.uplink_layers += copy(uplink_layers)
    new_P.downlink_layers += copy(downlink_layers)

    # shallow copy to create new list
    P.uplink_layers, P.downlink_layers = copy(uplink_layers), copy(downlink_layers)  # reassign link layers
    P.roott = roott  # reassign root
    # reassign other list params
    if Ptype == 1:
        new_P.P, new_P._P = P_derP, _P_derP
        P.P, P._P = P_derP, _P_derP
    elif Ptype == 2:
        P.mseg_levels, P.dseg_levels = mseg_levels, dseg_levels
        P.rlayers, P.dlayers = rlayers, dlayers
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.P__ = copy(P__)
        new_P.mseg_levels, new_P.dseg_levels = copy(mseg_levels), copy(dseg_levels)
    elif Ptype == 3:
        new_P.PP, new_P._PP = PP_derP, _PP_derP
        P.PP, P._PP = PP_derP, _PP_derP
    elif Ptype == 4:
        P.gPP_, P.cPP_ = gPP_, cPP_
        P.rlayers, P.dlayers = rlayers, dlayers
        P.roott = roott
        new_P.gPP_, new_P.cPP_ = [], []
        new_P.rlayers, new_P.dlayers = copy(rlayers), copy(dlayers)
        new_P.mlevels, new_P.dlevels = copy(mlevels), copy(dlevels)

    return new_P


def CBlob2graph(dir_blob, fseg, Cgraph):

    PPm_ = dir_blob.PPm_; PPd_ = dir_blob.PPd_

    # init graph params
    players, fds, valt = [], [], [0,0]  # players, fds and valt
    alt_players, alt_fds, alt_valt = [], [], [0,0]
    plevels = [[[players, fds, valt],[alt_players, alt_fds, alt_valt]]]
    gPPm_, gPPd_ = [], []  # converted gPPs, node_ and alt_node_
    root_fds = PPm_[0].fds[:-1]  # root fds is the shorter fork?

    root = Cgraph(node_= gPPm_, alt_node_=gPPd_, plevels=plevels, rng = PPm_[0].rng,
                  fds = root_fds, rdn=dir_blob.rdn, x0=PPm_[0].x0, xn=PPm_[0].xn, y0=PPm_[0].y0, yn=PPm_[0].yn)

    for fd, (PP_, gPP_) in enumerate(zip([PPm_, PPd_], [gPPm_, gPPd_])):
        for PP in PP_:
            # should be summing alts when fd= 1?
            if fd:  # sum from altPP's players
                for altPP in PP.altPP_:
                    sum_players(alt_players, altPP.players)
                    alt_valt[0] += altPP.valt[0]; alt_valt[1] += altPP.valt[1]
                    alt_fds[:] = deepcopy(altPP.fds)
            else:  # sum from players
                sum_players(players, PP.players)
                valt[0] += PP.valt[0]; valt[1] += PP.valt[1]
                fds[:] = deepcopy(PP.fds)

            # compute rdn
            if fseg: PP = PP.roott[PP.fds[-1]]  # seg root
            PP_P_ = [P for P_ in PP.P__ for P in P_]  # PPs' Ps
            for altPP in PP.altPP_:  # overlapping Ps from each alt PP
                altPP_P_ = [P for P_ in altPP.P__ for P in P_]  # altPP's Ps
                alt_rdn = len(set(PP_P_).intersection(altPP_P_))
                PP.alt_rdn += alt_rdn  # count overlapping PPs, not bilateral, each PP computes its own alt_rdn
                root.alt_rdn += alt_rdn  # sum across PP_

            # convert and pack gPP
            gPP_ += [CPP2graph(PP, fseg, Cgraph)]
            root.x0=min(root.x0, PP.x0)
            root.xn=max(root.xn, PP.xn)
            root.y0=min(root.y0, PP.y0)
            root.yn=max(root.yn, PP.yn)
    return root

def CPP2graph(PP, fseg, Cgraph):

    alt_players, alt_fds = [], []
    alt_valt = [0, 0]

    if not fseg and PP.altPP_:  # seg doesn't have altPP_
        alt_fds = PP.altPP_[0].fds
        for altPP in PP.altPP_[1:]:  # get fd sequence common for all altPPs:
            for i, (_fd, fd) in enumerate(zip(alt_fds, altPP.fds)):
                if _fd != fd:
                    alt_fds = alt_fds[:i]
                    break
        for altPP in PP.altPP_:
            sum_players(alt_players, altPP.players[:len(alt_fds)])  # sum same-fd players only
            alt_valt[0] += altPP.valt[0];  alt_valt[1] += altPP.valt[1]
    # Cgraph: plevels ( caTree ( players ( caTree ( ptuples:
    players = []
    valt = [0,0]
    for i, (ptuples, alt_ptuples, fd) in enumerate(zip_longest(deepcopy(PP.players), deepcopy(alt_players), PP.fds, fillvalue=[])):
        cval, aval = 0,0
        for i, (ptuple, alt_ptuple) in enumerate(zip_longest(ptuples, alt_ptuples, fillvalue=None)):
            if alt_ptuple:
                if isinstance(ptuple, list):
                    aval += alt_ptuple[0].val
                else:
                    aval += alt_ptuple.val
                    alt_ptuples[i] = [alt_ptuple, [[[[[[]]]]]]]  # convert to Ptuple
            if ptuple:
                if isinstance(ptuple, list):  # already converted
                    cval += ptuple[0].val
                else:  # convert to Ptuple
                    cval += ptuple.val
                    ptuples[i] = [ptuple, [[[[[[]]]]]]]

            cfork = [ptuples, cval]  # can't be empty
            afork = [alt_ptuples, aval] if alt_ptuples else []
            caTree = [[cfork, afork], [cval, aval]]
            valt[0] += cval; valt[1] += aval
            players += [caTree]

    caTree = [[players, valt, deepcopy(PP.fds)]]  # pack single playerst
    plevel = [caTree, valt]
    x0 = PP.x0; xn = PP.xn; y0 = PP.y0; yn = PP.yn

    return Cgraph(node_=PP.P__, plevels=[plevel], angle=(yn-y0,xn-x0), sparsity=1.0, valt=valt, fds=[1], x0=x0, xn=xn, y0=y0, yn=yn)
    # 1st plevel fd is always der+?