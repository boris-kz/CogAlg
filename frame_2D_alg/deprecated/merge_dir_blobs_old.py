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

