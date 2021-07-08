'''
- Segment input blob into dir_blobs by primary direction of kernel gradient: dy>dx
- Merge weakly directional dir_blobs, with dir_val < cost of comp_slice_
- Evaluate merged blobs for comp_slice_: if blob.M > ave_M
'''

import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob
import cv2
from copy import deepcopy

flip_ave = 10
ave_dir_val = 50
ave_M = -500  # high negative ave M for high G blobs

def segment_by_direction(iblob, **kwargs):

    dert__ = list(iblob.dert__)
    mask__ = iblob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]
    verbose = kwargs.get('verbose')
    render = kwargs.get('render')

    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:
    dir_blob_, idmap, adj_pairs = \
        flood_fill(dert__, abs(dy__) > abs(dx__), verbose=verbose, mask__=mask__, blob_cls=CBlob, fseg=True)
    assign_adjacents(adj_pairs, CBlob)  # fseg=True: skip adding the pose

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
            if adj_adj_blob2 is adj_blob and adj_adj_blob1 is not blob : # if adj blobs from adj blobs of adj blob is adj blob, update reference to the merged blob
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
    # img_overlap_merged = np.logical_and(~img_mask_weak_merged, ~img_mask_strong_merged)*255 # overlapping area (between blobs) to check if we merge blob twice

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