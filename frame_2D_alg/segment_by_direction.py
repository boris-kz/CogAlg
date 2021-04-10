'''
- Segment input blob into dir_blobs by primary direction of kernel gradient: dy>dx
- Merge weakly directional dir_blobs, with dir_val < cost of comp_slice_
- Evaluate merged blobs for comp_slice_: if blob.M > ave_M
'''

import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob
import cv2

flip_ave = 10
ave_dir_val = 50
ave_M = -500  # high negative ave M for high G blobs

def segment_by_direction(iblob, verbose=False):

    dert__ = list(iblob.dert__)
    mask__ = iblob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]

    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:
    dir_blob_, idmap, adj_pairs = \
        flood_fill(dert__, abs(dy__) > abs(dx__), verbose=False, mask__=mask__, blob_cls=CBlob, fseg=True, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    for blob in dir_blob_:
        rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
        blob.dir_val = abs(blob.Dert.G * rD) - ave_dir_val  # direction strength value
#   for debugging:
#   from draw_frame_blobs import visualize_blobs
#   visualize_blobs(idmap, dir_blob_)

    merged_ids = []  # ids of merged adjacent blobs, to skip in the rest of dir_blobs

    for i, blob in enumerate(dir_blob_):
        if blob.id not in merged_ids:
            blob = merge_adjacents_recursive(blob, blob.adj_blobs, merged_ids)

            if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
                blob.fsliced = True
                slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob
            iblob.dir_blobs.append(blob)

        for dir_blob in iblob.dir_blobs[:]: # strong blob is merged to another blob, remove it
            if dir_blob.id in merged_ids:
                iblob.dir_blobs.remove(dir_blob)

        visualize_merging_process(iblob, dir_blob_,mask__, i)


def merge_adjacents_recursive(blob, adj_blobs, merged_ids):

    if blob.dir_val < 0:  # directionally weak blob, no re-evaluation until all adjacent weak blobs are merged

        if blob in adj_blobs[0]: adj_blobs[0].remove(blob)  # remove current blob from adj adj blobs (assigned bilaterally)
        merged_adj_blobs = [[], []]  # adj_blob_, pose_

        for (adj_blob, pose) in zip(*adj_blobs):  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
            if (adj_blob.dir_val < 0) and adj_blob.id not in merged_ids:  # also directionally weak, merge adj blob to blob
                blob = merge_blobs(blob, adj_blob)
                merged_ids.append(adj_blob.id)

                for i, adj_adj_blob in enumerate(adj_blob.adj_blobs[0]):
                    # recursively add adj_adj_blobs to merged_adj_blobs:
                    if adj_adj_blob not in merged_adj_blobs[0] and adj_adj_blob is not blob and adj_adj_blob.id not in merged_ids:
                        merged_adj_blobs[0].append(adj_blob.adj_blobs[0][i])
                        merged_adj_blobs[1].append(adj_blob.adj_blobs[1][i])

        if merged_adj_blobs[0]:
            blob = merge_adjacents_recursive(blob, merged_adj_blobs, merged_ids)

        if blob.dir_val < 0:  # if merged blob is still weak，merge it into the stronger of vert_adj_blobs and lat_adj_blobs:

            dir_adj_blobs = [[0, [], []], [0, [], []]]  # lat_adj_blobs and vert_adj_blobs, each: dir_val, adj_blob_, pose_
            dir_adj_blobs[blob.sign][0] += blob.dir_val  # sum blob.dir_val into same-direction (vertical or lateral) dir_adj_blobs

            for adj_blob in merged_adj_blobs[0]:
                # add adj_blob into same-direction-sign adj_blobs: 1 if vertical, 0 if lateral:
                dir_adj_blobs[adj_blob.sign][0] += adj_blob.dir_val  # sum dir_val (abs)
                dir_adj_blobs[adj_blob.sign][1].append(adj_blob)  # buffer adj_blobs
                dir_adj_blobs[adj_blob.sign][2].append(adj_blob)  # buffer adj_blob poses, not needed?

            # merge final_weak_blob with all remaining strong blobs in the stronger of dir_adj_blobs:
            for adj_blob in dir_adj_blobs[ dir_adj_blobs[0][0] > dir_adj_blobs[1]] [1]:

                if adj_blob.id not in merged_ids and adj_blob is not blob:
                    blob = merge_blobs(blob, adj_blob)
                    merged_ids.append(adj_blob.id)

            '''
            not needed:
            final strong-blob merging is not recursive:
            for i, adj_adj_blob in enumerate(adj_blob.adj_blobs[0]):
                if adj_adj_blob not in dir_adj_blobs[adj_blob.sign][1]:
                    dir_adj_blobs[adj_blob.sign][1].append(adj_blob.adj_blobs[0][i])  # buffer adj_blobs
                    dir_adj_blobs[adj_blob.sign][2].append(adj_blob.adj_blobs[1][i])  # buffer adj_blob poses, not needed?
                    
            if dir_adj_blobs[0][0] > dir_adj_blobs[1][0]:
                merge_final_weak_blob(blob, dir_adj_blobs[0][1], merged_ids)  # merge blob with all dir_adj blobs
                blob.adj_blobs = dir_adj_blobs[1]  # remaining adj_blobs
            else:
                merge_final_weak_blob(blob, dir_adj_blobs[1][1], merged_ids)  # merge blob with all dir_adj blobs
                blob.adj_blobs = dir_adj_blobs[0]  # remaining adj_blobs
            
            def merge_final_weak_blob(blob, adj_blobs, merged_ids):

                for adj_blob in adj_blobs:
                    if adj_blob.id not in merged_ids and adj_blob is not blob:
                        blob = merge_blobs(blob, adj_blob)
                        merged_ids.append(adj_blob.id)
                        
            def dir_eval(blob):  # not used

                rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
                if  abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
                    fweak = 1
                else: fweak = 0
            '''
    return blob


def merge_blobs(blob, adj_blob):  # merge adj_blob into blob
    '''
    added AND operation for the merging process, still need to be optimized further
    '''
    # 0 = overlap between blob and adj_blob
    # 1 = overlap and blob is in adj blob
    # 2 = overlap and adj blob is in blob
    floc = 0
    # accumulate blob Dert
    blob.accumulate(**{param:getattr(adj_blob.Dert, param) for param in adj_blob.Dert.numeric_params})

    _y0, _yn, _x0, _xn = blob.box
    y0, yn, x0, xn = adj_blob.box

    if (_y0<=y0) and (_yn>=yn) and (_x0<=x0) and (_xn>=xn): # adj blob is inside blob
        floc = 2
        cy0, cyn, cx0, cxn =  blob.box # y0, yn, x0, xn for combined blob is blob box
    elif (y0<=_y0) and (yn>=_yn) and (x0<=_x0) and (xn>=_xn): # blob is inside adj blob
        floc = 1
        cy0, cyn, cx0, cxn =  adj_blob.box # y0, yn, x0, xn for combined blob is adj blob box
    else:
        # y0, yn, x0, xn for combined blob and adj blob box
        cy0 = min([blob.box[0],adj_blob.box[0]])
        cyn = max([blob.box[1],adj_blob.box[1]])
        cx0 = min([blob.box[2],adj_blob.box[2]])
        cxn = max([blob.box[3],adj_blob.box[3]])

    # offsets from combined box
    y0_offset = blob.box[0]-cy0
    x0_offset = blob.box[2]-cx0
    adj_y0_offset = adj_blob.box[0]-cy0
    adj_x0_offset = adj_blob.box[2]-cx0

    if floc == 1:  # blob is inside adj blob
        # extended mask is adj blob's mask, AND extended mask with blob mask
        extended_mask__ = adj_blob.mask__
        extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = \
        np.logical_and(blob.mask__, extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)])
        # if blob is inside adj blob, blob derts should be already in adj blob derts
        extended_dert__ = adj_blob.dert__

    elif floc == 2: # adj blob is inside blob
        # extended mask is blob's mask, AND extended mask with adj blob mask
        extended_mask__ = blob.mask__
        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])
        # if adj blob is inside blob, adj blob derts should be already in blob derts
        extended_dert__ = blob.dert__

    else:
        # create extended mask from combined box
        extended_mask__ = np.ones((cyn-cy0,cxn-cx0)).astype('bool')
        # AND extended mask with blob mask
        extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = \
        np.logical_and(blob.mask__, extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)])

        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])

        # AND extended mask with adj blob mask
        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])

        # create extended derts from combined box
        extended_dert__ = [np.zeros((cyn-cy0,cxn-cx0)) for _ in range(len(blob.dert__))]
        for i in range(len(blob.dert__)):
            extended_dert__[i][y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = blob.dert__[i]
            extended_dert__[i][adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = adj_blob.dert__[i]

    # update dert, mask , box and sign
    blob.dert__ = extended_dert__
    blob.mask__ = extended_mask__
    blob.box = [cy0,cyn,cx0,cxn]
    blob.sign = abs(blob.Dert.Dy)>abs(blob.Dert.Dx)

    # update adj blob 'adj blobs' adj_blobs reference from pointing adj blob into the merged blob
    for i, adj_adj_blob1 in enumerate(adj_blob.adj_blobs[0]):            # loop adj blobs of adj blob
        for j, adj_adj_blob2 in enumerate(adj_adj_blob1.adj_blobs[0]):   # loop adj blobs from adj blobs of adj blob
            if adj_adj_blob2 is adj_blob and adj_adj_blob1 is not blob : # if adj blobs from adj blobs of adj blob is adj blob, update reference to the merged blob
                adj_adj_blob1.adj_blobs[0][j] = blob

    return blob


def accum_dir_blob_Dert(blob, dert__, y, x):
    blob.Dert.I += dert__[0][y, x]
    blob.Dert.Dy += dert__[1][y, x]
    blob.Dert.Dx += dert__[2][y, x]
    blob.Dert.G += dert__[3][y, x]
    blob.Dert.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dert.Dyy += dert__[5][y, x]
        blob.Dert.Dyx += dert__[6][y, x]
        blob.Dert.Dxy += dert__[7][y, x]
        blob.Dert.Dxx += dert__[8][y, x]
        blob.Dert.Ga += dert__[9][y, x]
        blob.Dert.Ma += dert__[10][y, x]


def visualize_merging_process(iblob, dir_blob_, mask__, i):

    cv2.namedWindow('(1)weak blobs, (2)strong blobs, (3)strong+weak blobs, (4)overlapping between weak and strongblob (error)', cv2.WINDOW_NORMAL)

    img_mask_strong = np.ones_like(mask__).astype('bool')
    img_mask_weak = np.ones_like(mask__).astype('bool')

    # get mask of dir blobs
#    for dir_blob in dir_blob_:
#        y0, yn, x0, xn = dir_blob.box
#        # direction eval on the blob
#        if dir_eval(dir_blob): # weak blob
#            img_mask_weak[y0:yn, x0:xn] = np.logical_and(img_mask_weak[y0:yn, x0:xn], dir_blob.mask__)
#        else:
#            img_mask_strong[y0:yn, x0:xn] = np.logical_and(img_mask_strong[y0:yn, x0:xn], dir_blob.mask__)

    # get mask of merged blobs
    for dir_blob in iblob.dir_blobs:
        y0, yn, x0, xn = dir_blob.box
        # direction eval on the blob
        if dir_blob.dir_val < 0:   # weak blob
            img_mask_strong[y0:yn, x0:xn] = np.logical_and(img_mask_strong[y0:yn, x0:xn], dir_blob.mask__)
        else:  # strong blob
            img_mask_weak[y0:yn, x0:xn] = np.logical_and(img_mask_weak[y0:yn, x0:xn], dir_blob.mask__)

    img_separator = np.ones((mask__.shape[0],3)) * 45         # separator
    img_weak = ((~img_mask_weak)*90).astype('uint8')          # weak blobs
    img_strong = ((~img_mask_strong)*255).astype('uint8')     # strong blobs
    img_combined = img_weak + img_strong                      # merge weak and strong blobs
    img_overlap = np.logical_and(~img_mask_weak, ~img_mask_strong)*255 # overlapping area (between blobs) to check if we merge blob twice
    img_concat = np.concatenate((img_weak, img_separator, img_strong, img_separator, img_combined, img_separator, img_overlap), axis=1).astype('uint8')


    # plot image
    cv2.imshow('(1)weak blobs, (2)strong blobs, (3)strong+weak blobs, (4)overlapping between weak and strongblob (error)', img_concat)
    cv2.resizeWindow('(1)weak blobs, (2)strong blobs, (3)strong+weak blobs, (4)overlapping between weak and strongblob (error)', 1280, 720)
    cv2.waitKey(50)
    if i == len(dir_blob_) - 1:
        cv2.destroyAllWindows()