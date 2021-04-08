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

    for i, blob in enumerate(dir_blob_):
        
        # local params
        merge_pair_ = [[],[]] # merging pair containing merging blob and target blob id
        merged_ids =[]        # contains ids of merged blob, so that we wouldn't merge it twice
        strong_ids = []       # contains ids of strong blob, to check if strong blob is already check through their adjacents
        blob = merge_adjacents_recursive(blob, blob.adj_blobs, merge_pair_, merged_ids, strong_ids)
 
        if blob: # blob is empty if it is weak
            if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
                blob.fsliced = True
                slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob
            iblob.dir_blobs.append(blob)

        # To visualizate merging process
        cv2.namedWindow('gray=weak, white=strong', cv2.WINDOW_NORMAL)
        img_blobs   = mask__.copy().astype('float')*0
        # draw weak blobs
        for dir_blob in dir_blob_:
            fweak = dir_eval(dir_blob) # direction eval on the blob
            y0,yn,x0,xn = dir_blob.box
            if fweak:
                img_blobs[y0:yn,x0:xn] += ((~dir_blob.mask__) * 90) .astype('float')
            else:
                img_blobs[y0:yn,x0:xn] += ((~dir_blob.mask__) * 255) .astype('float')
        # draw strong blobs
        img_mask = np.ones_like(mask__).astype('bool')
        for dir_blob in iblob.dir_blobs:
            fweak = dir_eval(dir_blob) # direction eval on the blob
            y0,yn,x0,xn = dir_blob.box
            if ~fweak:
                img_mask[y0:yn,x0:xn] = np.logical_and(img_mask[y0:yn,x0:xn], dir_blob.mask__)
        img_blobs += ((~img_mask).astype('float')*255)
        img_blobs[img_blobs>100] = 255
        cv2.imshow('gray=weak, white=strong',img_blobs.astype('uint8'))
        cv2.resizeWindow('gray=weak, white=strong', 640, 480)
        cv2.waitKey(50)          
    cv2.destroyAllWindows()


def merge_adjacents_recursive_draft(blob, adj_blobs):

    if dir_eval(blob):  # returns directionality fweak, no re-evaluation until all adjacent weak blobs are merged

        if blob in adj_blobs[0]: adj_blobs[0].remove(blob)  # remove current blob from adj adj blobs (assigned bilaterally)
        merged_adj_blobs = [[], []]  # adj_blob_, pose_

        for (adj_blob, pose) in zip(*adj_blobs):  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
            if dir_eval(adj_blob):  # also directionally weak, merge adj blob to blob
                blob = merge_blobs(blob, adj_blob)

                for i, adj_adj_blob in enumerate(adj_blob.adj_blobs[0]):
                    if adj_adj_blob not in merged_adj_blobs[0]:  # add adj adj_blobs to merged_adj_blobs
                        merged_adj_blobs[0].append(adj_blob.adj_blobs[0][i])
                        merged_adj_blobs[1].append(adj_blob.adj_blobs[1][i])

        if merged_adj_blobs:
            blob = merge_adjacents_recursive_draft(blob, merged_adj_blobs)

        if dir_eval(blob):  # if merged blob is still weak，merge it into the stronger of vert_adj_blobs and lat_adj_blobs:

            dir_adj_blobs = [[0, [], []], [0, [], []]]  # vert_adj_blobs and lat_adj_blobs, each: dir_val, adj_blob_, pose_
            for adj_blob in merged_adj_blobs[0]:

                if adj_blob.sign:  # add to vertical adj_blobs
                    dir_adj_blobs[0][0] += adj_blob.dir_val  # sum dir_val
                    dir_adj_blobs[0][1].append(adj_blob.adj_blobs[0][i])
                    dir_adj_blobs[0][2].append(adj_blob.adj_blobs[1][i])
                else:  # add to lateral adj_blobs
                    dir_adj_blobs[1][0] += adj_blob.dir_val  # sum dir_val
                    dir_adj_blobs[1][1].append(adj_blob.adj_blobs[0][i])
                    dir_adj_blobs[1][2].append(adj_blob.adj_blobs[1][i])

                if blob.dir_val > 0:  # vertical
                    dir_adj_blobs[0][0] += blob.dir_val
                else:
                    dir_adj_blobs[1][0] += blob.dir_val

                '''merge final_weak_blob with all remaining strong blobs in the stronger dir_adj_blobs: '''

                if dir_adj_blobs[0][0] > dir_adj_blobs[1][0]:
                    merge_final_weak_blob(blob, dir_adj_blobs[0][1])  # to be defined
                else:
                    merge_final_weak_blob(blob, dir_adj_blobs[1][1])  # to be defined


def merge_adjacents_recursive(blob, adj_blobs, merge_pair_, merged_ids, strong_ids):

    # remove current blob reference in adj' adj blobs, since adj blob are assigned bilaterally
    if blob in adj_blobs[0]: adj_blobs[0].remove(blob)
    _fweak = dir_eval(blob) # direction eval on the input blob
    
    # local params
    adj_blob_list_ = [[],[]]
    merged_list_ = []

    for (adj_blob,pose) in zip(*adj_blobs):  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
        
        fweak = dir_eval(adj_blob) # direction eval on the adjacent blob
        
        # blob is weak and adjacent blob is strong, update adjacent reference
        if _fweak and not fweak: 
            
            # if strong adjacent blob is already checked through their adjacents, we will merge current blob to the strong adjacent blob after checking through all current adjacents
            if adj_blob in strong_ids:
                merged_list_.append(adj_blob)
            else: 
                strong_ids.append(adj_blob.id)     # update strong ids
                merge_pair_[0].append(blob)        # update merging pair merging blob
                merge_pair_[1].append(adj_blob.id) # update merging pair target id
                
            if blob not in adj_blob.adj_blobs[0]:   # if current blob not in adj_blob's adj_blobs list
                adj_blob.adj_blobs[0].append(blob)  # update weak blob into strong adj_blob's adj_blobs
                adj_blob.adj_blobs[1].append(pose)  # update pose

        # blob is strong or weak, but adjacent is weak, merge adjacent blob to blob
        elif fweak and adj_blob.id not in merged_ids:  

            # if blob is strong, update strong ids 
            if not _fweak: strong_ids.append(blob.id)

            if adj_blob in merge_pair_[0]:              # if adjacent blob is already paired up with strong blob
                if blob.id in merge_pair_[1]:           # if the target strong blob is current blob, merge them
                    blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
            else:                                       # if adjacent blob is not having strong blob pair yet
                blob = merge_blobs(blob, adj_blob)      # merge dert__ and accumulate params

            merged_ids.append(adj_blob.id)  # update adjacent blob id into the merged ids

            if pose != 1: # if adjacent is not internal
                for i,adj_adj_blob in enumerate(adj_blob.adj_blobs[0]):
                    if adj_adj_blob not in adj_blob_list_[0] and adj_adj_blob is not blob and  adj_adj_blob.id not in merged_ids :
                        adj_blob_list_[0].append(adj_blob.adj_blobs[0][i]) # add adj adj_blobs to search list if they are merged
                        adj_blob_list_[1].append(adj_blob.adj_blobs[1][i])  
                
        _fweak = dir_eval(blob) # direction eval again on the merged blob 

    # if there is merge list, merge them
    if (merged_list_): merge_blobs(merged_list_[0],blob)
    
    # if blob is not in merging pair, continue searching and merging from adjacent's adjacents
    elif not blob in merge_pair_[0]:
        # if merged blob is still weak， continue searching and merging with the merged blob's adj blobs
        # else they will stop merging adjacent blob
        if _fweak:
            if adj_blob_list_[0]:
                blob = merge_adjacents_recursive(blob, adj_blob_list_,merge_pair_,merged_ids,strong_ids) 
                if blob: 
                    _fweak = dir_eval(blob) # direction eval again on the merged blob 
                    if not _fweak: 
                        return blob # return only strong blob
    
        else: 
            return blob # return only strong blob


def dir_eval(blob):
    
    rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
    if abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
        fweak = 1
    else: fweak = 0
    
    return fweak
    
    
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