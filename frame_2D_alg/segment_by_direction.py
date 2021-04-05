'''
- Segment input blob into dir_blobs by primary direction of kernel gradient: dy>dx
- Merge weakly directional dir_blobs, with dir_val < cost of comp_slice_
- Evaluate merged blobs for comp_slice_: if blob.M > ave_M
'''

import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob

flip_ave = 10
ave_dir_val = 50
ave_M = -500  # high negative ave M for high G blobs

def segment_by_direction(blob, verbose=False):

    dert__ = list(blob.dert__)
    mask__ = blob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]

    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:
    dir_blob_, idmap, adj_pairs = \
        flood_fill(dert__, abs(dy__) > abs(dx__), verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    for blob in dir_blob_:
        blob = merge_adjacents_recursive(blob, blob.adj_blobs)

        if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
            blob.fsliced = True
            slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob

        blob.dir_blobs.append(blob)


def merge_adjacents_recursive(blob, adj_blobs):

    _fweak = directionality_eval(blob)  # direction eval on the input blob

    for adj_blob in adj_blobs[0]:  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
        # if adj_blob was merged, it should be replaced by a ref to merged blob, so we don't need fmerged?:
        if not adj_blob.fmerged:
            fweak = directionality_eval(blob)  # direction eval on the adjacent blob

            if _fweak: # blob is weak, merge blob to adj blob
                blob.fmerged = True
                blob = merge_blobs(adj_blob, blob)  # merge dert__ and accumulate params
                
            elif fweak:  # adj blob is weak, merge adj blob to blob
                adj_blob.fmerged = True
                blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
            
            _fweak = directionality_eval(blob)  # if merged blob is still weak, continue searching and merging
            if _fweak: merge_adjacents_recursive(blob, adj_blob.adj_blobs) 
            
    return blob


def directionality_eval(blob):
    
    rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
    if abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
        fweak = 1
    else: fweak = 0
    
    return fweak
    
    
def merge_blobs(blob, adj_blob):  # merge adj_blob into blob

    # accumulate blob Dert
    blob.accumulate(**{param:getattr(adj_blob.Dert, param) for param in adj_blob.Dert.numeric_params})
    # y0, yn, x0, xn for combined blob and adj blob box
    y0 = min([blob.box[0],adj_blob.box[0]])
    yn = max([blob.box[1],adj_blob.box[1]])
    x0 = min([blob.box[2],adj_blob.box[2]])
    xn = max([blob.box[3],adj_blob.box[3]])
    # offsets from combined box
    y0_offset = blob.box[0]-y0
    x0_offset = blob.box[2]-x0
    adj_y0_offset = adj_blob.box[0]-y0
    adj_x0_offset = adj_blob.box[2]-x0

    # create extended mask from combined box
    extended_mask__ = np.ones((yn-y0,xn-x0)).astype('bool')
    for y in range(blob.mask__.shape[0]): # blob mask
        for x in range(blob.mask__.shape[1]):
            if not blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+y0_offset,x+x0_offset] = blob.mask__[y,x]
    for y in range(adj_blob.mask__.shape[0]): # adj_blob mask
        for x in range(adj_blob.mask__.shape[1]):
            if not adj_blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+adj_y0_offset,x+adj_x0_offset] = adj_blob.mask__[y,x]

    # create extended derts from combined box
    extended_dert__ = [np.zeros((yn-y0,xn-x0)) for _ in range(len(blob.dert__))]
    for i in range(len(blob.dert__)):
        for y in range(blob.dert__[i].shape[0]): # blob derts
            for x in range(blob.dert__[i].shape[1]):
                if not blob.mask__[y,x]: # replace when mask = false
                    extended_dert__[i][y+y0_offset,x+x0_offset] = blob.dert__[i][y,x]
        for y in range(adj_blob.dert__[i].shape[0]): # adj_blob derts
            for x in range(adj_blob.dert__[i].shape[1]):
                if not adj_blob.mask__[y,x]: # replace when mask = false
                    extended_dert__[i][y+adj_y0_offset,x+adj_x0_offset] = adj_blob.dert__[i][y,x]

    # update dert, mask and box
    blob.dert__ = extended_dert__
    blob.mask__ = extended_mask__
    blob.box = [y0,yn,x0,xn]

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
