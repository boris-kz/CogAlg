'''
Draft of blob-parallel version of frame_blobs
'''

def frame_blobs_parallel(frame_dert__):
    '''
    grow blob.dert__ by merging same-sign connected blobs, where remaining vs merged blobs are prior in y(x.
    after extension: merged blobs may overlap with remaining blob, check and remove redundant derts
    pseudo code below:
    '''
    blob__ = frame_dert__  # initialize one blob per dert
    blob__[:].append(frame_dert__[:])  # add open_derts = dert per blob
    frame_dert__[:].append(blob__[:])  # add blob ID to each dert, not sure expression is correct

    for blob in (blob__):  # initial blobs are open derts
        open_AND = 0  # counter of AND(open_dert.sign, unfilled_rim_dert.sign)
        new_open_derts = []
        for dert in blob.open_derts:  # open border: list of derts in dert__ that have unfilled derts in their 3x3 rim
            i = 0
            while i < 7:
                _y, _x = compute_rim(dert.y, dert.x, i)  # compute rim _y _x from dert y x, clock-wise from top-left
                i += 1
                if not (_y in blob.dert__ and _x in blob.dert__):
                    _dert = frame_dert__[_y][_x]  # load unfilled rim dert from frame.dert__
                    new_open_derts.append(_dert)
                    if dert.sign and _dert.sign:
                        open_AND += 1
                        merge_blobs(blob, _dert.blob)  # merge blob with blob currently assigned to _dert,
                        # remaining vs. merged blob is the one prior in y ( x
                        # new_open_derts += merged_blob_open_derts - overlap, re-assign blob ID per dert
                        # remove merged blob from next extension cycle
        if open_AND==0:
            terminate_blob(blob)  # remove blob from next extension cycle

        '''
        old dert and box- based version:
        concentric vs. y(x dert__ order for composition, oriented re-order for comp_P?
        
        rim_AND = 0  # counter of AND(last_rim_dert_sign, new_rim_dert_sign)

        for j, _side in enumerate(blob.dert__[-1]):  # dert__ is a sequence of rims, each has four sides: top, right, bottom, left
            if _side:  # array of dert signs, vs. [] if side has no blob-sign derts
                if j==1:
                    side = dert__[_side[0]-1, _side[1]]  # derts above top side, _side[0]: ys, _side[1]: xs
                    direction = -1
                elif j==2:
                    side = dert__[_side[0], _side[1]+1]  # derts to the right of right side
                    direction = 1
                elif j==3:
                    side = dert__[_side[0]+1, _side[1]]  # derts below bottom side
                    direction = 1
                else:
                    side = dert__[_side[0], _side[1]-1]  # derts to the left of left side
                    direction = -1
                side_AND = 0
                for (_sign, _dert), (sign, dert) in zip(_side[2], side[2]):
                    if _sign and sign:
                        side_AND += 1
                        merge_blobs(blob, dert[-1])  # merge dert.blob into current blob
                if side_AND:
                    rim_AND += 1
                    # blob.box[j-1] += direction  # extend y0 | yn | x0 | xn of blob.box
                    # this should actually be part of merge_blobs: the box is extended by the box of merged blob?
        if ~rim_AND:
            terminate_blob(blob)        
        '''

