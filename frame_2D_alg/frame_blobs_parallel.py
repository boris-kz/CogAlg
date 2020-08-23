'''
Draft of blob-parallel version of frame_blobs
'''

def frame_blobs_parallel(dert__):

    blob__ = dert__  # initialize one blob per dert
    dert__[:].append(blob__[:])  # add blob ID to each dert, not sure expression is correct

    for i, blob in enumerate(blob__):
        '''
        grow blob.dert__: layers of rims, rectangle unless individual dert coords?
        y(x priority of merge_blobs, 
        merge after extension: current rim of merged blob may overlap with that of remaining blob? 
        concentric vs. y(x dert__ order for composition, oriented re-order for comp_P? 
        '''
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
