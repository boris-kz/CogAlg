'''
Draft of blob-parallel version of frame_blobs
'''

def frame_blobs_parallel(frame_dert__):
    '''
    grow blob.dert__ by merging same-sign connected blobs, where remaining vs merged blob is prior in y(x.
    after extension: merged blobs may overlap with remaining blob, check and remove redundant derts
    pseudo code below:
    '''
    blob__ = frame_dert__  # initialize one blob per dert
    frame_dert__[:].append(blob__[:])  # add blob ID to each dert, not sure expression is correct
    i = 0
    for blob, dert in zip(blob__, frame_dert__):
        blob[i] = [[blob],[dert]]  # add open_derts and empty connected_blobs to each blob
        i += 1
    '''
    blob flood-filling by single layer of open_derts per cycle, with stream compaction in next cycle: 
    while yn-y0 < Y or xn-x0 < X: 
        generic box < frame: iterate blob and box extension cycle by rim per open dert
        check frame_dert__ to assign lowest blob ID in each dert, 
        remove all other blobs from the stream
    '''
    for blob in (blob__):  # single extension cycle, initial blobs are open derts

        blob = merge_open_derts(blob)  # merge last-cycle new_open_derts, re-assign blob ID per dert
        open_AND = 0  # counter of AND(open_dert.sign, unfilled_rim_dert.sign)
        new_open_derts = []
        for dert in blob.open_derts:  # open_derts: list of derts in dert__ with unfilled derts in their 3x3 rim
            i = 0
            while i < 7:
                _y, _x = compute_rim(dert.y, dert.x, i)  # compute rim _y _x from dert y x, clock-wise from top-left
                i += 1
                if not (_y in blob.dert__ and _x in blob.dert__):
                    _dert = frame_dert__[_y][_x]  # load unfilled rim dert from frame.dert__
                    if dert.sign and _dert.sign:
                        open_AND += 1
                        new_open_derts.append(_dert)
                        frame_dert__[_y][_x].append(blob.ID)
                        # multiple blob IDs maybe assigned to each frame_dert, with the lowest one selected in next cycle
        if open_AND==0:
            terminate_blob(blob)  # remove blob from next extension cycle

