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
    frame_dert__[:].append(blob__[:])  # add blob ID to each dert, not sure expression is correct
    for blob, dert in zip(blob__, frame_dert__):
        blob = [blob]
        blob += [[dert], []]  # add open_derts and empty connected_blobs to each blob

    # while open_derts: iterate blob extension cycle:
    for blob in (blob__):  # single extension cycle, initial blobs are open derts

        merge_blobs(blob, blob.connected_blobs)
        # merge connected blobs added in last extension cycle, re-assign blob ID per dert
        # connected blobs of these connected blobs are merged in next cycle, etc.:
        blob.connected_blobs = []
        open_AND = 0  # counter of AND(open_dert.sign, unfilled_rim_dert.sign)
        new_open_derts = []
        for dert in blob.open_derts:  # open_derts: list of derts in dert__ with unfilled derts in their 3x3 rim
            i = 0
            while i < 7:
                _y, _x = compute_rim(dert.y, dert.x, i)  # compute rim _y _x from dert y x, clock-wise from top-left
                i += 1
                if not (_y in blob.dert__ and _x in blob.dert__):
                    _dert = frame_dert__[_y][_x]  # load unfilled rim dert from frame.dert__
                    new_open_derts.append(_dert)
                    if dert.sign and _dert.sign:
                        open_AND += 1
                        add_blobs(blob, _dert.blob)
                        # blob.connected_blobs += blob currently assigned to _dert, excluding connected_blobs of that blob
                        # remaining vs. added blob has lower blob ID, remove added blob from next cycle
                        # new_open_derts += connected_blob.open_derts - overlap
        if open_AND==0:
            terminate_blob(blob)  # remove blob from next extension cycle

