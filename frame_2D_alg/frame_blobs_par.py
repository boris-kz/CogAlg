'''
Draft of blob-parallel version of frame_blobs
'''

def frame_blobs_parallel_pseudo(frame_dert__):
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


'''
Chee's implementation of blob-parallel frame_blobs:
'''

from class_cluster import ClusterStructure, NoneType
from utils import (
    pairwise,
    imread, )
import multiprocessing as mp
from frame_blobs_ma import comp_pixel
from time import time
import numpy as np
from utils import *
from multiprocessing.pool import ThreadPool
from matplotlib import pyplot as plt

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

class CDert(ClusterStructure):
    # Derts
    i = int
    g = int
    dy = int
    dx = int
    # other data
    sign = NoneType
    x_coord = int
    y_coord = int
    blob_ids = list
    blob_id_min = int
    fopen = bool

class CBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool


def generate_blobs(dert, y, x):
    '''
    generate dert & blob based on given dert and their location
    '''

    dert_i = CDert(i=dert[0], g=dert[1] - ave, dy=dert[2], dx=dert[3],
                   x_coord=x, y_coord=y, sign=dert[1] - ave > 0, fopen=1)

    blob = CBlob(I=dert[0], G=dert[1] - ave, Dy=dert[2], Dx=dert[3],
                 dert_coord_=[[y, x]], sign=dert[1] - ave > 0)

    dert_i.blob_ids.append(blob.id)

    return [blob, dert_i]


def check_rim(blobdert, blobdert_rim):
    '''
    check connectivity of blob's rim derts and update each dert's ids
    '''
    c_blob = blobdert[0]  # current blob
    c_dert = blobdert[1]  # current dert

    if c_blob.sign:  # for + sign, check 8 ortho+diag rims
        for rim in blobdert_rim:
            if rim:  # not empty rim
                blob, dert = rim

                # check open dert flag and same sign
                if c_dert.sign == dert.sign:
                    c_dert.blob_ids.append(blob.id)  # add same sign rim blob's id
                    c_dert.blob_ids += dert.blob_ids  # chain the rim dert' blob' rim dert's blobs ids
                    c_dert.blob_ids = list(set(c_dert.blob_ids))  # remove duplicated ids

    else:  # for - sign, check 4 ortho rims
        for i, rim in enumerate(blobdert_rim):
            if rim and (i + 1) % 2:  # not empty rim and ortho rim
                blob, dert = rim

                # check open dert flag and same sign
                if c_dert.sign == dert.sign:
                    c_dert.blob_ids.append(blob.id)  # add same sign rim blob's id
                    c_dert.blob_ids += dert.blob_ids  # chain the rim dert' blob' rim dert's blobs ids
                    c_dert.blob_ids = list(set(c_dert.blob_ids))  # remove duplicated ids
    # min of ids
    c_dert.blob_id_min = min(c_dert.blob_ids)
    return [c_blob, c_dert]


# there could be a better way to replace this function with parallel process ,need to think about it
def get_rim_dert(blobdert__, height, width):
    '''
    generate rims' blobs & derts
    '''

    blobdert_rim_ = []
    for y in range(height):
        for x in range(width):
            # topleft
            if y - 1 >= 0 and x - 1 >= 0:
                blobdert_topleft = blobdert__[y - 1][x - 1]
            else:
                blobdert_topleft = []
            # top
            if y - 1 >= 0:
                blobdert_top = blobdert__[y - 1][x]
            else:
                blobdert_top = []
            # topright
            if y - 1 >= 0 and x + 1 <= width - 1:
                blobdert_topright = blobdert__[y - 1][x + 1]
            else:
                blobdert_topright = []
            # right
            if x + 1 <= width - 1:
                blobdert_right = blobdert__[y][x + 1]
            else:
                blobdert_right = []
            # botright
            if y + 1 <= height - 1 and x + 1 <= width - 1:
                blobdert_botright = blobdert__[y + 1][x + 1]
            else:
                blobdert_botright = []
            # bot
            if y + 1 <= height - 1:
                blobdert_bot = blobdert__[y + 1][x]
            else:
                blobdert_bot = []
            # botleft
            if y + 1 <= height - 1 and x - 1 >= 0:
                blobdert_botleft = blobdert__[y + 1][x - 1]
            else:
                blobdert_botleft = []
            # left
            if x - 1 >= 0:
                blobdert_left = blobdert__[y][x - 1]
            else:
                blobdert_left = []

            blobdert_rim_.append([blobdert_topleft, blobdert_top, blobdert_topright, blobdert_right,
                                  blobdert_botright, blobdert_bot, blobdert_botleft, blobdert_left])

    return blobdert_rim_


def frame_blobs_parallel_Chee(dert__):
    '''
    grow blob.dert__ by merge_blobs, where remaining vs merged blobs are prior in y(x.
    merge after extension: merged blobs may overlap with remaining blob, check and remove redundant derts
    '''

    pool = ThreadPool(mp.cpu_count())  # initialize pool of threads

    height, width = dert__[0].shape  # height and width of image

    # generate all x and y coordinates
    dert_coord = [[y, x] for x in range(width) for y in range(height)]
    y_ = [y for y, x in dert_coord]
    x_ = [x for y, x in dert_coord]

    # get each dert from coordinate
    dert_ = [dert__[:, y, x] for y, x in dert_coord]

    # (parallel process) generate instance of derts and blobs
    blobdert_ = pool.starmap(generate_blobs, zip(dert_, y_, x_))

    ite_count = 0  # count of iteration
    id_map_np_prior___ = np.zeros((height, width))  # prior id_map, to check when to stop iteration
    f_ite = 1  # flag to continue iteration, 0 = stop, 1 = continue

    ## 1st iteration ##

    # convert blobdert_ to 2D array
    blobdert__ = [blobdert_[i:i + width] for i in range(0, len(blobdert_), width)]
    # get each dert rims (8 rims)
    blobdert_rim_ = get_rim_dert(blobdert__, height, width)
    # (parallel process) get updated id in each dert
    blobdert_ = pool.starmap(check_rim, zip(blobdert_, blobdert_rim_))
    # get map of min id
    id_map_ = [dert.blob_id_min for [blob, dert] in blobdert_]
    id_map__ = [id_map_[i:i + width] for i in range(0, len(id_map_), width)]
    id_map_np__ = np.array(id_map__)

    ## end of 1st iteration ##

    while f_ite:

        print("Running iteration " + str(ite_count))

        id_map_np_prior___ = id_map_np__

        ## consecutive nth iteration ##

        # convert blobdert_ to 2D array
        blobdert__ = [blobdert_[i:i + width] for i in range(0, len(blobdert_), width)]

        # get each dert rims (8 rims)
        blobdert_rim_ = get_rim_dert(blobdert__, height, width)

        # get updated id in each dert
        blobdert_ = pool.starmap(check_rim, zip(blobdert_, blobdert_rim_))

        # get map of min id
        id_map_ = [dert.blob_id_min for [blob, dert] in blobdert_]
        id_map__ = [id_map_[i:i + width] for i in range(0, len(id_map_), width)]
        id_map_np__ = np.array(id_map__)

        ## end of consecutive nth iteration ##

        # check whether there is any change in id
        dif = id_map_np__ - id_map_np_prior___

        # if there is no change in ids, stop the iteration
        if (np.sum(dif) == 0):
            f_ite = 0

        # save image
        cv2.imwrite("./images/parallel/id_ite_" + str(ite_count) + ".png", ((np.fliplr(np.rot90(np.array(id_map_np__), 3)) * 255) / (width * height)).astype('uint8'))

        # increase interation count
        ite_count += 1

    print("total iteration = " + str(ite_count))

    # close pool of threads
    pool.close()
    pool.join()


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()

    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    frame_blobs_parallel_Chee(dert__)

    end_time = time() - start_time
    print("time elapsed = " + str(end_time))