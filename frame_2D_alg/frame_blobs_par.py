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

    while blob__:  # flood-fill blobs by single layer of open_derts per cycle, stream compaction in next cycle
        cycle(blob__, frame_dert__)
    '''
    also check for:
    while yn-y0 < Y or xn-x0 < X: 
        generic box < frame: iterate blob and box extension cycle by rim per open dert
        check frame_dert__ to assign lowest blob ID in each dert, 
        remove all other blobs from the stream
    '''

def cycle(blob__, frame_dert__):  # blob extension cycle, initial blobs are open derts

    blob__ = remove_overlapping blobs(blob__, frame_dert__)
    # check ids in all frame derts, remove blobs in blob__ that have >lowest id in any frame dert
    # not sure how to do it in parallel?

    for i, blob in enumerate(blob__):
        open_AND = 0  # counter of AND(open_dert.sign, unfilled_rim_dert.sign)
        new_open_derts = []

        for dert in blob.open_derts:  # open_derts: list of derts in dert__ with unfilled derts in their 3x3 rim
            j = 0
            while j < 7:
                _y, _x = compute_rim(dert.y, dert.x, j)  # compute rim _y _x from dert y x, clock-wise from top-left
                j += 1
                if not (_y in blob.dert__ and _x in blob.dert__):
                    _dert = frame_dert__[_y][_x]  # load unfilled rim dert from frame.dert__
                    if dert.sign and _dert.sign:
                        open_AND += 1
                        new_open_derts.append(_dert)
                        # add: blob params += _dert params
                        frame_dert__[_y][_x].append(blob.ID)
                        # multiple blob IDs maybe assigned to each frame_dert, with the lowest one selected in next cycle
        if open_AND==0:
            del blob[i]  # blob terminates remove from next extension cycle


def compute_rim(y, x, j):
    # topleft:
    if j == 0: _y = y - 1; _x = x - 1
    # top:
    elif j==1: _y = y - 1; _x = x
    # topright:
    elif j==2: _y = y - 1; _x = x + 1
    # right:
    elif j==3: _y = y; _x = x + 1
    # bottomright:
    elif j==4: _y = y + 1; _x = x + 1
    # bottom:
    elif j==5: _y = y + 1; _x = x
    # bottomleft:
    elif j==6: _y = y + 1; _x = x - 1
    # left:
    else:      _y = y; _x = x - 1

    return _y, _x

'''
Chee's implementation of blob-parallel frame_blobs,
with a bunch of my edits, unfinished:
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
    blob = list
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
    dert = object


def initialize_blobs(dert_input, y, x):

    # dert instance from their class
    dert = CDert(i=dert_input[0], g=dert_input[1] - ave, dy=dert_input[2], dx=dert_input[3],
                 x_coord=x, y_coord=y, sign=dert_input[1] - ave > 0, fopen=1)

    # blob instance from their class
    blob = CBlob(I=dert_input[0], G=dert_input[1] - ave, Dy=dert_input[2], Dx=dert_input[3],
                 dert_coord_=[[y, x]], sign=dert_input[1] - ave > 0, dert=dert)

    # update blob params into dert
    dert.blob = blob
    dert.blob_ids.append(blob.id)
    dert.blob_id_min = blob.id

    return [blob, dert]


def check_rims(blob, blob_rims):
    '''
    check connectivity of blob's rim derts and update each dert's ids
    I use "rim" as in kernel only, blob has open_derts instead.
    So this should be "check rims of open_derts?
    '''
    dert = blob.dert  # get dert from blob

    if blob.sign:  # for + sign, check 8 ortho+diag rims
        for blob_rim in blob_rims:
            if blob_rim:  # not empty rim
                _dert = blob_rim.dert

                # check same sign
                if dert.sign == _dert.sign:
                    dert.blob_ids.append(blob_rim.id)  # add same sign rim blob's id
                    dert.blob_ids += _dert.blob_ids  # chain the rim dert' blob' rim dert's blobs ids
                    dert.blob_ids = list(set(dert.blob_ids))  # remove duplicated ids

    else:  # for - sign, check 4 ortho rims
        for i, blob_rim in enumerate(blob_rims):
            if blob_rim and i % 2:  # not empty rim and ortho rim
                _dert = blob_rim.dert

                # check same sign
                if dert.sign == _dert.sign:
                    dert.blob_ids.append(blob_rim.id)  # add same sign rim blob's id
                    dert.blob_ids += _dert.blob_ids  # chain the rim dert' blob' rim dert's blobs ids
                    dert.blob_ids = list(set(dert.blob_ids))  # remove duplicated ids

    # min of ids
    dert.blob_id_min = min(dert.blob_ids)
    return [blob, dert]


# there could be a better way to replace this function with parallel process, need to think about it

def get_rim_derts(dert_, height, width):

    # convert frame dert_ to 2D array
    # this should be original frame_dert__, already 2D array?
    # blob__ is a separate array, blob growth and pruning should not affect frame_dert__?

    dert__ = [dert_[i:i + width] for i in range(0, len(dert_), width)]
    blob_rims_ = []

    for y in range(height):
        for x in range(width):
        # sorry, you are getting rims for all derts in blob box?
        # it should only be for open derts, added in the last cycle, check line 29 above

            if y - 1 >= 0 and x - 1 >= 0:
                topleft = dert__[y - 1][x - 1]
            else: topleft = []
            if y - 1 >= 0:
                top = dert__[y - 1][x]
            else: top = []
            if y - 1 >= 0 and x + 1 <= width - 1:
                topright = dert__[y - 1][x + 1]
            else: topright = []
            if x + 1 <= width - 1:
                right = dert__[y][x + 1]
            else: right = []
            if y + 1 <= height - 1 and x + 1 <= width - 1:
                botright = dert__[y + 1][x + 1]
            else: botright = []
            if y + 1 <= height - 1:
                bottom = dert__[y + 1][x]
            else: bottom = []
            if y + 1 <= height - 1 and x - 1 >= 0:
                botleft = dert__[y + 1][x - 1]
            else: botleft = []
            if x - 1 >= 0:
                left = dert__[y][x - 1]
            else: left = []

            blob_rims_.append([topleft, top, topright, right, botright, bottom, botleft, left])
    return blob_rims_


def cycle_Chee(pool, blob_, height, width):
    '''
    cycle includes getting updated rim blobs and update the rims
    this should be
    '''
    # get each dert rims (8 rims)
    blob_rims_ = get_rim_derts(blob_, height, width)
    # (parallel process) get updated id in each dert
    blob_, dert_ = zip(*pool.starmap(check_rims, zip(blob_, blob_rims_)))

    # get map of min id
    id_map_ = [dert.blob_id_min for dert in dert_]
    id_map__ = [id_map_[i:i + width] for i in range(0, len(id_map_), width)]
    id_map_np__ = np.array(id_map__)

    return blob_, id_map_np__


def frame_blobs_parallel(dert__):
    '''
    Draft of parallel blobs forming process, consists
    '''

    pool = ThreadPool(mp.cpu_count())  # initialize pool of threads

    height, width = dert__[0].shape  # height and width of image

    # generate all x and y coordinates
    dert_coord = [[y, x] for x in range(width) for y in range(height)]
    y_, x_ = zip(*[[y, x] for y, x in dert_coord])

    # get each non class instance dert from coordinates
    dert_ = [dert__[:, y, x] for y, x in dert_coord]

    # (parallel process) generate instance of derts and blobs from their class
    blob_, dert_ = zip(*pool.starmap(initialize_blobs, zip(dert_, y_, x_)))

    cycle_count = 0
    id_map_np_prior___ = np.zeros((height, width))  # prior id_map, to check when to stop iteration
    fcontinue = 1  # 0 = stop, 1 = continue

    ## 1st cycle ##
    blob_, id_map_np__ = cycle_Chee(pool, blob_, height, width)

    # save output image
    cv2.imwrite("./images/parallel/id_cycle_0.png", ((np.fliplr(np.rot90(np.array(id_map_np__), 3)) * 255) / (width * height)).astype('uint8'))
    while fcontinue:

        print("Running cycle " + str(cycle_count + 1))
        id_map_np_prior___ = id_map_np__  # update prior id map

        ## consecutive cycles ##
        blob_, id_map_np__ = cycle_Chee(pool, blob_, height, width)
        dif_id = id_map_np__ - id_map_np_prior___

        # if no change in ids, stop the iteration
        if (np.sum(dif_id) == 0):
            fcontinue = 0

        # save image
        cv2.imwrite("./images/parallel/id_cycle_" + str(cycle_count + 1) + ".png", ((np.fliplr(np.rot90(np.array(id_map_np__), 3)) * 255) / (width * height)).astype('uint8'))
        cycle_count += 1

    print("total cycle= " + str(cycle_count))

    # close pool of threads
    pool.close()
    pool.join()

    return np.fliplr(np.rot90(np.array(id_map_np__)))


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()

    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    id_map__ = frame_blobs_parallel(dert__)

    end_time = time() - start_time
    print("time elapsed = " + str(end_time))