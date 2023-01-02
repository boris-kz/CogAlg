from class_cluster import ClusterStructure, NoneType
from utils import (
    pairwise,
    imread, )
import multiprocessing as mp

from time import time
import numpy as np
from utils import *
from multiprocessing.pool import ThreadPool
from matplotlib import pyplot as plt
import cv2
import numpy.ma as ma

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
    blob_id = int
    fopen = bool
    dert_rims = list
    blob_min_id = int

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
    rim_ids = list
    dert_open_ = list
    dert_ = list
    min_id = int
'''
Draft of blob-parallel version of frame_blobs, pseudo code:

def frame_blobs_parallel(frame_dert__):  # grow blob.dert__ in all blobs in parallel:

    blob__ = [Cblob() for dert in frame_dert__]  # initialize one blob per dert
    for i, (blob, dert) in enumerate( zip(blob__, frame_dert__)):
        blob__[i] = [dert], blob  # add open_derts and empty connected_blobs to each blob

    while blob__:  # flood-fill blobs by single layer of open_derts per cycle, stream compaction in next cycle
        cycle(blob__, frame_dert__)

def cycle(blob__, frame_dert__):  # parallel blob extension cycle, initial blobs are open derts

    blob__ = remove_overlapping blobs(blob__, frame_dert__)
    # check ids in all frame derts, remove blob__ blobs with >lowest id in any frame dert
    # or by blob.G: remove blobs with < highest-G in any frame dert?
    # each frame dert is mapped to a node?

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

        if open_AND==0:  # add: or yn-y0 < Y or xn-x0 < X: generic box < frame?
            del blob[i]  # blob is terminated, remove from next extension cycle
'''
# Chee's implementation:

def comp_pixel(image):  # 2x2 pixel cross-correlation within image, as in edge detection operators

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # same as decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # same as decomposition of two diagonal differences into Gx

    G__ = np.hypot(Gy__, Gx__)  # central gradient per kernel, between its four vertex pixels
    # why ma?
    return ma.stack((topleft__, G__, Gy__, Gx__))  # tuple of 2D arrays per param of dert (derivatives' tuple)


def generate_blobs(dert_input, y, x):
    # generate dert__ and blob at each dert
    # dert class instance
    dert = CDert(i=dert_input[0], g=dert_input[1] - ave, dy=dert_input[2], dx=dert_input[3],
                 x_coord=x, y_coord=y, sign=dert_input[1] - ave > 0, fopen=1)
    # blob class instance
    blob = CBlob(I=dert_input[0], G=dert_input[1] - ave, Dy=dert_input[2], Dx=dert_input[3],
                 dert_coord_=[[y, x]], sign=dert_input[1] - ave > 0, dert_open_=[dert])

    dert.blob_id = blob.id  # each dert contains id of the blob it belongs to.

    return blob, dert


def check_open_rims(blob):
    # check connectivity of blob's rim derts and update each dert ids
    new_dert_open_ = []  # for next cycle

    while blob.dert_open_:
        dert = blob.dert_open_.pop()  # retrieve open dert
        if dert.sign:  # for + sign, check 8 ortho+diag rims
            for rim_dert in dert.rim_derts:
                if rim_dert:  # not empty
                    if dert.sign == rim_dert.sign:
                        blob.rim_dert_ids.append(rim_dert.blob_id)  # add id of rim dert's blob  into blob
                        blob.rim_dert_ids = list(set(blob.rim_dert_ids))  # remove duplicated ids
                        new_dert_open_.append(rim_dert)  # add rim dert into open dert list

        else:  # for - sign, check 4 ortho rims
            for i, rim_dert in enumerate(dert.rim_derts):
                if rim_dert and i % 2:  # not empty rim and ortho rim
                    # check same sign
                    if dert.sign == rim_dert.sign:
                        blob.rim_dert_ids.append(rim_dert.blob_id)  # add id of rim dert's blob  into blob
                        blob.rim_dert_ids = list(set(blob.rim_dert_ids))  # remove duplicated ids
                        new_dert_open_.append(rim_dert)  # add rim dert into open dert list

        blob.dert_.append(dert)  # add processed dert into list of closed dert
    blob.dert_open_ = new_dert_open_

    # remove duplicated open derts
    dert_open_unique = []
    for dert in blob.dert_open_:
        if dert not in dert_open_unique:
            dert_open_unique.append(dert)
    blob.dert_open_ = dert_open_unique

    # update closed dert id
    for dert in blob.dert_:
        dert.blob_id = blob.id

    # check min of ids and remove blob if blob's id > min id
    if blob.rim_ids:
        blob.min_id = min(blob.rim_ids)
        if blob.id > blob.min_id:
            blob = []

    return blob

# there could be a better way to replace this function with parallel process ,need to think about it
def get_rim_dert(dert_, height, width):
    '''
    generate dert's rims
    '''
    # convert dert_ to 2D array dert__
    dert__ = [dert_[i:i + width] for i in range(0, len(dert_), width)]

    n = 0
    for y in range(height):
        for x in range(width):
            # topleft
            if y - 1 >= 0 and x - 1 >= 0:
                dert_topleft = dert__[y - 1][x - 1]
            else:
                dert_topleft = []
            # top
            if y - 1 >= 0:
                dert_top = dert__[y - 1][x]
            else:
                dert_top = []
            # topright
            if y - 1 >= 0 and x + 1 <= width - 1:
                dert_topright = dert__[y - 1][x + 1]
            else:
                dert_topright = []
            # right
            if x + 1 <= width - 1:
                dert_right = dert__[y][x + 1]
            else:
                dert_right = []
            # botright
            if y + 1 <= height - 1 and x + 1 <= width - 1:
                dert_botright = dert__[y + 1][x + 1]
            else:
                dert_botright = []
            # bot
            if y + 1 <= height - 1:
                dert_bot = dert__[y + 1][x]
            else:
                dert_bot = []
            # botleft
            if y + 1 <= height - 1 and x - 1 >= 0:
                dert_botleft = dert__[y + 1][x - 1]
            else:
                dert_botleft = []
            # left
            if x - 1 >= 0:
                dert_left = dert__[y][x - 1]
            else:
                dert_left = []

            dert_rims = [dert_topleft, dert_top, dert_topright, dert_right,
                         dert_botright, dert_bot, dert_botleft, dert_left]

            if dert_[n]:
                dert_[n].dert_rims = dert_rims
            n += 1

    return dert_


def get_id_map(blob_, height, width):
    '''
    generate map of derts' id
    '''

    id_map__ = np.zeros((height, width)) - 1

    for blob in blob_:
        for dert in blob.dert_:
            id_map__[dert.y_coord, dert.x_coord] = dert.blob_id
        for dert in blob.dert_open_:
            id_map__[dert.y_coord, dert.x_coord] = dert.blob_id

    return id_map__


def accumulate_blob_(blob_):
    '''
    accumulate dert's param into blob at the end of the cycle
    '''
    for blob in blob_:
        y = []
        x = []
        for dert in blob.dert_:
            # accumulate params
            blob.I += dert.i
            blob.G += dert.g
            blob.Dy += dert.dy
            blob.Dx += dert.dx
            blob.S += 1
            y.append(dert.y_coord)
            x.append(dert.y_coord)

        blob.box = (min(y), max(y), min(x), max(x))


def extension_cycle(pool, blob_, height, width):
    '''
    1 cycle operation, including getting the updated rim blobs and update the rims
    '''

    blob_ = pool.starmap(check_open_rims, zip(blob_))  # check and add rim derts to blob
    blob_ = [blob for blob in blob_ if blob]  # remove empty blob
    id_map__ = get_id_map(blob_, height, width)  # get map of dert's id

    return blob_, id_map__


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
    blob_, dert_ = zip(*pool.starmap(generate_blobs, zip(dert_, y_, x_)))

    get_rim_dert(dert_, height, width)  # get rim per dert

    cycle_count = 0
    f_cycle = 1  # flag to continue cycle, 0 = stop, 1 = continue

    ## 1st cycle ##
    blob_, id_map__ = extension_cycle(pool, blob_, height, width)
    id_map_prior__ = id_map__  # prior id_map, to check when to stop iteration

    # save output image
    cv2.imwrite("./images/parallel/id_cycle_0.png", (((id_map__) * 255) / (width * height)).astype('uint8'))

    while f_cycle:

        print("Running cycle " + str(cycle_count + 1))
        ## consecutive cycles ##
        blob_, id_map__ = extension_cycle(pool, blob_, height, width)
        # check if ids changed:
        dif = id_map__ - id_map_prior__
        # update id map
        id_map_prior__ = id_map__
        # if no change in ids, stop the iteration:
        if (np.sum(dif) == 0):
            f_cycle = 0
        # save image
        cv2.imwrite("./images/parallel/id_cycle_" + str(cycle_count + 1) + ".png", (((id_map__) * 255) / (width * height)).astype('uint8'))

        cycle_count += 1

    accumulate_blob_(blob_)  # accumulate dert param into their blob

    print("total cycle= " + str(cycle_count))

    # close pool of threads
    pool.close()
    pool.join()

    return blob_, id_map__


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_head.jpg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()

    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    blob_, id_map__ = frame_blobs_parallel(dert__)

    print('total blob formed = ' + str(len(blob_)))

    end_time = time() - start_time
    print("time elapsed = " + str(end_time))