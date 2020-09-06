'''
    2D version of first-level core algorithm will have frame_blobs, intra_blob (recursive search within blobs), and comp_P.
    frame_blobs() performs three types of operations:
    - comp_pixel:
    Comparison between diagonal pixels in 2x2 kernels of image forms derts: tuples of pixel + derivatives per kernel.
    The output is dert__: 2D pixel-mapped array of pixel-mapped derts.
    - dert_blobs:
    Image dert__ is segmented into blobs: contiguous areas of same-sign G (deviation of gradient per kernel).
    Each blob is parameterized with summed derivatives of its constituent derts.
    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.
'''

import sys
import numpy as np

from collections import deque
from frame_blobs_defs import CBlob, FrameOfBlobs
from frame_blobs_cwrapper import cwrapped_derts2blobs
from frame_blobs_imaging import visualize_blobs
from utils import minmax

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback
UNFILLED = -1
EXCLUDED_ID = -2

def accum_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.G += dert__[1][y, x]
    blob.Dy += dert__[2][y, x]
    blob.Dx += dert__[3][y, x]


def comp_pixel(image):  # 2x2 pixel cross-correlation within image, as in edge detection operators
    # see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # same as decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # same as decomposition of two diagonal differences into Gx

    G__ = np.hypot(Gy__, Gx__) - ave  # central gradient per kernel, between its four vertex pixels

    return (topleft__, G__, Gy__, Gx__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed dert__ = (p__, g__, dy__, dx__) for readability in functions below


def derts2blobs(dert__, verbose=False):

    blob_, idmap, adj_pairs = flood_fill(dert__,
                                         sign__=dert__[1] > 0,
                                         verbose=verbose)
    I = 0; G = 0; Dy = 0; Dx = 0
    for blob in blob_:
        I += blob.I
        G += blob.G
        Dy += blob.Dy
        Dx += blob.Dx

    frame = FrameOfBlobs(I=I, G=G, Dy=Dy, Dx=Dx, blob_=blob_, dert__=dert__)

    return frame, idmap, adj_pairs

def flood_fill(dert__, sign__, verbose=False,
               exluded_derts=None,
               blob_cls=CBlob,
               accum_func=accum_blob_Dert):

    height, width = dert__[0].shape
    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if exluded_derts is not None:
        idmap[[*zip(*exluded_derts)]] = EXCLUDED_ID

    if verbose:
        step = 100 / height / width     # progress % percent per pixel
        progress = 0.0
        print(f"\rClustering... {round(progress)} %", end="")
        sys.stdout.flush()

    blob_ = []
    adj_pairs = set()
    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts
                # initialize new blob
                blob = blob_cls(sign=sign__[y, x], root_dert__=dert__)
                blob_.append(blob)
                idmap[y, x] = blob.id

                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    blob.dert_coord_.add((y1, x1))  # add dert coordinate to blob
                    accum_func(blob, dert__, y1, x1)
                    blob.S += 1

                    # determine neighbors' coordinates, 4 for -, 8 for +
                    if blob.sign:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]

                    # search through neighboring derts
                    for y2, x2 in adj_dert_coords:
                        # check if image boundary is reached
                        if (y2 < 0 or y2 >= height or
                            x2 < 0 or x2 >= width or
                            idmap[y2, x2] == EXCLUDED_ID):
                            blob.fopen = True
                        # check if filled
                        elif idmap[y2, x2] == UNFILLED:
                            # check if same-signed
                            if blob.sign == sign__[y2, x2]:
                                idmap[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts.append((y2, x2))
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))     # blob.id always bigger

                # terminate blob
                y_coords, x_coords = zip(*blob.dert_coord_)
                y0, yn = minmax(y_coords)
                x0, xn = minmax(x_coords)
                blob.box = (
                    y0, yn + 1,  # y0, yn
                    x0, xn + 1,  # x0, xn
                )
                blob.adj_blobs = [[], 0, 0]

                if verbose:
                    progress += blob.S * step
                    print(f"\rClustering... {round(progress)} %", end="")
                    sys.stdout.flush()
    if verbose:
        print("")

    return blob_, idmap, adj_pairs


def assign_adjacents(adj_pairs, blob_cls=CBlob):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in adj_pairs:
        assert blob_id1 < blob_id2
        blob1 = blob_cls.get_instance(blob_id1)
        blob2 = blob_cls.get_instance(blob_id2)

        y01, yn1, x01, xn1 = blob1.box
        y02, yn2, x02, xn2 = blob2.box

        if blob1.fopen and blob2.fopen:
            pose1 = pose2 = 2
        elif y01 < y02 and x01 < x02 and yn1 > yn2 and xn1 > xn2:
            pose1, pose2 = 0, 1  # 0: internal, 1: external
        elif y01 > y02 and x01 > x02 and yn1 < yn2 and xn1 < xn2:
            pose1, pose2 = 1, 0  # 1: external, 0: internal
        else:
            raise ValueError("something is wrong with pose")

        # bilateral assignments
        blob1.adj_blobs[0].append((blob2, pose2))
        blob2.adj_blobs[0].append((blob1, pose1))
        blob1.adj_blobs[1] += blob2.S
        blob2.adj_blobs[1] += blob1.S
        blob1.adj_blobs[2] += blob2.G
        blob2.adj_blobs[2] += blob1.G


if __name__ == "__main__":
    # Imports
    import argparse
    from time import time
    from utils import imread

    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=1)
    argument_parser.add_argument('-c', '--clib', help='use C shared library', type=int, default=0)
    args = argument_parser.parse_args()
    image = imread(args.image)
    # verbose = args.verbose
    # intra = args.intra
    # render = args.render

    # frame-blobs start here
    start_time = time()
    dert__ = comp_pixel(image)
    if args.clib:
        frame, idmap, adj_pairs = cwrapped_derts2blobs(dert__)
    else:
        frame, idmap, adj_pairs = derts2blobs(dert__, verbose=args.verbose)
    assign_adjacents(adj_pairs)
    print(f"{len(frame.blob_)} blobs formed in {time() - start_time} seconds")

    if args.render:  # will be replaced with interactive adjacent blobs display
        visualize_blobs(idmap, CBlob)

    # TODO: intra_blob initialization

    # # Test if the two versions give identical results
    # from itertools import zip_longest
    # frame, idmap, adj_pairs = cwrapped_derts2blobs(dert__)
    # frame1, idmap1, adj_pairs1 = derts2blobs(dert__, verbose=args.verbose)
    # did = 0
    # dI = 0
    # dG = 0
    # dDy = 0
    # dDx = 0
    # dbox = 0
    # dfopen = 0
    # dsign = 0
    # for blob, blob1 in zip_longest(frame.blob_, frame1.blob_):
    #     did += abs(blob.id - blob1.id)
    #     dI += abs(blob.I - blob1.I)
    #     dG += abs(blob.G - blob1.G)
    #     dDy += abs(blob.Dy - blob1.Dy)
    #     dDx += abs(blob.Dx - blob1.Dx)
    #     dfopen += abs(blob.fopen - blob1.fopen)
    #     dsign += abs(int(blob.sign) - int(blob1.sign))
    #     dbox += abs(blob.box[0] - blob1.box[0])
    #     dbox += abs(blob.box[1] - blob1.box[1])
    #     dbox += abs(blob.box[2] - blob1.box[2])
    #     dbox += abs(blob.box[3] - blob1.box[3])
    # print(np.array([did, dI, dG, dDy, dDx, dbox, dfopen, dsign]) / len(frame.blob_))jacent blobs display

        visualize_blobs(idmap, CBlob)