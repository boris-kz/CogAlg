'''
    2D version of first-level core algorithm includes frame_blobs, intra_blob (recursive search within blobs), and blob2_P_blob.

    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity)
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object
    (predictive value is physical density, hardness, inertia that represent resistance to change in positional parameters)

    Comparison range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions,
    orientation. These params are essential because value of prediction = precision of what * precision of where.
    Clustering here is nearest-neighbor only, to avoid overlap among blobs.
    Main functions:

    - comp_pixel:
    Comparison between diagonal pixels in 2x2 kernels of image forms derts: tuples of pixel + derivatives per kernel.
    The output is dert__: 2D pixel-mapped array of pixel-mapped derts.

    - derts2blobs:
    Segmentation of image dert__ into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
    Each blob is parameterized with summed params of constituent derts, derived by pixel cross-comparison (cross-correlation).
    These params represent predictive value per pixel, so they are also predictive on a blob level,
    thus should be cross-compared between blobs on the next level of search.

    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.
'''

import sys
import numpy as np

from collections import deque
# from frame_blobs_wrapper import wrapped_flood_fill
from frame_blobs_imaging import visualize_blobs
from utils import minmax
from collections import namedtuple
from class_cluster import ClusterStructure, NoneType

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback
UNFILLED = -1
EXCLUDED_ID = -2

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, G, M, blob_, dert__')

class CBlob(ClusterStructure):
    # Dert params
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # blob params
    A = int  # blob area
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool

class CDeepBlob(ClusterStructure):
    # Dert params
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    # blob params
    A = int  # blob area
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    fia = bool  # flag: input is from comp angle
    fca = bool  # flag: current fork is comp angle
    rdn = float
    rng = int
    Ls = int  # for visibility and next-fork rdn
    sub_layers = list
    a_depth = int
    prior_forks = list
    stack_ = list


def comp_pixel(image):  # 2x2 pixel cross-correlation within image, as in edge detection operators
    # see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # same as decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # same as decomposition of two diagonal differences into Gx

    # according to Stephan, this is less accurate than rotating dert__ to convert diagonals into orthogonals.
    # makes sense because any summation is a reduction in accuracy, but then we need to rotate image 45 degree.

    G__ = (np.hypot(Gy__, Gx__) - ave).astype('int')  # deviation of central gradient per kernel, between four vertex pixels
    M__ = ave - (abs(topleft__ - bottomright__) + abs(topright__ - bottomleft__))  # inverse deviation of SAD: variation in kernel

    return (topleft__, Gy__, Gx__, G__, M__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed dert__ = (p__, dy__, dx__, g__, m__) for readability in functions below


def derts2blobs(dert__, verbose=False, render=False, use_c=False):

    if verbose:
        start_time = time()

    if use_c:
        dert__ = dert__[0], np.empty(0), np.empty(0), *dert__[1:], np.empty(0)
        frame, idmap, adj_pairs = wrapped_flood_fill(dert__)
    else:
        blob_, idmap, adj_pairs = flood_fill(dert__, sign__=dert__[3] > 0,  # sign of deviation of gradient
                                             verbose=verbose)
        I = Dy = Dx = G = M = 0
        for blob in blob_:
            I += blob.I
            Dy += blob.Dy
            Dx += blob.Dx
            G += blob.G
            M += blob.M
        frame = FrameOfBlobs(I=I, Dy=Dy, Dx=Dx, G=G, M=M, blob_=blob_, dert__=dert__)

    assign_adjacents(adj_pairs)

    if verbose:
        print(f"{len(frame.blob_)} blobs formed in {time() - start_time} seconds")
    if render:
        visualize_blobs(idmap, frame.blob_)

    return frame


def accum_blob_Dert(blob, dert__, y, x):

    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]


def flood_fill(dert__, sign__, verbose=False, mask=None, blob_cls=CBlob, accum_func=accum_blob_Dert):

    if mask is None: # non intra dert
        height, width = dert__[0].shape
    else: # intra dert
        height, width = mask.shape

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask is not None:
        idmap[mask] = EXCLUDED_ID

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
                y0, yn = y, y
                x0, xn = x, x

                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()
                    # add dert to blob
                    accum_func(blob, dert__, y1, x1)
                    blob.A += 1
                    if y1 < y0:
                        y0 = y1
                    elif y1 > yn:
                        yn = y1
                    if x1 < x0:
                        x0 = x1
                    elif x1 > xn:
                        xn = x1
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
                yn += 1
                xn += 1
                blob.box = y0, yn, x0, xn
                blob.mask = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[], 0, 0, 0, 0]

                if verbose:
                    progress += blob.A * step
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
        blob1.adj_blobs[1] += blob2.A
        blob2.adj_blobs[1] += blob1.A
        blob1.adj_blobs[2] += blob2.G
        blob2.adj_blobs[2] += blob1.G
        blob1.adj_blobs[3] += blob2.M
        blob2.adj_blobs[3] += blob1.M
        if hasattr(blob1,'Ma'): # add Ma to deep blob only
            blob1.adj_blobs[4] += blob2.Ma
            blob2.adj_blobs[4] += blob1.Ma


def print_deep_blob_forking(deep_layer):

    def check_deep_blob(deep_layer,i):
        for deep_blob_layer in deep_layer:
            if isinstance(deep_blob_layer,list):
                check_deep_blob(deep_blob_layer,i)
            else:
                print('blob num = '+str(i)+', forking = '+'->'.join(deep_blob_layer.prior_forks))

    for i, deep_layer in enumerate(deep_layers):
        if len(deep_layer)>0:
            check_deep_blob(deep_layer,i)

if __name__ == "__main__":
    # Imports
    import argparse
    from time import time
    from utils import imread

    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//toucan.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=1)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=0)
    argument_parser.add_argument('-c', '--clib', help='use C shared library', type=int, default=0)
    args = argument_parser.parse_args()
    image = imread(args.image)
    # verbose = args.verbose
    # intra = args.intra
    # render = args.render

    start_time = time()
    dert__ = comp_pixel(image)
    frame = derts2blobs(dert__, verbose=args.verbose, render=args.render, use_c=args.clib)

    if args.intra:  # call to intra_blob, omit for testing frame_blobs only:

        if args.verbose: print("\rRunning intra_blob...")
        from intra_blob import intra_blob, aveB

        deep_frame = frame, frame  # 1st frame initializes summed representation of hierarchy, 2nd is individual top layer
        deep_blob_i_ = []  # index of a blob with deep layers
        deep_layers = [[]] * len(frame.blob_)  # for visibility only
        empty = np.zeros_like(frame.dert__[0])
        deep_root_dert__ = (  # update root dert__
            frame.dert__[0],  # i
            frame.dert__[1],  # dy
            frame.dert__[2],  # dx
            frame.dert__[3],  # g
            frame.dert__[4],  # m
            )

        for i, blob in enumerate(frame.blob_):  # print('Processing blob number ' + str(bcount))
            '''
            Blob G: -|+ predictive value, positive value of -G blobs is lent to the value of their adjacent +G blobs. 
            +G "edge" blobs are low-match, valuable only as contrast: to the extent that their negative value cancels 
            positive value of adjacent -G "flat" blobs.
            '''
            G = blob.G
            M = blob.M
            adj_G = blob.adj_blobs[2]
            borrow_G = min(abs(G), abs(adj_G) / 2)
            '''
            int_G / 2 + ext_G / 2, because both borrow or lend bilaterally, 
            same as pri_M and next_M in line patterns but direction here is radial: inside-out
            borrow_G = min, ~ comp(G,_G): only value present in both parties can be borrowed from one to another
            Add borrow_G -= inductive leaking across external blob?
            '''
            blob = CDeepBlob(I=blob.I, Dy=blob.Dy, Dx=blob.Dx, G=blob.G, M=blob.M, A=blob.A, box=blob.box, sign=blob.sign,
                             mask=blob.mask, root_dert__=deep_root_dert__, adj_blobs=blob.adj_blobs, fopen=blob.fopen)

            blob_height = blob.box[1] - blob.box[0]
            blob_width = blob.box[3] - blob.box[2]

            if blob.sign:
                # +G on first fork
                if min(G, borrow_G) > aveB and blob_height > 3 and blob_width  > 3:  # min blob dimensions
                    blob.rdn = 1; blob.fca = 1 # +G blob' dert' comp_a
                    blob.prior_forks.append('g')
                    deep_layers[i] = intra_blob(blob, render=args.render, verbose=args.verbose)

            # +M on first fork
            elif M - borrow_G > aveB and blob_height > 3 and blob_width  > 3:  # min blob dimensions
                blob.rdn = 1; blob.rng = 1; blob.fia = 0
                blob.prior_forks.append('g')
                deep_layers[i] = intra_blob(blob, render=args.render, verbose=args.verbose)  # -G blob' dert__' comp_r in 3x3 kernels

            if deep_layers[i]:  # if there are deeper layers
                deep_blob_i_.append(i)  # indices of blobs with deep layers


        if args.verbose:
            print_deep_blob_forking(deep_layers)
            print("\rFinished intra_blob")

    end_time = time() - start_time

    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)