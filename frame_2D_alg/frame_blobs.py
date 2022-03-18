'''
    2D version of first-level core algorithm includes frame_blobs, intra_blob (recursive search within blobs), and blob2_P_blob.
    -
    Blob is 2D pattern: connectivity cluster defined by the sign of gradient deviation. Gradient represents 2D variation
    per pixel. It is used as inverse measure of partial match (predictive value) because direct match (min intensity)
    is not meaningful in vision. Intensity of reflected light doesn't correlate with predictive value of observed object
    (predictive value is physical density, hardness, inertia that represent resistance to change in positional parameters)
    -
    Comparison range is fixed for each layer of search, to enable encoding of input pose parameters: coordinates, dimensions,
    orientation. These params are essential because value of prediction = precision of what * precision of where.
    Clustering here is nearest-neighbor only, same as image segmentation, to avoid overlap among blobs.
    -
    Main functions:
    - comp_pixel:
    Comparison between diagonal pixels in 2x2 kernels of image forms derts: tuples of pixel + derivatives per kernel.
    The output is dert__: 2D pixel-mapped array of pixel-mapped derts.
    - frame_blobs_root:
    Segmentation of image dert__ into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
    Each blob is parameterized with summed params of constituent derts, derived by pixel cross-comparison (cross-correlation).
    These params represent predictive value per pixel, so they are also predictive on a blob level,
    thus should be cross-compared between blobs on the next level of search.
    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.
    Frame_blobs is a root function for all deeper processing in 2D alg.
    -
    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/blob_params.drawio
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs_intra_blob.drawio
'''

import sys
import numpy as np
from collections import deque, namedtuple
from itertools import zip_longest
from draw_frame_blobs import visualize_blobs
from class_cluster import ClusterStructure
# from frame_blobs_wrapper import wrapped_flood_fill, from utils import minmax, from time import time

# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED_ID = -2
# FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, M, blob_, dert__')

class CBlob(ClusterStructure):
    # comp_pixel:
    I = float
    Dy = float
    Dx = float
    G = float
    A = float  # blob area
    sign = bool
    # composite params:
    box = list  # x0, xn, y0, yn
    mask__ = object
    dert__ = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_angle:
    Sin_da0 = float
    Cos_da0 = float
    Sin_da1 = float
    Cos_da1 = float
    Ga = float
    # comp_dx:
    Mdx = float
    Ddx = float
    # derivation hierarchy:
    rsublayers = list  # list of layers across sub_blob derivation tree, deeper layers are nested with both forks
    asublayers = list  # separate for range and angle forks per blob
    prior_forks = list
    fBa = bool  # in root_blob: next fork is comp angle, else comp_r
    rdn = lambda: 1.0  # redundancy to higher blob layers, or combined?
    rng = int  # comp range, set before intra_comp
    # comp_slice:
    M = int  # summed PP.M, for both types of recursion?
    dir_blobs = list  # primarily vertically | laterally oriented edge blob segments, formed in segment_by_direction
    fsliced = bool
    fflip = bool  # x-y swap in comp_slice
    P__ = list
    derP_ = list  # redundant to P__ upconnects?
    PP_t = list  # or reuse P__?
    levels = list
    # frame_bblob:
    root_bblob = object
    sublevels = list  # input levels


def frame_blobs_root(image, intra=False, render=False, verbose=False, use_c=False):

    if verbose: start_time = time()
    dert__ = comp_pixel(image)

    blob_, idmap, adj_pairs = flood_fill(dert__, sign__= ave-dert__[3] > 0, verbose=verbose)  # dert__[3] is g, https://en.wikipedia.org/wiki/Flood_fill
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    I, Dy, Dx = 0, 0, 0
    for blob in blob_: I += blob.I; Dy += blob.Dy; Dx += blob.Dx
    frame = CBlob(I = I, Dy = Dy, Dx = Dx, dert__=dert__, prior_forks=["g"], rsublayers = [blob_])  # asublayers = []: no comp_a yet

    if verbose: print(f"{len(frame.rsublayers[0])} blobs formed in {time() - start_time} seconds")
    if render: visualize_blobs(idmap, frame.rsublayers[0])

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frame.rsublayers += intra_blob_root(frame, render, verbose, fBa=0)  # recursive eval cross-comp range| angle| slice per blob
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks:
    '''
    if use_c:  # old version, no longer updated:
        dert__ = dert__[0], np.empty(0), np.empty(0), *dert__[1:], np.empty(0)
        frame, idmap, adj_pairs = wrapped_flood_fill(dert__)
    '''
    return frame

def comp_pixel(image):  # 2x2 pixel cross-correlation within image, see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    d_upright__ = bottomleft__ - topright__
    d_upleft__ = bottomright__ - topleft__

    G__ = np.hypot(d_upright__, d_upleft__)  # 2x2 kernel gradient (variation), match = inverse deviation, for sign_ only
    rp__ = topleft__ + topright__ + bottomleft__ + bottomright__  # sum of 4 rim pixels -> mean, not summed in blob param

    return (topleft__, d_upleft__, d_upright__, G__, rp__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed dert__ = (i__, dy__, dx__, g__, ri__) for readability in deeper functions
'''
    old version:
    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # decomposition of two diagonal differences into Gx
'''

def flood_fill(dert__, sign__, verbose=False, mask__=None, fseg=False, prior_forks=[]):

    if mask__ is None: height, width = dert__[0].shape  # init dert__
    else:              height, width = mask__.shape  # intra dert__

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None:
        idmap[mask__] = EXCLUDED_ID
    if verbose:
        step = 100 / height / width  # progress % percent per pixel
        progress = 0.0; print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()
    blob_ = []
    adj_pairs = set()

    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts

                blob = CBlob(sign=sign__[y, x], root_dert__=dert__, prior_forks=['g'])
                if prior_forks: # update prior forks in deep blob
                    blob.prior_forks= prior_forks.copy()
                blob_.append(blob)
                idmap[y, x] = blob.id
                y0, yn = y, y
                x0, xn = x, x
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    if len(dert__) > 5: # comp_angle
                        blob.accumulate(I  = dert__[3][y1][x1],  # rp__,
                                        Dy = dert__[4][y1][x1],
                                        Dx = dert__[5][y1][x1],
                                        Sin_da0 = dert__[6][y1][x1],
                                        Cos_da0 = dert__[7][y1][x1],
                                        Sin_da1 = dert__[8][y1][x1],
                                        Cos_da1 = dert__[9][y1][x1])
                    else:  # comp_pixel or comp_range
                        blob.accumulate(I  = dert__[4][y1][x1],  # rp__,
                                        Dy = dert__[1][y1][x1],
                                        Dx = dert__[2][y1][x1])
                    blob.A += 1
                    if y1 < y0:   y0 = y1
                    elif y1 > yn: yn = y1
                    if x1 < x0:   x0 = x1
                    elif x1 > xn: xn = x1
                    # neighbors coordinates, 4 for -, 8 for +
                    if blob.sign or fseg:   # include diagonals
                        adj_dert_coords = [(y1 - 1, x1 - 1), (y1 - 1, x1),
                                           (y1 - 1, x1 + 1), (y1, x1 + 1),
                                           (y1 + 1, x1 + 1), (y1 + 1, x1),
                                           (y1 + 1, x1 - 1), (y1, x1 - 1)]
                    else:
                        adj_dert_coords = [(y1 - 1, x1), (y1, x1 + 1),
                                           (y1 + 1, x1), (y1, x1 - 1)]
                    # search neighboring derts:
                    for y2, x2 in adj_dert_coords:
                        # image boundary is reached:
                        if (y2 < 0 or y2 >= height or
                            x2 < 0 or x2 >= width or
                            idmap[y2, x2] == EXCLUDED_ID):
                            blob.fopen = True
                        # pixel is filled:
                        elif idmap[y2, x2] == UNFILLED:
                            # same-sign dert:
                            if blob.sign == sign__[y2, x2]:
                                idmap[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts.append((y2, x2))
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))  # blob.id always increases
                # terminate blob
                yn += 1; xn += 1
                blob.box = y0, yn, x0, xn
                blob.dert__ = tuple([param_dert__[y0:yn, x0:xn] for param_dert__ in blob.root_dert__])  # add None__ for m__?
                blob.mask__ = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = np.hypot(blob.Dy, blob.Dx)  # recompute G
                if len(dert__) > 5:  # recompute Ga
                    blob.Ga = (blob.Cos_da0 + 1) + (blob.Cos_da1 + 1)  # +1 for all positives
                if verbose:
                    progress += blob.A * step; print(f"\rClustering... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("")

    return blob_, idmap, adj_pairs


def assign_adjacents(adj_pairs):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in adj_pairs:
        blob1 = CBlob.get_instance(blob_id1)
        blob2 = CBlob.get_instance(blob_id2)

        y01, yn1, x01, xn1 = blob1.box
        y02, yn2, x02, xn2 = blob2.box

        if blob1.fopen and blob2.fopen:
            pose1 = pose2 = 2
        elif y01 < y02 and x01 < x02 and yn1 > yn2 and xn1 > xn2:
            pose1, pose2 = 0, 1  # 0: internal, 1: external
        elif y01 > y02 and x01 > x02 and yn1 < yn2 and xn1 < xn2:
            pose1, pose2 = 1, 0  # 1: external, 0: internal
        else:
            if blob2.A > blob1.A:
                pose1, pose2 = 0, 1  # 0: internal, 1: external
            else:
                pose1, pose2 = 1, 0  # 1: external, 0: internal
        # bilateral assignments
        '''
        if f_segment_by_direction:  # pose is not needed
            blob1.adj_blobs.append(blob2)
            blob2.adj_blobs.append(blob1)
        '''
        blob1.adj_blobs[0].append(blob2)
        blob1.adj_blobs[1].append(pose2)
        blob2.adj_blobs[0].append(blob1)
        blob2.adj_blobs[1].append(pose1)


if __name__ == "__main__":
    import argparse
    from time import time
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//toucan.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=0)
    argument_parser.add_argument('-c', '--clib', help='use C shared library', type=int, default=0)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=1)
    argument_parser.add_argument('-e', '--extra', help='run frame_recursive after frame_blobs', type=int, default=0)
    args = argument_parser.parse_args()
    image = imread(args.image)
    verbose = args.verbose
    intra = args.intra
    render = args.render

    start_time = time()
    if args.extra:
        from frame_recursive import frame_recursive
        frame = frame_recursive(image, intra, render, verbose)
    else:
        frame = frame_blobs_root(image, intra, render, verbose)

    end_time = time() - start_time
    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)

    '''
    Test fopen:
        if args.verbose:
        for i, blob in enumerate(frame.blob_):
        # check if fopen is correct
            # if fopen, y0 = 0, or x0 = 0, or yn = frame's y size or xn = frame's x size
            if blob.box[0] == 0 or blob.box[2] == 0 or blob.box[1] == blob.root_dert__[0].shape[0] or blob.box[3] == blob.root_dert__[0].shape[1]:
                if not blob.fopen: # fopen should be true when blob touches the frame boundaries
                    print('fopen is wrong on blob '+str(i))
    '''