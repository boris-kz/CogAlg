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
    The output is der__t: 2D array of pixel-mapped derts.
    - frame_blobs_root:
    Flood-fill segmentation of image der__t into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
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
from __future__ import annotations

import sys
from collections import deque
from time import time
from types import SimpleNamespace
import numpy as np
from visualization.draw_frame import visualize
from utils import kernel_slice_3x3 as ks, get_instance, box2slice, accum_box, sub_box2box
# from vectorize_edge.classes import Ct
# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED = -2
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

def frame_blobs_root(i__, intra=False, render=False, verbose=False):
    Y, X = i__.shape[:2]
    frame = SimpleNamespace(
        I=0, Dy=0, Dx=0, rdn=1,
        i__=i__, box=(0, 0, Y, X), rlayers=[[]], rng=1, prior_forks='')

    if verbose: start_time = time()

    dy__, dx__, g__ = frame.der__t = comp_pixel(i__)
    sign__ = ave - g__ > 0   # sign is positive for below-average g

    # https://en.wikipedia.org/wiki/Flood_fill:
    frame.rlayers[0], idmap, adj_pairs = flood_fill(
        frame, fork='', fork_ibox=(1, 1, Y - 1, X - 1),
        der__t=frame.der__t, sign__=sign__, verbose=verbose)
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    for blob in frame.rlayers[0]:
        frame.I += blob.I
        frame.Dy += blob.Dy
        frame.Dx += blob.Dx
    # dlayers = []: no comp_a yet
    if verbose: print(f"{len(frame.rlayers[0])} blobs formed in {time() - start_time} seconds")

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frame.rlayers += intra_blob_root(frame, render, verbose)  # recursive eval cross-comp range| angle| slice per blob
        if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks

    if render: visualize(frame)
    return frame


def comp_pixel(pi__):
    # compute directional derivatives:
    dy__ = (
        (pi__[ks.bl] - pi__[ks.tr]) * 0.25 +
        (pi__[ks.bc] - pi__[ks.tc]) * 0.50 +
        (pi__[ks.br] - pi__[ks.tl]) * 0.25
    )
    dx__ = (
        (pi__[ks.tr] - pi__[ks.bl]) * 0.25 +
        (pi__[ks.mr] - pi__[ks.mc]) * 0.50 +
        (pi__[ks.br] - pi__[ks.tl]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)                          # compute gradient magnitude

    return dy__, dx__, g__


def flood_fill(root_blob, fork, fork_ibox, der__t, sign__, mask__=None, verbose=False):
    dy__, dx__, g__ = der__t
    height, width = g__.shape  # = der__t shape
    fork_i__ = root_blob.i__[box2slice(fork_ibox)]
    assert height, width == fork_i__.shape  # same shape as der__t

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None: idmap[~mask__] = EXCLUDED
    if verbose:
        n_pixels = (height*width) if mask__ is None else mask__.sum()
        step = 100 / n_pixels  # progress % percent per pixel
        progress = 0.0; print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()
    blob_ = []
    adj_pairs = set()

    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts
                blob = SimpleNamespace(
                    I=0, Dy=0, Dx=0, A=0,  #  no rlayers yet
                    i__=root_blob.i__, sign=sign__[y, x], root_ibox=fork_ibox, root_der__t=der__t,
                    prior_forks=root_blob.prior_forks + fork,
                    box=(y, x, y + 1, x + 1), rng=root_blob.rng, fopen=False, rdn=1)
                blob_ += [blob]
                idmap[y, x] = id(blob)
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()
                    # add dert to blob
                    blob.I += fork_i__[y1][x1]
                    blob.Dy += dy__[y1][x1]
                    blob.Dx += dx__[y1][x1]
                    blob.A += 1
                    blob.box = accum_box(blob.box, y1, x1)
                    # neighbors coordinates, 4 for -, 8 for +
                    if blob.sign:   # include diagonals
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
                        if (y2 < 0 or y2 >= height or x2 < 0 or x2 >= width or
                                idmap[y2, x2] == EXCLUDED):
                            blob.fopen = True
                        # pixel is filled:
                        elif idmap[y2, x2] == UNFILLED:
                            # same-sign dert:
                            if blob.sign == sign__[y2, x2]:
                                idmap[y2, x2] = id(blob)  # add blob ID to each dert
                                unfilled_derts += [(y2, x2)]
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], id(blob)))  # blob.id always increases
                # terminate blob
                blob.ibox = sub_box2box(fork_ibox, blob.box)
                blob.der__t = tuple(par__[box2slice(blob.box)] for par__ in der__t)
                blob.mask__ = (idmap[box2slice(blob.box)] == id(blob))
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = np.hypot(blob.Dy, blob.Dx)
                if verbose:
                    progress += blob.A * step; print(f"\rClustering... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    return blob_, idmap, adj_pairs


def assign_adjacents(adj_pairs):  # adjacents are connected opposite-sign blobs
    '''
    Assign adjacent blobs bilaterally according to adjacent pairs' ids in blob_binder.
    '''
    for blob_id1, blob_id2 in adj_pairs:
        blob1 = get_instance(blob_id1)
        blob2 = get_instance(blob_id2)

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
            blob1.adj_blobs += [blob2]
            blob2.adj_blobs += [blob1]
        '''
        blob1.adj_blobs[0] += [blob2]
        blob1.adj_blobs[1] += [pose2]
        blob2.adj_blobs[0] += [blob1]
        blob2.adj_blobs[1] += [pose1]


if __name__ == "__main__":
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
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
    if args.extra:  # not functional yet
        from frame_recursive import frame_recursive
        frame = frame_recursive(image, intra, render, verbose)
    else:
        frame = frame_blobs_root(image, intra, render, verbose)

    end_time = time() - start_time
    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)