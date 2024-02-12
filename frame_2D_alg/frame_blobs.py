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
from numbers import Real
from typing import Any, NamedTuple, Tuple
from time import time
from copy import copy
import numpy as np
from visualization.draw_frame import visualize
from class_cluster import CBase, init_param as z
from utils import kernel_slice_3x3 as ks    # use in comp_pixel
# from vectorize_edge.classes import Ct
# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED = -2

class Cder__t(NamedTuple):
    dy: Any
    dx: Any
    g: Any

    def get_pixel(self, y: Real, x: Real) -> Tuple[Real, Real, Real]:
        return self.dy[y, x], self.dx[y, x], self.g[y, x]

class Cbox(NamedTuple):
    n: Real
    w: Real
    s: Real
    e: Real
    # properties
    @property
    def cy(self) -> Real: return (self.n + self.s) / 2
    @property
    def cx(self) -> Real: return (self.w + self.e) / 2
    @property
    def slice(self) -> Tuple[slice, slice]: return slice(self.n, self.s), slice(self.w, self.e)
    # operators:
    def extend(self, other: Cbox) -> Cbox:
        """Add 2 boxes."""
        return Cbox(min(self.n, other.n), min(self.w, other.w), max(self.s, other.s), max(self.e, other.e))

    # methods
    def __add__(self, other): return [a+b for a,b in zip(self, other)] if self else copy(other)
    def __sub__(self, other): return [a-b for a,b in zip(self, other)] if self else copy(other)

    def accumulate(self, y: Real, x: Real) -> Cbox:
        """Box coordinate accumulation."""
        return Cbox(min(self.n, y), min(self.w, x), max(self.s, y + 1), max(self.e, x + 1))

    def expand(self, r: int, h: Real, w: Real) -> Cbox:
        """Box expansion by margin r."""
        return Cbox(max(0, self.n - r), max(0, self.w - r), min(h, self.s + r), min(w, self.e + r))

    def shrink(self, r: int) -> Cbox:
        """Box shrink by margin r."""
        return Cbox(self.n + r, self.w + r, self.s - r, self.e - r)

    def sub_box2box(self, sb: Cbox) -> Cbox:
        """sub_box to box transform."""
        return Cbox(self.n + sb.n, self.w + sb.w, sb.s + self.n, sb.e + self.w)

    def box2sub_box(self, b: Cbox) -> Cbox:
        """box to sub_box transform."""
        return Cbox(b.n - self.n, b.w - self.w, b.s - self.n, b.e - self.w)


class CBlob(CBase):
    # comp_pixel:
    sign : bool = None
    I : float = 0.0
    Dy : float = 0.0
    Dx : float = 0.0
    G : float = 0.0
    A : float = 0.0 # blob area
    # composite params:
    M : float = 0.0 # summed PP.M, for both types of recursion?
    box : Cbox = Cbox(0, 0, 0, 0)  # n,w,s,e
    ibox : Cbox = Cbox(0, 0, 0, 0) # box for i__
    mask__ : object = None
    i__ : object = None     # reference to original input (no shrinking)
    der__t : Cder__t = None   # tuple of derivatives arrays, consistent in shape
    adj_blobs : list = z([])  # adjacent blobs
    fopen : bool = False
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_dx:
    Mdx : float = 0.0
    Ddx : float = 0.0
    # derivation hierarchy:
    root_ibox : Cbox = Cbox(0, 0, 0, 0)  # from root blob
    root_der__t : list = z([])  # from root blob
    prior_forks : str = ''
    fBa : bool = False  # in root_blob: next fork is comp angle, else comp_r
    rdn : float = 1.0  # redundancy to higher blob layers, or combined?
    rng : int = 1  # comp range, set before intra_comp
    rlayers : list = z([])  # list of layers across sub_blob derivation tree, deeper layers are nested with both forks
    dlayers : list = z([])  # separate for range and angle forks per blob
    PPm_ : list = z([])  # mblobs in frame
    PPd_ : list = z([])  # dblobs in frame
    valt : list = z([])  # PPm_ val, PPd_ val, += M,G?
    fsliced : bool = False  # from comp_slice
    root : object = None  # frame or from frame_bblob
    mgraph : object = None  # reference to converted blob
    dgraph : object = None  # reference to converted blob

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

    if verbose: start_time = time()
    Y, X = i__.shape[:2]
    der__t = comp_pixel(i__)
    sign__ = ave - der__t.g > 0   # sign is positive for below-average g
    frame = CBlob(i__=i__, der__t=der__t, box=Cbox(0, 0, Y, X), rlayers=[[]])
    fork_data = '', Cbox(1, 1, Y - 1, X - 1), der__t, sign__, None  # fork, fork_ibox, der__t, sign__, mask__
    # https://en.wikipedia.org/wiki/Flood_fill:
    frame.rlayers[0], idmap, adj_pairs = flood_fill(frame, fork_data, verbose=verbose)
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    for blob in frame.rlayers[0]:
        frame.accumulate(I=blob.I, Dy=blob.Dy, Dx=blob.Dx)
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
    G__ = np.hypot(dy__, dx__)                          # compute gradient magnitude

    return Cder__t(dy__, dx__, G__)


def flood_fill(root_blob, fork_data, verbose=False):
    # unpack and derive required fork data
    fork, fork_ibox, der__t, sign__, mask__ = fork_data
    height, width = der__t.g.shape  # = der__t shape
    fork_i__ = root_blob.i__[fork_ibox.slice]
    assert height, width == fork_i__.shape  # same shape as der__t

    idmap = np.full((height, width), UNFILLED, 'int32')  # blob's id per dert, initialized UNFILLED
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
                blob = CBlob(
                    i__=root_blob.i__, sign=sign__[y, x], root_ibox=fork_ibox, root_der__t=der__t,
                    box=Cbox(y, x, y + 1, x + 1), rng=root_blob.rng, prior_forks=root_blob.prior_forks + fork)
                blob_ += [blob]
                idmap[y, x] = blob.id
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()
                    # add dert to blob
                    blob.accumulate(I  = fork_i__[y1][x1],
                                    Dy = der__t.dy[y1][x1],
                                    Dx = der__t.dx[y1][x1])
                    blob.A += 1
                    blob.box = blob.box.accumulate(y1, x1)
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
                                idmap[y2, x2] = blob.id  # add blob ID to each dert
                                unfilled_derts += [(y2, x2)]
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))  # blob.id always increases
                # terminate blob
                blob.ibox = fork_ibox.sub_box2box(blob.box)
                blob.der__t = Cder__t(
                    *(par__[blob.box.slice] for par__ in der__t))
                blob.mask__ = (idmap[blob.box.slice] == blob.id)
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