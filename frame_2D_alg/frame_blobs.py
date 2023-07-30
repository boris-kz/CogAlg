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
    - comp_axis:
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

import sys
from typing import Union

import numpy as np
from time import time
from collections import deque, namedtuple
from visualization.draw_frame_blobs import visualize_blobs
from class_cluster import ClusterStructure, init_param as z
from utils import kernel_slice_3x3 as ks    # use in comp_pixel

# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100

# Constants:
UNFILLED = -1
EXCLUDED_ID = -2
DIAG_DIST = 2*2**0.5
ORTHO_DIST = 2

idert = namedtuple('idert', 'i, dy, dx, g')
adert = namedtuple('adert', 'i, g, ga, dy, dx, dyy, dyx, dxy, dxx')

class CBlob(ClusterStructure):
    # comp_pixel:
    sign : bool = None
    I : float = 0.0
    Dy : float = 0.0
    Dx : float = 0.0
    G : float = 0.0
    A : float = 0.0 # blob area
    # composite params:
    M : float = 0.0 # summed PP.M, for both types of recursion?
    box : tuple = (0,0,0,0)  # y0, yn, x0, xn
    mask__ : object = None
    der__t : Union[idert, adert] = None
    der__t_roots : object = None  # map to der__t
    adj_blobs : list = z([])  # adjacent blobs
    fopen : bool = False
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_angle:
    Dyy : float = 0.0
    Dyx : float = 0.0
    Dxy : float = 0.0
    Dxx : float = 0.0
    Ga : float = 0.0
    # derivation hierarchy:
    root_der__t : list = z([])  # from root blob, to extend der__t
    prior_forks : str = ''
    fBa : bool = False  # in root_blob: next fork is comp angle, else comp_r
    rdn : float = 1.0  # redundancy to higher blob layers, or combined?
    rng : int = 1  # comp range, set before intra_comp
    rlayers : list = z([])  # list of layers across sub_blob derivation tree, deeper layers are nested with both forks
    dlayers : list = z([])  # separate for range and angle forks per blob
    sub_blobs : list = z([[],[]])
    root : object = None  # frame or from frame_bblob
    # valt : list = z([])  # PPm_ val, PPd_ val, += M,G?
    # fsliced : bool = False  # from comp_slice
    # comp_dx:
    # Mdx : float = 0.0
    # Ddx : float = 0.0

class CEdge(ClusterStructure):  # edge blob
    sign: bool = None
    # replace with directional params:
    I : float = 0.0
    Ddl: float = 0.0    # down-left
    Dd: float = 0.0     # down
    Ddr: float = 0.0    # down-right
    Dr: float = 0.0     # right
    box: tuple = (0, 0, 0, 0)  # y0, yn, x0, xn
    mask__ : object = None
    dir__t : object = None
    dir__t_roots: object = None  # map to dir__t
    adj_blobs: list = z([])  # adjacent blobs
    node_ : list = z([])  # default P_, node_tt: list = z([[[],[]],[[],[]]]) in select PP_ or G_ forks
    root : object= None  # list root_ if fork overlap?
    derH : list = z([])  # formed in PPs, inherited in graphs
    aggH : list = z([[]])  # [[subH, valt, rdnt]]: cross-fork composition layers
    valt : list = z([0,0])
    rdnt : list = z([1,1])
    fback_ : list = z([])  # [feedback aggH,valt,rdnt per node]
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

def frame_blobs_root(image, intra=False, render=False, verbose=False):

    if verbose: start_time = time()
    Y, X = image.shape[:2]

    dir__t = comp_axis(image)  # nested tuple of 2D arrays: [i,[4|8 ds]]: single difference per axis
    i__, g__t = dir__t
    # combine ds: (diagonal is projected to orthogonal, cos(45) = sin(45) = 0.5**0.5)
    dy__ = (g__t[3]-g__t[2])*(0.5**0.5) + g__t[0]
    dx__ = (g__t[2]-g__t[3])*(0.5**0.5) + g__t[1]
    g__ = np.hypot(dy__, dx__)  # gradient magnitude
    der__t = i__, dy__, dx__, g__

    # compute signs
    g_sqr__t = g__t*g__t
    val__ = np.sqrt(
        # value is ratio between edge ones and the rest:
        # https://www.rastergrid.com/blog/2011/01/frei-chen-edge-detector/#:~:text=When%20we%20are%20using%20the%20Frei%2DChen%20masks%20for%20edge%20detection%20we%20are%20searching%20for%20the%20cosine%20defined%20above%20and%20we%20use%20the%20first%20four%20masks%20as%20the%20elements%20of%20importance%20so%20the%20first%20sum%20above%20goes%20from%20one%20to%20four.
        (g_sqr__t[0] + g_sqr__t[1] + g_sqr__t[2] + g_sqr__t[3]) /
        (g_sqr__t[0] + g_sqr__t[1] + g_sqr__t[2] + g_sqr__t[3] + g_sqr__t[4] + g_sqr__t[5] + g_sqr__t[6] + g_sqr__t[7] + g_sqr__t[8])
    )
    dsign__ = ave - val__ > 0   # max d per kernel
    gsign__ = ave - g__   > 0   # below-average g
    # https://en.wikipedia.org/wiki/Flood_fill
    # edge_, idmap, adj_pairs = flood_fill(dir__t, dsign__, prior_forks='', verbose=verbose, cls=CEdge)
    # assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    # I, Ddl, Dd, Ddr, Dr = 0, 0, 0, 0, 0
    # for edge in edge_: I += edge.I; Ddl += edge.Ddl; Dd += edge.Dd; Ddr += edge.Ddr; Dr += edge.Dr
    # frameE = CEdge(I=I, Ddl=Ddl, Dd=Dd, Ddr=Ddr, Dr=Dr, dir__t=dir__t, node_tt=[[[], blob_], [[], []]], box=(0, Y, 0, X))

    blob_, idmap, adj_pairs = flood_fill(der__t, gsign__, prior_forks='', verbose=verbose)  # overlap or for negative edge blobs only?
    assign_adjacents(adj_pairs)
    # not updated:
    I, Dy, Dx = 0, 0, 0
    for blob in blob_: I += blob.I; Dy += blob.Dy; Dx += blob.Dx
    frameB = CBlob(I=I, Dy=Dy, Dx=Dx, der__t=der__t, rlayers=[blob_], box=(0,Y,0,X))
    if verbose: print(f"{len(frameB.rlayers[0])} blobs formed in {time() - start_time} seconds")  # dlayers = []: no comp_a yet

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frameB.rlayers += intra_blob_root(frameB, render, verbose, fBa=0)  # recursive eval cross-comp range| angle| slice per blob
        if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks

    if render: visualize_blobs(frame)
    return frameB


def comp_axis(image):

    pi__ = np.pad(image, pad_width=1, mode='edge')      # pad image with edge values

    g___ = np.zeros((9,) + image.shape, dtype=float)    # g is gradient per axis

    # take 3x3 kernel slices of pixels:
    tl = pi__[ks.tl]; tc = pi__[ks.tc]; tr = pi__[ks.tr]
    ml = pi__[ks.ml]; mc = pi__[ks.mc]; mr = pi__[ks.mr]
    bl = pi__[ks.bl]; bc = pi__[ks.bc]; br = pi__[ks.br]

    # apply Frei-chen filter to image:
    # https://www.rastergrid.com/blog/2011/01/frei-chen-edge-detector/
    # First 4 values are edges:
    g___[0] = (tl+tr-bl-br)/DIAG_DIST + (tc-bc)/ORTHO_DIST
    g___[1] = (tl+bl-tr-br)/DIAG_DIST + (ml-mr)/ORTHO_DIST
    g___[2] = (ml+bc-tc-mr)/DIAG_DIST + (tr-bl)/ORTHO_DIST
    g___[3] = (mr+bc-tc-ml)/DIAG_DIST + (tr-bl)/ORTHO_DIST
    # The next 4 are lines
    g___[4] = (tc+bc-ml-mr)/ORTHO_DIST
    g___[5] = (tr+bl-tl-br)/ORTHO_DIST
    g___[6] = (mc*4-(tc+bc+ml+mr)*2+(tl+tr+bl+br))/6
    g___[7] = (mc*4-(tl+br+tr+bl)*2+(tc+bc+ml+mr))/6
    # The last one is average
    g___[8] = (tl+tc+tr+ml+mc+mr+bl+bc+br)/9

    return (pi__[ks.mc], g___)


# not fully revised to include edge fork:

def flood_fill(der__t, sign__, prior_forks, verbose=False, mask__=None, fseg=False, cls=CBlob):

    if mask__ is None: height, width = der__t[0].shape  # init der__t
    else:              height, width = mask__.shape  # intra der__t

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None:
        idmap[mask__] = EXCLUDED_ID
    if verbose:
        n_masked = 0 if mask__ is None else mask__.sum()
        step = 100 / (height * width - n_masked)  # progress % percent per pixel
        progress = 0.0; print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()
    blob_ = []
    adj_pairs = set()

    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts

                blob = cls(sign=sign__[y, x], root_der__t=der__t, prior_forks=prior_forks)
                blob_ += [blob]
                idmap[y, x] = blob.id
                y0, yn = y, y
                x0, xn = x, x
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()
                    # add dert to blob
                    if cls == CBlob:
                        if len(der__t) > 5: # comp_angle
                            blob.accumulate(I  = der__t[2][y1][x1],  # rp__,
                                            Dy = der__t[3][y1][x1],
                                            Dx = der__t[4][y1][x1],
                                            Dyy = der__t[5][y1][x1],
                                            Dyx = der__t[6][y1][x1],
                                            Dxy = der__t[7][y1][x1],
                                            Dxx = der__t[8][y1][x1])
                        else:  # comp_pixel or comp_range
                            blob.accumulate(I  = der__t[0][y1][x1],  # i__,  (this is i now)
                                            Dy = der__t[1][y1][x1],
                                            Dx = der__t[2][y1][x1])
                    else:  # edge
                        blob.accumulate(I   = der__t[0][y1][x1],
                                        Ddl = der__t[1][y1][x1],
                                        Dd  = der__t[2][y1][x1],
                                        Ddr = der__t[3][y1][x1],
                                        Dr  = der__t[4][y1][x1])
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
                                unfilled_derts += [(y2, x2)]
                        # else check if same-signed
                        elif blob.sign != sign__[y2, x2]:
                            adj_pairs.add((idmap[y2, x2], blob.id))  # blob.id always increases
                # terminate blob
                yn += 1; xn += 1
                blob.box = y0, yn, x0, xn
                '''
                blob.der__t = type(der__t)(
                    *(par__[y0:yn, x0:xn] for par__ in der__t))
                '''
                # getting error above : list/tuple expected at most 1 arguments, got 5
                blob.der__t = [(par__[y0:yn, x0:xn] for par__ in der__t)]
                blob.mask__ = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = recompute_dert(blob.Dy, blob.Dx)
                if len(der__t) > 5:
                    blob.Ga, blob.Dyy, blob.Dyx, blob.Dxy, blob.Dxx = recompute_adert(blob.Dyy, blob.Dyx, blob.Dxy, blob.Dxx)
                if verbose:
                    progress += blob.A * step; print(f"\rClustering... {round(progress)} %", end=""); sys.stdout.flush()
    if verbose: print("\r" + " " * 79, end=""); sys.stdout.flush(); print("\r", end="")

    return blob_, idmap, adj_pairs


def recompute_dert(Dy, Dx):   # recompute params after accumulation
    return np.hypot(Dy, Dx)  # recompute G from Dy, Dx


def recompute_adert(Dyy, Dyx, Dxy, Dxx):  # recompute angle fork params after accumulation
    # normalize
    Dyy, Dyx = [Dyy, Dyx] / np.hypot(Dyy, Dyx)
    Dxy, Dxx = [Dxy, Dxx] / np.hypot(Dxy, Dxx)
    # recompute Ga
    Ga = (1 - Dyx) + (1 - Dxx)  # +1 for all positives
    return Ga, Dyy, Dyx, Dxy, Dxx


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
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//toucan_small.jpg')
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

    '''
    Test fopen:
        if args.verbose:
        for i, blob in enumerate(frame.blob_):
        # check if fopen is correct
            # if fopen, y0 = 0, or x0 = 0, or yn = frame's y size or xn = frame's x size
            if blob.box[0] == 0 or blob.box[2] == 0 or blob.box[1] == blob.root_der__t[0].shape[0] or blob.box[3] == blob.root_der__t[0].shape[1]:
                if not blob.fopen: # fopen should be true when blob touches the frame boundaries
                    print('fopen is wrong on blob '+str(i))
    '''