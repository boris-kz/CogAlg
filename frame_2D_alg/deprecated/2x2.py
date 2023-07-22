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

import sys
import numpy as np
from time import time
from collections import deque
from visualization.draw_frame_blobs import visualize_blobs
from class_cluster import ClusterStructure, init_param as z
from copy import copy
# from frame_blobs_wrapper import wrapped_flood_fill, from utils import minmax

# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
UNFILLED = -1
EXCLUDED_ID = -2
# FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, M, blob_, der__t')

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
    der__t : object = None
    der__t_roots : object = None  # map to der__t
    adj_blobs : list = z([])  # adjacent blobs
    fopen : bool = False
    # intra_blob params: # or pack in intra = lambda: Cintra
    # comp_angle:
    Sin_da0 : float = 0.0
    Cos_da0 : float = 0.0
    Sin_da1 : float = 0.0
    Cos_da1 : float = 0.0
    Ga : float = 0.0
    # comp_dx:
    Mdx : float = 0.0
    Ddx : float = 0.0
    # derivation hierarchy:
    root_der__t : list = z([])  # from root blob, to extend der__t
    prior_forks : str = ''
    fBa : bool = False  # in root_blob: next fork is comp angle, else comp_r
    rdn : float = 1.0  # redundancy to higher blob layers, or combined?
    rng : int = 1  # comp range, set before intra_comp
    P_ : list = z([])  # input + derPs, no internal sub-recursion
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

def frame_blobs_root(image, intra=False, render=False, verbose=False, use_c=False):

    if verbose: start_time = time()
    Y, X = image.shape[:2]
    der__t = comp_pixel(image)
    sign__ = ave - der__t[3] > 0   # sign is positive for below-average g in [i__, dy__, dx__, g__, ri]
    # https://en.wikipedia.org/wiki/Flood_fill:
    blob_, idmap, adj_pairs = flood_fill(der__t, sign__, prior_forks='', verbose=verbose)
    assign_adjacents(adj_pairs)  # forms adj_blobs per blob in adj_pairs
    I, Dy, Dx = 0, 0, 0
    for blob in blob_: I += blob.I; Dy += blob.Dy; Dx += blob.Dx

    frame = CBlob(I=I, Dy=Dy, Dx=Dx, der__t=der__t, rlayers=[blob_], box=(0,Y,0,X))
    # dlayers = []: no comp_a yet
    if verbose: print(f"{len(frame.rlayers[0])} blobs formed in {time() - start_time} seconds")

    if intra:  # omit for testing frame_blobs without intra_blob
        if verbose: print("\rRunning frame's intra_blob...")
        from intra_blob import intra_blob_root

        frame.rlayers += intra_blob_root(frame, render, verbose, fBa=0)  # recursive eval cross-comp range| angle| slice per blob
        if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)
        # sublayers[0] is fork-specific, deeper sublayers combine sub-blobs of both forks
    '''
    if use_c:  # old version, no longer updated:
        der__t = der__t[0], np.empty(0), np.empty(0), *der__t[1:], np.empty(0)
        frame, idmap, adj_pairs = wrapped_flood_fill(der__t)
    '''
    if render: visualize_blobs(frame)
    return frame

def comp_pixel(image):  # 2x2 pixel cross-correlation within image, see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    d_upright__ = bottomleft__ - topright__
    d_upleft__ = bottomright__ - topleft__

    # rotate back to 0 deg:
    dy__ = 0.5 * (d_upleft__ + d_upright__)
    dx__ = 0.5 * (d_upleft__ - d_upright__)

    G__ = np.hypot(d_upright__, d_upleft__)  # 2x2 kernel gradient (variation), match = inverse deviation, for sign_ only
    rp__ = topleft__ + topright__ + bottomleft__ + bottomright__  # sum of 4 rim pixels -> mean, not summed in blob param

    return (topleft__, dy__, dx__, G__, rp__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed der__t = (i__, dy__, dx__, g__, ri__) for readability in deeper functions
'''
    old version:
    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # decomposition of two diagonal differences into Gx
'''

def flood_fill(der__t, sign__, prior_forks, verbose=False, mask__=None, fseg=False):

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

                blob = CBlob(sign=sign__[y, x], root_der__t=der__t, prior_forks=prior_forks)
                blob_ += [blob]
                idmap[y, x] = blob.id
                y0, yn = y, y
                x0, xn = x, x
                # flood fill the blob, start from current position
                unfilled_derts = deque([(y, x)])
                while unfilled_derts:
                    y1, x1 = unfilled_derts.popleft()

                    # add dert to blob
                    if len(der__t) > 5: # comp_angle
                        blob.accumulate(I  = der__t[3][y1][x1],  # rp__,
                                        Dy = der__t[4][y1][x1],
                                        Dx = der__t[5][y1][x1],
                                        Sin_da0 = der__t[6][y1][x1],
                                        Cos_da0 = der__t[7][y1][x1],
                                        Sin_da1 = der__t[8][y1][x1],
                                        Cos_da1 = der__t[9][y1][x1])
                    else:  # comp_pixel or comp_range
                        blob.accumulate(I  = der__t[4][y1][x1],  # rp__,
                                        Dy = der__t[1][y1][x1],
                                        Dx = der__t[2][y1][x1])
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
                blob.der__t = tuple([par__[y0:yn, x0:xn] for par__ in blob.root_der__t])  # add None__ for m__?
                blob.mask__ = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses
                blob.G = np.hypot(blob.Dy, blob.Dx)  # recompute G
                if len(der__t) > 5:  # recompute Ga
                    blob.Ga = (blob.Cos_da0 + 1) + (blob.Cos_da1 + 1)  # +1 for all positives
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
'''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_angle: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - comp_slice_ forms roughly edge-orthogonal Ps, their stacks evaluated for rotation, comp_d, and comp_slice
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
'''
import numpy as np
from itertools import zip_longest

from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_r, comp_a
from vectorize_edge_blob.root import vectorize_root

# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
aveR = 10  # for range+, fixed overhead per blob
aveA = 10  # for angle+
pcoef = 2  # for vectorize_root; no ave_ga = .78, ave_ma = 2: no eval for comp_aa..
ave_nsub = 4  # ave n sub_blobs per blob: 4x higher costs? or eval costs only, separate clustering ave = aveB?
# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob_root(root_blob, render, verbose, fBa):  # recursive evaluation of cross-comp slice| range| angle per blob

    # deep_blobs = []  # for visualization
    spliced_layers = []
    if fBa: blob_ = root_blob.dlayers[0]
    else:   blob_ = root_blob.rlayers[0]

    for blob in blob_:  # fork-specific blobs, print('Processing blob number ' + str(bcount))
        # increment forking sequence: g -> r|a, a -> v
        extend_der__t(blob)  # der__t += 1: cross-comp in larger kernels or possible rotation
        blob.root_der__t = root_blob.der__t
        blob_height = blob.box[1] - blob.box[0]; blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions: Ly, Lx
            if root_blob.fBa:  # vectorize fork in angle blobs
                if (blob.G - aveA*(blob.rdn+2)) + (aveA*(blob.rdn+2) - blob.Ga) > 0:  # G * angle match, x2 costs?
                    blob.fBa = 0; blob.rdn = root_blob.rdn+1
                    blob.prior_forks += 'v'
                    if verbose: print('fork: v')  # if render and blob.A < 100: deep_blobs += [blob]
                    vectorize_root(blob, verbose=verbose)
            else:
                if blob.G < aveR * blob.rdn and blob.sign:  # below-average G, eval for comp_r
                    blob.fBa = 0; blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                    # comp_r 4x4:
                    new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                    sign__ = ave * (blob.rdn+1) - new_der__t[3] > 0  # m__ = ave - g__
                    # if min Ly and Lx, der__t>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] =\
                            cluster_fork_recursive( blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=0)
                # || forks:
                if blob.G > aveA * blob.rdn and not blob.sign:  # above-average G, eval for comp_a
                    blob.fBa = 1; blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                    # comp_a 2x2:
                    new_der__t, new_mask__ = comp_a(blob.der__t, blob.mask__)
                    sign__ = (new_der__t[1] - ave*(blob.rdn+1)) + (ave*(blob.rdn+1)*pcoef - new_der__t[2]) > 0
                    # vectorize if dev_gr + inv_dev_ga, if min Ly and Lx, der__t>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] =\
                            cluster_fork_recursive( blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=1)
            '''
            this is comp_r || comp_a, gap or overlap version:
            if aveBa < 1: blobs of ~average G are processed by both forks
            if aveBa > 1: blobs of ~average G are not processed

            else exclusive forks:
            vG = blob.G - ave_G  # deviation of gradient, from ave per blob, combined max rdn = blob.rdn+1:
            vvG = abs(vG) - ave_vG * blob.rdn  # 2nd deviation of gradient, from fixed costs of if "new_der__t" loop below
            # vvG = 0 maps to max G for comp_r if vG < 0, and to min G for comp_a if vG > 0:
            
            if blob.sign:  # sign of pixel-level g, which corresponds to sign of blob vG, so we don't need the later
                if vvG > 0:  # below-average G, eval for comp_r...
                elif vvG > 0:  # above-average G, eval for comp_a...
            '''
    # if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)

    return spliced_layers


def cluster_fork_recursive(blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa):

    fork = 'a' if fBa else 'r'
    if verbose: print('fork:', blob.prior_forks + fork)
    # form sub_blobs:
    sub_blobs, idmap, adj_pairs = \
        flood_fill(new_der__t, sign__, prior_forks=blob.prior_forks + fork, verbose=verbose, mask__=new_mask__)
    '''
    adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
    sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
    + if sub_blob_val > alt_blob_val, else -?  
    '''
    adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
    # blob.rdn += adj_rdn
    # for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
    assign_adjacents(adj_pairs)
    # if render: visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (froot_Ba = {blob.fBa}, froot_Ba = {blob.prior_forks[-1] == 'a'})")
    if fBa: sublayers = blob.dlayers
    else:   sublayers = blob.rlayers

    sublayers += [sub_blobs]  # r|a fork- specific sub_blobs, then add deeper layers of mixed-fork sub_blobs:
    sublayers += intra_blob_root(blob, render, verbose, fBa)  # recursive eval cross-comp range| angle| slice per blob

    new_spliced_layers = [spliced_layer + sublayer for spliced_layer, sublayer in
                          zip_longest(spliced_layers, sublayers, fillvalue=[])]
    return new_spliced_layers


def extend_der__t(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_der__t[0].shape  # higher dert size
    # set pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended
    # take ext_der__t from part of root_der__t:
    ext_der__t = []
    for par__ in blob.root_der__t:
        if isinstance(par__,list):  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
            ext_der__t += [par__[0][y0e:yne, x0e:xne]]
            ext_der__t += [par__[1][y0e:yne, x0e:xne]]
        else:
            ext_der__t += [par__[y0e:yne, x0e:xne]]
    ext_der__t = tuple(ext_der__t)  # change list to tuple
    # extend mask__:
    ext_mask__ = np.pad(blob.mask__,
                        ((y0 - y0e, yne - yn),
                         (x0 - x0e, xne - xn)),
                        constant_values=True, mode='constant')
    blob.der__t = ext_der__t
    blob.mask__ = ext_mask__
    blob.der__t_roots = [[[] for _ in range(x0e, xne)] for _ in range(y0e, yne)]
    if y0e != y0: blob.box = (y0+1, yn+1, x0, xn)
    if x0e != x0: blob.box = (y0, yn, x0+1, xn+1)


def print_deep_blob_forking(deep_layers):

    def check_deep_blob(deep_layer,i):
        for deep_blob_layer in deep_layer:
            if isinstance(deep_blob_layer,list):
                check_deep_blob(deep_blob_layer,i)
            else:
                print('blob num = '+str(i)+', forking = '+'->'.join([*deep_blob_layer.prior_forks]))

    for i, deep_layer in enumerate(deep_layers):
        if len(deep_layer)>0:
            check_deep_blob(deep_layer,i)

"""
Cross-comparison of pixels or gradient angles in 2x2 kernels
"""

import numpy as np
import functools
# no ave_ga = .78, ave_ma = 2  # at 22.5 degrees
# https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png

def comp_r(dert__, rng, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient in shorter-range cross-comp: input intensity didn't vary much.
    Such input is predictable enough for selective sampling: skipping current rim in following comparison kernels.
    Skipping forms increasingly sparse dert__ for next-range cross-comp,
    hence kernel width increases as 2^rng: 1: 2x2 kernel, 2: 4x4 kernel, 3: 8x8 kernel
    There is also skipping within greater-rng rims, so configuration of compared derts is always 2x2
    '''

    i__ = dert__[0]  # pixel intensity, should be separate from i__sum
    # sparse aligned rim arrays:
    i__topleft = i__[:-1:2, :-1:2]  # also assignment to new_dert__[0]
    i__topright = i__[:-1:2, 1::2]
    i__bottomleft = i__[1::2, :-1:2]
    i__bottomright = i__[1::2, 1::2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[:-1:2, :-1:2].astype(int)
                          + mask__[:-1:2, 1::2].astype(int)
                          + mask__[1::2, 1::2].astype(int)
                          + mask__[1::2, :-1:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function

    d_upleft__ = dert__[1][:-1:2, :-1:2].copy()  # sparse step=2 sampling
    d_upright__= dert__[2][:-1:2, :-1:2].copy()
    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 8x8, *4 for 16x16
    # combined distance and extrapolation coeffs, or separate distance coef: ave * (rave / dist), rave = ave abs d / ave i?
    # compare pixels diagonally:
    d_upright__+= (i__bottomleft - i__topright) * rngSkip
    d_upleft__ += (i__bottomright - i__topleft) * rngSkip

    g__ = np.hypot(d_upright__, d_upleft__)  # match = inverse of abs gradient (variation), recomputed at each comp_r
    ri__ = i__topleft + i__topright + i__bottomleft + i__bottomright

    return (i__topleft, d_upleft__, d_upright__, g__, ri__), majority_mask__


def comp_a(dert__, mask__=None):  # cross-comp of gradient angle in 2x2 kernels

    if mask__ is not None:
        majority_mask__ = (mask__[:-1, :-1].astype(int) +
                           mask__[:-1, 1:].astype(int) +
                           mask__[1:, 1:].astype(int) +
                           mask__[1:, :-1].astype(int)
                           ) > 1
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, ri__ = dert__[:5]  # day__,dax__,ma__ are recomputed

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        angle__ = [dy__, dx__] / np.hypot(dy__, dx__)
        for angle_ in angle__: angle_[np.where(np.isnan(angle_))] = 0  # set nan to 0, to avoid error later

    # angle__ shifted in 2x2 kernels:
    angle__topleft  = angle__[:, :-1, :-1]  # a is 3 dimensional
    angle__topright = angle__[:, :-1, 1:]
    angle__botright = angle__[:, 1:, 1:]
    angle__botleft  = angle__[:, 1:, :-1]

    sin_da0__, cos_da0__ = angle_diff(angle__botleft, angle__topright)  # dax__ contains 2 component arrays: sin(dax), cos(dax) ...
    sin_da1__, cos_da1__ = angle_diff(angle__botright, angle__topleft)  # ... same for day

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        ga__ = (cos_da0__ + 1) + (cos_da1__ + 1)  # +1 for all positives
        # or ga__ = np.hypot( np.arctan2(*day__), np.arctan2(*dax__)?

    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines
    day__ = [-sin_da0__ - sin_da1__, cos_da0__ + cos_da1__]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    dax__ = [-sin_da0__ + sin_da1__, cos_da0__ + cos_da1__]
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    in conventional notation: G = (Ix, Iy), A = (Ix, Iy) / hypot(G), DA = (dAdx, dAdy), abs_GA = hypot(DA)?
    '''
    i__ = i__[:-1, :-1]
    dy__ = dy__[:-1, :-1]  # passed on as idy, not rotated
    dx__ = dx__[:-1, :-1]  # passed on as idx, not rotated
    g__ = g__[:-1, :-1]
    ri__ = ri__[:-1, :-1]  # for summation in Dert

    return (i__, g__, ga__, ri__, dy__, dx__, day__[0], dax__[0], day__[1], dax__[1]), majority_mask__


def angle_diff(a2, a1):  # compare angle_1 to angle_2 (angle_1 to angle_2)

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]

'''
alternative versions below:
'''
def comp_r_odd(dert__, ave, rng, root_fia, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel,
    ...
    Sobel coefficients to decompose ds into dy and dx:
    YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
    XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
        |--(clockwise)--+  |--(clockwise)--+
        YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
                0       0  ¦          -2       2  ¦
                1   2   1  ¦          -1   0   1  ¦
    Scharr coefs:
    YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
    XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, using Sobel coeffs, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_d.drawio
    '''

    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    rotate in first call only: same orientation as from frame_blobs?
    '''
    i__center = i__[1:-1:2, 1:-1:2]  # also assignment to new_dert__[0]
    i__topleft = i__[:-2:2, :-2:2]
    i__top = i__[:-2:2, 1:-1:2]
    i__topright = i__[:-2:2, 2::2]
    i__right = i__[1:-1:2, 2::2]
    i__bottomright = i__[2::2, 2::2]
    i__bottom = i__[2::2, 1:-1:2]
    i__bottomleft = i__[2::2, :-2:2]
    i__left = i__[1:-1:2, :-2:2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[1:-1:2, 1:-1:2].astype(int)
                          + mask__[:-2:2, :-2:2].astype(int)
                          + mask__[:-2:2, 1:-1: 2].astype(int)
                          + mask__[:-2:2, 2::2].astype(int)
                          + mask__[1:-1:2, 2::2].astype(int)
                          + mask__[2::2, 2::2].astype(int)
                          + mask__[2::2, 1:-1:2].astype(int)
                          + mask__[2::2, :-2:2].astype(int)
                          + mask__[1:-1:2, :-2:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function
    '''
    can't happen:
    if root_fia:  # initialize derivatives:  
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)
    else: 
    '''
     # root fork is comp_r, accumulate derivatives:
    dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
    dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
    m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs * rim skip ratio:

    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 9x9, *4 for 17x17

    dy__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * -2  * rngSkip +
             (i__topright - i__bottomleft) * -1 * rngSkip +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 * rngSkip+
             (i__right - i__left) * 2 * rngSkip)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    '''
    m__ += ( abs(i__center - i__topleft) * 1 * rngSkip
           + abs(i__center - i__top) * 2 * rngSkip
           + abs(i__center - i__topright) * 1 * rngSkip
           + abs(i__center - i__right) * 2 * rngSkip
           + abs(i__center - i__bottomright) * 1 * rngSkip
           + abs(i__center - i__bottom) * 2 * rngSkip
           + abs(i__center - i__bottomleft) * 1 * rngSkip
           + abs(i__center - i__left) * 2 * rngSkip
           )

    return (i__center, dy__, dx__, g__, m__), majority_mask__


def comp_a_complex(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels
    '''
    More concise but also more opaque version
    https://github.com/khanh93vn/CogAlg/commit/1f3499c4545742486b89e878240d5c291b81f0ac
    '''
    if mask__ is not None:
        majority_mask__ = (mask__[:-1, :-1].astype(int) +
                           mask__[:-1, 1:].astype(int) +
                           mask__[1:, 1:].astype(int) +
                           mask__[1:, :-1].astype(int)
                           ) > 1
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    az__ = dx__ + 1j * dy__  # take the complex number (z), phase angle is now atan2(dy, dx)

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        az__ /=  np.absolute(az__)  # normalize by g, cosine = a__.real, sine = a__.imag

    # a__ shifted in 2x2 kernel, rotate 45 degrees counter-clockwise to cancel clockwise rotation in frame_blobs:
    az__left = az__[:-1, :-1]  # was topleft
    az__top = az__[:-1, 1:]  # was topright
    az__right = az__[1:, 1:]  # was botright
    az__bottom = az__[1:, :-1]  # was botleft
    '''
    imags and reals of the result are sines and cosines of difference between angles
    a__ is rotated 45 degrees counter-clockwise:
    '''
    dazx__ = az__right * az__left.conj()  # cos_az__right + j * sin_az__left
    dazy__ = az__bottom * az__top.conj()  # cos_az__bottom * j * sin_az__top

    dax__ = np.angle(dazx__)  # phase angle of the complex number, same as np.atan2(dazx__.imag, dazx__.real)
    day__ = np.angle(dazy__)

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        ma__ = .125 - (np.abs(dax__) + np.abs(day__)) / 2 * np.pi  # the result is in range in 0-1
    '''
    da deviation from ave da: 0.125 @ 22.5 deg: (π/8 + π/8) / 2*π, or 0.75 @ 45 deg: (π/4 + π/4) / 2*π
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(day__, dax__) - 0.2777  # same as old formula, atan2 and angle are equivalent
    '''
    ga deviation from ave = 0.2777 @ 22.5 deg, 0.5554 @ 45 degrees = π/4 radians, sqrt(0.5)*π/4 
    '''
    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):  # root fork is frame_blobs, recompute orthogonal dy and dx
        i__topleft = i__[:-1, :-1]
        i__topright = i__[:-1, 1:]
        i__botright = i__[1:, 1:]
        i__botleft = i__[1:, :-1]
        dy__ = (i__botleft + i__botright) - (i__topleft + i__topright)  # decomposition of two diagonal differences
        dx__ = (i__topright + i__botright) - (i__topleft + i__botleft)  # decomposition of two diagonal differences
    else:
        dy__ = dy__[:-1, :-1]  # passed on as idy, not rotated
        dx__ = dx__[:-1, :-1]  # passed on as idx, not rotated

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]

    return (i__, dy__, dx__, g__, m__, dazy__, dazx__, ga__, ma__), majority_mask__  # dazx__, dazy__ may not be needed


def angle_diff_complex(az2, az1):  # unpacked in comp_a
    '''
    compare phase angle of az1 to that of az2
    az1 = cos_1 + j*sin_1
    az2 = cos_2 + j*sin_2
    (sin_1, cos_1, sin_2, cos_2 below in angle_diff2)
    Assuming that the formula in angle_diff is correct, the result is:
    daz = cos_da + j*sin_da
    Substitute cos_da, sin_da (from angle_diff below):
    daz = (cos_1*cos_2 + sin_1*sin_2) + j*(cos_1*sin_2 - sin_1*cos_2)
        = (cos_1 - j*sin_1)*(cos_2 + j*sin_2)
    Substitute (1) and (2) into the above eq:
    daz = az1 * complex_conjugate_of_(az2)
    az1 = a + bj; az2 = c + dj
    daz = (a + bj)(c - dj)
        = (ac + bd) + (ad - bc)j
        (same as old formula, in angle_diff2() below)
     '''
    return az2 * az1.conj()  # imags and reals of the result are sines and cosines of difference between angles

