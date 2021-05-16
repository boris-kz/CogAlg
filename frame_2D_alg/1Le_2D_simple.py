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
    - derts2blobs:
    Segmentation of image dert__ into blobs: contiguous areas of positive | negative deviation of gradient per kernel.
    Each blob is parameterized with summed params of constituent derts, derived by pixel cross-comparison (cross-correlation).
    These params represent predictive value per pixel, so they are also predictive on a blob level,
    thus should be cross-compared between blobs on the next level of search.
    - assign_adjacents:
    Each blob is assigned internal and external sets of opposite-sign blobs it is connected to.

    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs_intra_blob.drawio
'''

import sys
import numpy as np

from collections import deque
# from frame_blobs_wrapper import wrapped_flood_fill
from draw_frame_blobs import visualize_blobs
from utils import minmax
from collections import namedtuple
from class_cluster import ClusterStructure, NoneType

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback
aveB = 50
ave_M = 100
ave_mP = 100
UNFILLED = -1
EXCLUDED_ID = -2

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, G, M, blob_, dert__')


class CDert(ClusterStructure):
    # Dert params, comp_pixel:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # Dert params, comp_angle:
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    # Dert params, comp_dx:
    Mdx = int
    Ddx = int

class CFlatBlob(ClusterStructure):  # from frame_blobs only, no sub_blobs
    # Dert params
    Dert = object
    # blob params
    A = int  # blob area
    sign = NoneType
    box = list
    mask__ = object
    dert__ = object
    root_dert__ = object
    adj_blobs = list
    prior_forks = list
    fopen = bool

class CBlob(ClusterStructure):
    # Dert params, comp_pixel:
    Dert = object
    # blob params:
    A = int  # blob area
    sign = NoneType
    box = list
    mask__ = object
    dert__ = object
    root_dert__ = object
    fopen = bool     # the blob is bordering masked area
    f_root_a = bool  # input is from comp angle
    f_comp_a = bool  # current fork is comp angle
    fflip = bool     # x-y swap
    rdn = float      # redundancy to higher blob layers
    rng = int        # comp range
    # deep and external params:
    Ls = int   # for visibility and next-fork rdn
    sub_layers = list
    a_depth = int  # currently not used

    prior_forks = list
    adj_blobs = list  # for borrowing and merging
    dir_blobs = list  # primarily vertically | laterally oriented edge blobs
    fsliced = bool

    PPmm_ = list  # comp_slice_ if not empty
    PPdm_ = list  # comp_slice_ if not empty
    derP__ = list
    P__ = list
    PPmd_ = list  # PP_derPd_
    PPdd_ = list  # PP_derPd_
    derPd__ = list
    Pd__ = list

    # comp blobs
    DerBlob = object
    derBlob_ = list
    distance = int  # common per derBlob_
    neg_mB = int    # common per derBlob_


def comp_pixel(image):  # 2x2 pixel cross-correlation within image, a standard edge detection operator
    # see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    rot_Gy__ = bottomright__ - topleft__  # rotated to bottom__ - top__
    rot_Gx__ = topright__ - bottomleft__  # rotated to right__ - left__

    G__ = (np.hypot(rot_Gy__, rot_Gx__) - ave).astype('int')
    # deviation of central gradient per kernel, between four vertex pixels
    M__ = int(ave * 1.2) - (abs(rot_Gy__) + abs(rot_Gx__))
    # inverse deviation of SAD, which is a measure of variation. Ave * coeff = ave_SAD / ave_G, 1.2 is a guess

    return (topleft__, rot_Gy__, rot_Gx__, G__, M__)  # tuple of 2D arrays per param of dert (derivatives' tuple)
    # renamed dert__ = (p__, dy__, dx__, g__, m__) for readability in functions below
'''
    rotate dert__ 45 degrees clockwise, convert diagonals into orthogonals to avoid summation, which degrades accuracy of Gy, Gx
    Gy, Gx are used in comp_a, which returns them, as well as day, dax back to orthogonal
    
    Sobel version:
    Gy__ = -(topleft__ - bottomright__) - (topright__ - bottomleft__)   # decomposition of two diagonal differences into Gy
    Gx__ = -(topleft__ - bottomright__) + (topright__ - bottomleft__))  # decomposition of two diagonal differences into Gx

    old not-rotated version:
    Gy__ = ((bottomleft__ + bottomright__) - (topleft__ + topright__))  # decomposition of two diagonal differences into Gy
    Gx__ = ((topright__ + bottomright__) - (topleft__ + bottomleft__))  # decomposition of two diagonal differences into Gx
'''

def derts2blobs(dert__, verbose=False, render=False, use_c=False):

    if verbose: start_time = time()

    if use_c:
        dert__ = dert__[0], np.empty(0), np.empty(0), *dert__[1:], np.empty(0)
        frame, idmap, adj_pairs = wrapped_flood_fill(dert__)
    else:
        # [flood_fill](https://en.wikipedia.org/wiki/Flood_fill)
        blob_, idmap, adj_pairs = flood_fill(dert__, sign__=dert__[3] > 0,  verbose=verbose)
        I, Dy, Dx, G, M = 0, 0, 0, 0, 0
        for blob in blob_:
            I += blob.Dert.I
            Dy += blob.Dert.Dy
            Dx += blob.Dert.Dx
            G += blob.Dert.G
            M += blob.Dert.M
        frame = FrameOfBlobs(I=I, Dy=Dy, Dx=Dx, G=G, M=M, blob_=blob_, dert__=dert__)

    assign_adjacents(adj_pairs)  # f_segment_by_direction=False

    if verbose: print(f"{len(frame.blob_)} blobs formed in {time() - start_time} seconds")
    if render: visualize_blobs(idmap, frame.blob_)

    return frame


def accum_blob_Dert(blob, dert__, y, x):

    blob.Dert.I += dert__[0][y, x]
    blob.Dert.Dy += dert__[1][y, x]
    blob.Dert.Dx += dert__[2][y, x]
    blob.Dert.G += dert__[3][y, x]
    blob.Dert.M += dert__[4][y, x]


def flood_fill(dert__, sign__, verbose=False, mask__=None, blob_cls=CBlob, fseg=False, accum_func=accum_blob_Dert, prior_forks=[]):

    if mask__ is None: # non intra dert
        height, width = dert__[0].shape
    else: # intra dert
        height, width = mask__.shape

    idmap = np.full((height, width), UNFILLED, 'int64')  # blob's id per dert, initialized UNFILLED
    if mask__ is not None:
        idmap[mask__] = EXCLUDED_ID

    if verbose:
        step = 100 / height / width     # progress % percent per pixel
        progress = 0.0
        print(f"\rClustering... {round(progress)} %", end="");  sys.stdout.flush()

    blob_ = []
    adj_pairs = set()
    for y in range(height):
        for x in range(width):
            if idmap[y, x] == UNFILLED:  # ignore filled/clustered derts
                # initialize new blob
                blob = blob_cls(Dert=CDert(),sign=sign__[y, x], root_dert__=dert__)
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
                    if blob.sign or fseg:   # include diagonals
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
                blob.dert__ = tuple([param_dert__[y0:yn, x0:xn] for param_dert__ in blob.root_dert__])
                blob.mask__ = (idmap[y0:yn, x0:xn] != blob.id)
                blob.adj_blobs = [[],[]] # iblob.adj_blobs[0] = adj blobs, blob.adj_blobs[1] = poses

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
        '''
        if f_segment_by_direction:  # pose is not needed
            blob1.adj_blobs.append(blob2)
            blob2.adj_blobs.append(blob1)
        '''
        blob1.adj_blobs[0].append(blob2)
        blob1.adj_blobs[1].append(pose2)
        blob2.adj_blobs[0].append(blob1)
        blob2.adj_blobs[1].append(pose1)


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
    import argparse
    from time import time
    from utils import imread
    from comp_blob_draft import cross_comp_blobs

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
        root_dert__ = (  # update root dert__
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
            G = blob.Dert.G
            M = blob.Dert.M
            blob.root_dert__=root_dert__
            blob.prior_forks=['g']  # not sure about this
            blob_height = blob.box[1] - blob.box[0]
            blob_width = blob.box[3] - blob.box[2]

            if blob.sign:  # +G on first fork
                if G > aveB and blob_height > 3 and blob_width  > 3:  # min blob dimensions
                    blob.rdn = 1
                    blob.f_comp_a = 1
                    deep_layers[i] = intra_blob(blob, render=args.render, verbose=args.verbose)
                    # dert__ comp_a in 2x2 kernels

            elif M > aveB and blob_height > 3 and blob_width  > 3:  # min blob dimensions
                blob.rdn = 1
                blob.rng = 1
                blob.f_root_a = 0
                deep_layers[i] = intra_blob(blob, render=args.render, verbose=args.verbose)
                # dert__ comp_r in 3x3 kernels

            if deep_layers[i]:  # if there are deeper layers
                deep_blob_i_.append(i)  # indices of blobs with deep layers

        if args.verbose:
            print_deep_blob_forking(deep_layers)
            print("\rFinished intra_blob")

        bblob_ = cross_comp_blobs(frame)

    end_time = time() - start_time

    if args.verbose:
        print(f"\nSession ended in {end_time:.2} seconds", end="")
    else:
        print(end_time)

    '''
    Test fopen:
        if args.verbose:
        for i, blob in enumerate(frame.blob_):
        # simple check on correctness of fopen
            # if fopen, y0 = 0, or x0 = 0, or yn = frame's y size or xn = frame's x size
            if blob.box[0] == 0 or blob.box[2] == 0 or blob.box[1] == blob.root_dert__[0].shape[0] or blob.box[3] == blob.root_dert__[0].shape[1]:
                if not blob.fopen: # fopen should be true when blob touches the frame boundaries
                    print('fopen is wrong on blob '+str(i))
    '''

    '''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - comp_slice_ forms roughly edge-orthogonal Ps, their stacks evaluated for rotation, comp_d, and comp_slice
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
    -
    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = A, Ly, I, Dy, Dx, G, M, Day, Dax, Ga, Ma
    # A: area, Ly: vertical dimension, I: input; Dy, Dx: renamed Gy, Gx; G: gradient; M: match; Day, Dax, Ga, Ma: angle Dy, Dx, G, M
    sign,
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, dy, dx, g, m, day, dax, ga, ma
    # next fork:
    f_root_a,  # flag: input is from comp angle
    f_comp_a,  # flag: current fork is comp angle
    rdn,  # redundancy to higher layers
    rng,  # comparison range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''

import numpy as np
from frame_blobs import assign_adjacents, flood_fill, CBlob
from intra_comp import comp_r, comp_a
from draw_frame_blobs import visualize_blobs
from itertools import zip_longest
from comp_slice_ import *
from slice_utils import *
from segment_by_direction import segment_by_direction

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering

# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob(blob, **kwargs):  # slice_blob or recursive input rng+ | angle cross-comp within input blob

    Ave = int(ave * blob.rdn)
    AveB = int(aveB * blob.rdn)
    verbose = kwargs.get('verbose')

    if kwargs.get('render') is not None:  # don't render small blobs
        if blob.A < 100: kwargs['render'] = False
    spliced_layers = []  # to extend root_blob sub_layers

    if blob.f_root_a:
        # root fork is comp_a -> slice_blob
        if blob.mask__.shape[0] > 2 and blob.mask__.shape[1] > 2 and False in blob.mask__:  # min size in y and x, at least one dert in dert__

            if (-blob.Dert.M * blob.Dert.Ma - AveB > 0) and blob.Dert.Dx:  # vs. G reduced by Ga: * (1 - Ga / (4.45 * A)), max_ga=4.45
                blob.f_comp_a = 0
                blob.prior_forks.extend('p')
                if kwargs.get('verbose'): print('\nslice_blob fork\n')
                segment_by_direction(blob, verbose=True)
                # derP_ = slice_blob(blob, [])  # cross-comp of vertically consecutive Ps in selected stacks
                # blob.PP_ = derP_2_PP_(derP_, blob.PP_)  # form vertically contiguous patterns of patterns
    else:
        # root fork is frame_blobs or comp_r
        ext_dert__, ext_mask__ = extend_dert(blob)  # dert__ boundaries += 1, for cross-comp in larger kernels

        if blob.Dert.G > AveB:  # comp_a fork, replace G with borrow_M when known

            adert__, mask__ = comp_a(ext_dert__, Ave, blob.prior_forks, ext_mask__)  # compute ma and ga
            blob.f_comp_a = 1
            if kwargs.get('verbose'): print('\na fork\n')
            blob.prior_forks.extend('a')

            if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, least one dert in dert__
                sign__ = adert__[3] * adert__[8] > 0  # g * (ma / ave: deviation rate, no independent value, not co-measurable with g)

                dert__ = tuple([adert__[0], adert__[1], adert__[2], adert__[3], adert__[4],
                                adert__[5][0], adert__[5][1], adert__[6][0], adert__[6][1],
                                adert__[7], adert__[8]])  # flatten adert

                cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs)  # forms sub_blobs of sign in unmasked area
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.Dert.M > AveB * 1.2:  # comp_r fork, ave M = ave G * 1.2

            dert__, mask__ = comp_r(ext_dert__, Ave, blob.f_root_a, ext_mask__)
            blob.f_comp_a = 0
            if kwargs.get('verbose'): print('\na fork\n')
            blob.prior_forks.extend('r')

            if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, at least one dert in dert__
                sign__ = dert__[4] > 0  # m__ is inverse deviation of SAD

                cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs)  # forms sub_blobs of sign in unmasked area
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    return spliced_layers


def cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs):  # comp_r or comp_a eval per sub_blob:

    AveB = aveB * blob.rdn

    sub_blobs, idmap, adj_pairs = flood_fill(dert__, sign__, verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    if kwargs.get('render', False):
        visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (f_comp_a = {blob.f_comp_a}, f_root_a = {blob.f_root_a})")

    blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
    blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

    for sub_blob in sub_blobs:  # evaluate sub_blob
        G = blob.Dert.G  # Gr, Grr...
        #adj_M = blob.adj_blobs[3]  # adj_M is incomplete, computed within current dert_only, use root blobs instead:
        # adjacent valuable blobs of any sign are tracked from frame_blobs to form borrow_M?
        # track adjacency of sub_blobs: wrong sub-type but right macro-type: flat blobs of greater range?
        # G indicates or dert__ extend per blob G?

        # borrow_M = min(G, adj_M / 2)
        sub_blob.prior_forks = blob.prior_forks.copy()  # increments forking sequence: g->a, g->a->p, etc.

        if sub_blob.Dert.G > AveB:  # replace with borrow_M when known
            # comp_a:
            sub_blob.f_root_a = 1
            sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub_blob, currently not used
            sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
            blob.sub_layers += intra_blob(sub_blob, **kwargs)

        elif sub_blob.Dert.M - aveG > AveB:
            # comp_r:
            sub_blob.rng = blob.rng * 2
            sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
            blob.sub_layers += intra_blob(sub_blob, **kwargs)


def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_dert__[0].shape  # higher dert size

    # determine pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended

    # take ext_dert__ from part of root_dert__
    ext_dert__ = []
    for dert in blob.root_dert__:
        if type(dert) == list:  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
            ext_dert__.append(dert[0][y0e:yne, x0e:xne])
            ext_dert__.append(dert[1][y0e:yne, x0e:xne])
        else:
            ext_dert__.append(dert[y0e:yne, x0e:xne])
    ext_dert__ = tuple(ext_dert__)  # change list to tuple

    # extended mask__
    ext_mask__ = np.pad(blob.mask__,
                        ((y0 - y0e, yne - yn),
                         (x0 - x0e, xne - xn)),
                        constant_values=True, mode='constant')

    return ext_dert__, ext_mask__


def accum_blob_Dert(blob, dert__, y, x):
    blob.Dert.I += dert__[0][y, x]
    blob.Dert.Dy += dert__[1][y, x]
    blob.Dert.Dx += dert__[2][y, x]
    blob.Dert.G += dert__[3][y, x]
    blob.Dert.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dert.Dyy += dert__[5][y, x]
        blob.Dert.Dyx += dert__[6][y, x]
        blob.Dert.Dxy += dert__[7][y, x]
        blob.Dert.Dxx += dert__[8][y, x]
        blob.Dert.Ga += dert__[9][y, x]
        blob.Dert.Ma += dert__[10][y, x]


"""
Cross-comparison of pixels 3x3 kernels or gradient angles in 2x2 kernels
"""

import numpy as np
import functools

# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = np.array([-1, -2, -1, 0, 1, 2, 1, 0])
XCOEFs = np.array([-1, 0, 1, 2, 1, 0, -1, -2])
''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
            
Scharr coefs:
YCOEFs = np.array([-47, -162, -47, 0, 47, 162, 47, 0])
XCOEFs = np.array([-47, 0, 47, 162, 47, 0, -47, -162])
'''

def comp_r(dert__, ave, root_fia, mask__=None):
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
    rng = 4: 9x9 kernel,
    ...
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, and we use Sobel coeffs,
    see:
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

    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
        m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs:

    dy__ += ((i__topleft - i__bottomright) * -1 +
             (i__top - i__bottom) * -2 +
             (i__topright - i__bottomleft) * -1 +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 +
             (i__right - i__left) * 2)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    ave SAD = ave g * 1.2:
    '''
    m__ += int(ave * 1.2) - ( abs(i__center - i__topleft)
                            + abs(i__center - i__top) * 2
                            + abs(i__center - i__topright)
                            + abs(i__center - i__right) * 2
                            + abs(i__center - i__bottomright)
                            + abs(i__center - i__bottom) * 2
                            + abs(i__center - i__bottomleft)
                            + abs(i__center - i__left) * 2
                            )

    return (i__center, dy__, dx__, g__, m__), majority_mask__


def comp_a(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels

    # angles can't be summed: https://rosettacode.org/wiki/Averages/Mean_angle

    if mask__ is not None:
        majority_mask__ = (mask__[:-1, :-1].astype(int) +
                           mask__[:-1, 1:].astype(int) +
                           mask__[1:, 1:].astype(int) +
                           mask__[1:, :-1].astype(int)
                           ) > 1
    else:
        majority_mask__ = None

    i__, dy__, dx__, g__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    a__ = [dy__, dx__] / (g__ + ave + 0.001)  # + ave to restore abs g, + .001 to avoid / 0
    # g and m are rotation invariant, but da is more accurate with rot_a__:

    # a__ shifted in 2x2 kernel, rotate 45 degrees counter-clockwise to cancel clockwise rotation in frame_blobs:
    a__left   = a__[:, :-1, :-1]  # was topleft
    a__top    = a__[:, :-1, 1:]   # was topright
    a__right  = a__[:, 1:, 1:]    # was botright
    a__bottom = a__[:, 1:, :-1]   # was botleft

    sin_da0__, cos_da0__ = angle_diff(a__right, a__left)
    sin_da1__, cos_da1__ = angle_diff(a__bottom, a__top)

    ''' 
    match of angle = inverse deviation rate of SAD of angles from ave ma of all possible angles.
    we use ave 2: (2 + 2) / 2, 2 is average not-deviation ma, when da is 90 degree (because da varies from 0-180 degree). 
    That's just a rough guess, as all filter initializations, actual average will be lower because adjacent angles don't vary as much, 
    there is general correlation between proximity and similarity.
    Normally, we compute match as inverse deviation: ave - value. Here match is defined directly(?), so it's value - ave
    '''
    ma__ = (cos_da0__ + 1.001) + (cos_da1__ + 1.001) - 2  # +1 to convert to all positives, +.001 to avoid / 0, ave ma = 2

    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines
    day__ = [-sin_da0__ - sin_da1__, cos_da0__ + cos_da1__]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed
    dax__ = [-sin_da0__ + sin_da1__, cos_da0__ + cos_da1__]
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot( np.arctan2(*day__), np.arctan2(*dax__) )
    '''
    ga value is a deviation; interruption | wave is sign-agnostic: expected reversion, same for d sign?
    extended-kernel gradient from decomposed diffs: np.hypot(dydy, dxdy) + np.hypot(dydx, dxdx)?
    '''
    # if root fork is frame_blobs, recompute orthogonal dy and dx
    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):
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


    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask__


# -----------------------------------------------------------------------------
# Utilities

def angle_diff(a2, a1):  # compare angle_1 to angle_2

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]


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
        az__ /= np.absolute(az__)  # normalized, cosine = a__.real, sine = a__.imag

    # a__ shifted in 2x2 kernel, rotate 45 degrees counter-clockwise to cancel clockwise rotation in frame_blobs:
    az__left = az__[:-1, :-1]  # was topleft
    az__top = az__[:-1, 1:]  # was topright
    az__right = az__[1:, 1:]  # was botright
    az__bottom = az__[1:, :-1]  # was botleft

    dazx__ = angle_diff_complex(az__right, az__left)
    dazy__ = angle_diff_complex(az__bottom, az__top)
    # (a__ is rotated 45 degrees counter-clockwise)
    dax__ = np.angle(dazx__)  # phase angle of the complex number, same as np.atan2(dazx__.imag, dazx__.real)
    day__ = np.angle(dazy__)

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        ma__ = (np.abs(dax__) + np.abs(day__)) - 2 * np.pi  # * pi to make the result range in 0-1; or ave at 45 | 22 degree?
    '''
    sin(-θ) = -sin(θ), cos(-θ) = cos(θ): 
    sin(da) = -sin(-da), cos(da) = cos(-da) => (sin(-da), cos(-da)) = (-sin(da), cos(da))
    '''
    ga__ = np.hypot(day__, dax__)  # same as old formula, atan2 and angle are equivalent
    '''
    ga value is deviation; interruption | wave is sign-agnostic: expected reversion, same for d sign?
    extended-kernel gradient from decomposed diffs: np.hypot(dydy, dxdy) + np.hypot(dydx, dxdx)?
    '''
    # if root fork is frame_blobs, recompute orthogonal dy and dx

    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):
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

    # dax__, day__ may not be needed
    return (i__, dy__, dx__, g__, m__, dazy__, dazx__, ga__, ma__), majority_mask__


def angle_diff_complex(az2, az1):  # compare phase angle of az1 to that of az2
    # az1 = a + bj; az2 = c + dj
    # daz = (a + bj)(c - dj)
    #     = (ac + bd) + (ad - bc)j
    #     (same as old formula, in angle_diff2() below)
    return az1*az2.conj()  # imags and reals of the result are sines and cosines of difference between angles


'''
Comp_slice is a terminal fork of intra_blob.
-
It traces blob axis by cross-comparing vertically adjacent Ps: horizontal slices across an edge blob.
These low-M high-Ma blobs are vectorized into outlines of adjacent flat or high-M blobs.
(high match: M / Ma, roughly corresponds to low gradient: G / Ga)
-
Vectorization is clustering of Ps + their derivatives (derPs) into PPs: patterns of Ps that describe an edge.
This process is a reduced-dimensionality (2D->1D) version of cross-comp and clustering cycle, common across this project.
As we add higher dimensions (2D alg, 3D alg), this dimensionality reduction is done in salient high-aspect blobs
(likely edges / contours in 2D or surfaces in 3D) to form more compressed skeletal representations of full-D patterns.
-
Please see diagram:
https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/comp_slice_flip.drawio
'''

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
from frame_blobs import CDert
#from slice_utils import draw_PP_

import warnings  # to detect overflow issue, in case of infinity loop
warnings.filterwarnings('error')

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_dX = 10  # difference between median x coords of consecutive Ps
ave_Dx = 10
ave_mP = 8  # just a random number right now.
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20

class CP(ClusterStructure):

    Dert = object  # summed kernel parameters
    L = int
    x0 = int
    dX = int  # shift of average x between P and _P, if any
    y = int  # for visualization only
    sign = NoneType  # sign of gradient deviation
    dert_ = list   # array of pixel-level derts: (p, dy, dx, g, m), extended in intra_blob
    upconnect_ = list
    downconnect_cnt = int
    derP = object # derP object reference
    # only in Pd:
    Pm = object  # reference to root P
    dxdert_ = list
    # only in Pm:
    Pd_ = list

class CderDert(ClusterStructure):

    mP = int
    dP = int
    mx = int
    dx = int
    mL = int
    dL = int
    mDx = int
    dDx = int
    mDy = int
    dDy = int
    # dDdx,mDdx,dMdx,mMdx is used by comp_dx
    mDyy = int
    mDyx = int
    mDxy = int
    mDxx = int
    mGa = int
    mMa = int
    mMdx = int
    mDdx = int
    dDyy = int
    dDyx = int
    dDxy = int
    dDxx = int
    dGa = int
    dMa = int
    dMdx = int
    dDdx = int


class CderP(ClusterStructure):

    derDert = object
    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    # from comp_dx
    fdx = NoneType


class CPP(ClusterStructure):

    Dert  = object  # set of P params accumulated in PP
    derDert = object  # set of derP params accumulated in PP
    # between PPs:
    upconnect_ = list
    downconnect_cnt = int
    fPPm = NoneType  # PPm if 1, else PPd; not needed if packed in PP_?
    fdiv = NoneType
    box = list   # for visualization only, original box before flipping
    dert__ = list
    mask__ = bool
    # PP params
    derP__ = list
    P__ = list
    PPmm_ = list
    PPdm_ = list
    # PPd params
    derPd__ = list
    Pd__ = list
    PPmd_ = list
    PPdd_ = list  # comp_dx params

# Functions:
'''
leading '_' denotes higher-line variable or structure, vs. same-type lower-line variable or structure
trailing '_' denotes array name, vs. same-name elements of that array. '__' is a 2D array
leading 'f' denotes flag
-
rough workflow:
-
intra_blob -> slice_blob(blob) -> derP_ -> PP,
if flip_val(PP is FPP): pack FPP in blob.PP_ -> flip FPP.dert__ -> slice_blob(FPP) -> pack PP in FPP.PP_
else       (PP is PP):  pack PP in blob.PP_
'''

def slice_blob(blob, verbose=False):
    '''
    Slice_blob converts selected smooth-edge blobs (high G, low Ga or low M, high Ma) into sliced blobs,
    adding horizontal blob slices: Ps or 1D patterns
    '''
    dert__ = blob.dert__
    mask__ = blob.mask__
    height, width = dert__[0].shape
    if verbose: print("Converting to image...")

    for fPPd in range(2):  # run twice, 1st loop fPPd=0: form PPs, 2nd loop fPPd=1: form PPds

        P__ , derP__, Pd__, derPd__ = [], [], [], []
        zip_dert__ = zip(*dert__)
        _P_ = form_P_(list(zip(*next(zip_dert__))), mask__[0], 0)  # 1st upper row
        P__ += _P_  # frame of Ps

        for y, dert_ in enumerate(zip_dert__, start=1):  # scan top down
            if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

            P_ = form_P_(list(zip(*dert_)), mask__[y], y)  # horizontal clustering - lower row
            derP_ = scan_P_(P_, _P_)  # tests for x overlap between Ps, calls comp_slice

            Pd_ = form_Pd_(P_)  # form Pds within Ps
            derPd_ = scan_Pd_(P_, _P_)  # adds upconnect_ in Pds and calls derPd_2_PP_derPd_, same as derP_2_PP_

            derP__ += derP_; derPd__ += derPd_  # frame of derPs
            P__ += P_; Pd__ += Pd_
            _P_ = P_  # set current lower row P_ as next upper row _P_

        form_PP_root(blob, derP__, P__, derPd__, Pd__, fPPd)  # form PPs in blob or in FPP

    # yet to be updated
    # draw PPs
    #    if not isinstance(blob, CPP):
    #        draw_PP_(blob)


def form_P_(idert_, mask_, y):  # segment dert__ into P__ in horizontal ) vertical order, sum dert params into P params

    P_ = []  # rows of derPs
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0 = 0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # pixel mask

        if mask:  # masks: if 1,_0: P termination, if 0,_1: P initialization, if 0,_0: P accumulation:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma), L=L, x0=x0, dert_=dert_, y=y)
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; L = 1; x0 = x; dert_ = [dert]
            else:
                I += dert[0]  # _dert is not masked, accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                Dy += dert[1]
                Dx += dert[2]
                G += dert[3]
                M += dert[4]
                Dyy += dert[5]
                Dyx += dert[6]
                Dxy += dert[7]
                Dxx += dert[8]
                Ga += dert[9]
                Ma += dert[10]
                L += 1
                dert_.append(dert)
        _mask = mask

    if ~_mask:  # terminate last P in a row
        P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma), L=L, x0=x0, dert_=dert_, y=y)
        P_.append(P)

    return P_

def form_Pd_(P_):  # form Pds from Pm derts by dx sign, otherwise same as form_P

    Pd__ = []
    for iP in P_:
        if (iP.downconnect_cnt>0) or (iP.upconnect_):  # form Pd s if at least one connect in P, else they won't be compared
            P_Ddx = 0  # sum of Ddx across Pd s
            P_Mdx = 0  # sum of Mdx across Pd s
            Pd_ = []   # Pds in P
            _dert = iP.dert_[0]  # 1st dert
            dert_ = [_dert]
            I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma= _dert; L = 1; x0 = iP.x0  # initialize P params with first dert
            _sign = _dert[2] > 0
            x = 1  # relative x within P

            for dert in iP.dert_[1:]:
                sign = dert[2] > 0
                if sign == _sign: # same Dx sign
                    I += dert[0]  # accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                    Dy += dert[1]
                    Dx += dert[2]
                    G += dert[3]
                    M += dert[4]
                    Dyy += dert[5]
                    Dyx += dert[6]
                    Dxy += dert[7]
                    Dxx += dert[8]
                    Ga += dert[9]
                    Ma += dert[10]
                    L += 1
                    dert_.append(dert)

                else:  # sign change, terminate P
                    P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma),
                           L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
                    if Dx > ave_Dx:
                        # cross-comp of dx in P.dert_
                        comp_dx(P); P_Ddx += P.Dert.Ddx; P_Mdx += P.Dert.Mdx
                    Pd_.append(P)
                    # reinitialize params
                    I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; x0 = iP.x0+x; L = 1; dert_ = [dert]

                _sign = sign
                x += 1
            # terminate last P
            P = CP(Dert=CDert(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma),
                   L=L, x0=x0, dert_=dert_, y=iP.y, sign=_sign, Pm=iP)
            if Dx > ave_Dx:
                comp_dx(P); P_Ddx += P.Dert.Ddx; P_Mdx += P.Dert.Mdx
            Pd_.append(P)
            # update Pd params in P
            iP.Pd_ = Pd_; iP.Dert.Ddx = P_Ddx; iP.Dert.Mdx = P_Mdx
            Pd__ += Pd_

    return Pd__


def scan_P_(P_, _P_):  # test for x overlap between Ps, call comp_slice

    derP_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            # test for x overlap between P and _P in 8 directions
            if (P.x0 - 1 < (_P.x0 + _P.L) and (P.x0 + P.L) + 1 > _P.x0):  # all Ps here are positive

                fcomp = [1 for derP in P.upconnect_ if P is derP.P]  # upconnect could be derP or dirP
                if not fcomp:
                    derP = comp_slice_full(_P, P)  # form vertical and directional derivatives
                    derP_.append(derP)
                    P.upconnect_.append(derP)
                    _P.downconnect_cnt += 1

            elif (P.x0 + P.L) < _P.x0:  # stop scanning the rest of lower P_ if there is no overlap
                break
    return derP_


def scan_Pd_(P_, _P_):  # test for x overlap between Pds

    derPd_ = []
    for P in P_:  # lower row
        for _P in _P_:  # upper row
            for Pd in P.Pd_: # lower row Pds
                for _Pd in _P.Pd_: # upper row Pds
                    # test for same sign & x overlap between Pd and _Pd in 8 directions
                    if (Pd.x0 - 1 < (_Pd.x0 + _Pd.L) and (Pd.x0 + Pd.L) + 1 > _Pd.x0) and (Pd.sign == _Pd.sign):

                        fcomp = [1 for derPd in Pd.upconnect_ if Pd is derPd.P]  # upconnect could be derP or dirP
                        if not fcomp:
                            derPd = comp_slice_full(_Pd, Pd)
                            derPd_.append(derPd)
                            Pd.upconnect_.append(derPd)
                            _Pd.downconnect_cnt += 1

                    elif (Pd.x0 + Pd.L) < _Pd.x0:  # stop scanning the rest of lower P_ if there is no overlap
                        break
    return derPd_


def form_PP_root(blob, derP__, P__, derPd__, Pd__, fPPd):
    '''
    form vertically contiguous patterns of patterns by the sign of derP, in blob or in FPP
    '''
    blob.derP__ = derP__; blob.P__ = P__
    blob.derPd__ = derPd__; blob.Pd__ = Pd__
    if fPPd:
        derP_2_PP_(blob.derP__, blob.PPdm_, 0, 1)   # cluster by derPm dP sign
        derP_2_PP_(blob.derPd__, blob.PPdd_, 1, 1)  # cluster by derPd dP sign, not used
    else:
        derP_2_PP_(blob.derP__, blob.PPmm_, 0, 0)   # cluster by derPm mP sign
        derP_2_PP_(blob.derPd__, blob.PPmd_, 1, 0)  # cluster by derPd mP sign, not used


def derP_2_PP_(derP_, PP_, fderPd, fPPd):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
            PP = CPP(Dert=CDert(), derDert=CderDert())  # init
            accum_PP(PP,derP)

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_, fderPd, fPPd)  # form PPs across _P upconnects
            else:
                PP_.append(derP.PP)


def upconnect_2_PP_(iderP, PP_, fderPd, fPPd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP__:  # derP should not in current iPP derP_ list, but this may occur after the PP merging

            if fPPd: same_sign = (iderP.derDert.dP > 0) == (derP.derDert.dP > 0)  # comp dP sign
            else: same_sign = (iderP.derDert.mP > 0) == (derP.derDert.mP > 0)  # comp mP sign

            if same_sign:  # upconnect derP has different PP, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else:  # accumulate derP in current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            elif not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                PP = CPP(Dert=CDert(), derDert=CderDert())
                accum_PP(PP,derP)
                derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fderPd, fPPd)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        PP_.append(iderP.PP)  # iPP is terminated after all upconnects are checked


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP__:
        if derP not in _PP.derP__:
            _PP.derP__.append(derP)
            derP.PP = _PP  # update reference
            # accumulate Dert
            _PP.Dert.accumulate(**{param:getattr(derP.P.Dert, param) for param in _PP.Dert.numeric_params})
            # accumulate derDert
            _PP.derDert.accumulate(**{param:getattr(derP.derDert, param) for param in _PP.derDert.numeric_params})

    if PP in PP_:
        PP_.remove(PP)  # remove merged PP


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})

def accum_PP(PP, derP):  # accumulate params in PP
    # accumulate Dert
    PP.Dert.accumulate(**{param:getattr(derP.P.Dert, param) for param in PP.Dert.numeric_params})
    # accumulate derDert
    PP.derDert.accumulate(**{param:getattr(derP.derDert, param) for param in PP.derDert.numeric_params})

    PP.derP__.append(derP)

    derP.PP = PP  # update reference


def comp_dx(P):  # cross-comp of dx s in P.dert_

    Ddx = 0
    Mdx = 0
    dxdert_ = []
    _dx = P.dert_[0][2]  # first dx
    for dert in P.dert_[1:]:
        dx = dert[2]
        ddx = dx - _dx
        if dx > 0 == _dx > 0: mdx = min(dx, _dx)
        else: mdx = -min(abs(dx), abs(_dx))
        dxdert_.append((ddx, mdx))  # no dx: already in dert_
        Ddx += ddx  # P-wide cross-sign, P.L is too short to form sub_Ps
        Mdx += mdx
        _dx = dx
    P.dxdert_ = dxdert_
    P.Dert.Ddx = Ddx
    P.Dert.Mdx = Mdx


def comp_slice(_P, P, _derP_):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, Dx, Dy, G, M, L, Ddx, Mdx = P.sign, P.x0, P.Dert.Dx, P.Dert.Dy, P.Dert.G, P.Dert.M, P.L, P.Dert.Ddx, P.Dert.Mdx  # params per comp branch
    _s, _x0, _Dx, _Dy, _G, _M, _dX, _L, _Ddx, _Mdx = _P.sign, _P.x0, _P.Dert.Dx, _P.Dert.Dy, _P.Dert.G, _P.Dert.M, _P.dX, _P.L, _P.Dert.Ddx, _P.Dert.Mdx

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?
    hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1

    L /= hyp  # orthogonal L is reduced by hyp
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    M /= hyp  # orthogonal M is reduced by hyp
    dM = M - _M; mM = min(M, _M)  # use abs M?  no Mx, My: non-core, lesser and redundant bias?

    dP = dL + dM  # -> directional PPd, equal-weight params, no rdn?
    mP = mL + mM  # -> complementary PPm, rdn *= Pd | Pm rolp?
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    P.Dert.flip_val = (dX * (P.Dert.Dy / (P.Dert.Dx+.001)) - flip_ave)  # +.001 to avoid division by zero

    derP = CderP(derDert=CderDert(mP=mP, dP=dP, dX=dX, mL=mL, dL=dL), P=P, _P=_P)
    P.derP = derP

    return derP


def comp_slice_full(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    s, x0, Dx, Dy, G, M, L, Ddx, Mdx = P.sign, P.x0, P.Dert.Dx, P.Dert.Dy, P.Dert.G, P.Dert.M, P.L, P.Dert.Ddx, P.Dert.Mdx
    # params per comp branch, add angle params
    _s, _x0, _Dx, _Dy, _G, _M, _dX, _L, _Ddx, _Mdx = _P.sign, _P.x0, _P.Dert.Dx, _P.Dert.Dy, _P.Dert.G, _P.Dert.M, _P.dX, _P.L, _P.Dert.Ddx, _P.Dert.Mdx

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        xn = x0 + L - 1
        _xn = _x0 + _L - 1
        mX = min(xn, _xn) - max(x0, _x0)  # overlap = abs proximity: summed binary x match
        rX = dX / mX if mX else dX*2  # average dist / prox, | prox / dist, | mX / max_L?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?

    if dX * P.Dert.G > ave_ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
        # diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/orthogonalization.png
        # Long axis is a curve of connections between ave_xs: mid-points of consecutive Ps.

        # Ortho virtually rotates P to connection-orthogonal direction:
        hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1
        L = L / hyp  # orthogonal L
        # combine derivatives in proportion to the contribution of their axes to orthogonal axes:
        # contribution of Dx should increase with hyp(dX,dY=1), this is original direction of Dx:
        Dy = (Dy / hyp + Dx * hyp) / 2  # estimated along-axis D
        Dx = (Dy * hyp + Dx / hyp) / 2  # estimated cross-axis D
        '''
        alternatives:
        oDy = (Dy * hyp - Dx / hyp) / 2;  oDx = (Dx / hyp + Dy * hyp) / 2;  or:
        oDy = hypot( Dy / hyp, Dx * hyp);  oDx = hypot( Dy * hyp, Dx / hyp)
        '''
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    dM = M - _M; mM = min(M, _M)  # use abs M?  no Mx, My: non-core, lesser and redundant bias?
    # no comp G: Dy, Dx are more specific:
    dDx = Dx - _Dx  # same-sign Dx if Pd
    mDx = min(abs(Dx), abs(_Dx))
    if Dx > 0 != _Dx > 0: mDx = -mDx
    # min is value distance for opposite-sign comparands, vs. value overlap for same-sign comparands
    dDy = Dy - _Dy  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI
    mDy = min(abs(Dy), abs(_Dy))
    if (Dy > 0) != (_Dy > 0): mDy = -mDy

    dDdx, dMdx, mDdx, mMdx = 0, 0, 0, 0
    if P.dxdert_ and _P.dxdert_:  # from comp_dx
        fdx = 1
        dDdx = Ddx - _Ddx
        mDdx = min( abs(Ddx), abs(_Ddx))
        if (Ddx > 0) != (_Ddx > 0): mDdx = -mDdx
        # Mdx is signed:
        dMdx = min( Mdx, _Mdx)
        mMdx = -min( abs(Mdx), abs(_Mdx))
        if (Mdx > 0) != (_Mdx > 0): mMdx = -mMdx
    else:
        fdx = 0
    # coeff = 0.7 for semi redundant parameters, 0.5 for fully redundant parameters:
    dP = ddX + dL + 0.7*(dM + dDx + dDy)  # -> directional PPd, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?
    if fdx: dP += 0.7*(dDdx + dMdx)

    mP = mdX + mL + 0.7*(mM + mDx + mDy)  # -> complementary PPm, rdn *= Pd | Pm rolp?
    if fdx: mP += 0.7*(mDdx + mMdx)
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    derP = CderP(P=P, _P=_P, derDert=CderDert(mP=mP, dP=dP, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy))
    P.derP = derP

    if fdx:
        derP.fdx=1; derP.derDert.dDdx=dDdx; derP.derDert.mDdx=mDdx; derP.derDert.dMdx=dMdx; derP.derDert.mMdx=mMdx

    '''
    min comp for rotation: L, Dy, Dx, no redundancy?
    mParam weighting by relative contribution to mP, /= redundancy?
    div_f, nvars: if abs dP per PPd, primary comp L, the rest is normalized?
    '''
    return derP

''' radial comp extension for co-internal blobs:
    != sign comp x sum( adj_blob_) -> intra_comp value, isolation value, cross-sign merge if weak, else:
    == sign comp x ind( adj_adj_blob_) -> same-sign merge | composition:
    borrow = adj_G * rA: default sum div_comp S -> relative area and distance to adjj_blob_
    internal sum comp if mA: in thin lines only? comp_norm_G or div_comp_G -> rG?
    isolation = decay + contrast:
    G - G * (rA * ave_rG: decay) - (rA * adj_G: contrast, = lend | borrow, no need to compare vG?)
    if isolation: cross adjj_blob composition eval,
    else:         cross adjj_blob merge eval:
    blob merger if internal match (~raG) - isolation, rdn external match:
    blob compos if external match (~rA?) + isolation,
    Also eval comp_slice over fork_?
    rng+ should preserve resolution: rng+_dert_ is dert layers,
    rng_sum-> rng+, der+: whole rng, rng_incr-> angle / past vs next g,
    rdn Rng | rng_ eval at rng term, Rng -= lost coord bits mag, always > discr?
    
    Add comp_PP_recursive
'''


'''
Cross-compare blobs with incrementally intermediate adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object
    _blob = object
    mB = int
    dB = int
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int

class CBblob(ClusterStructure):

    DerBlob = object
    blob_ = list

ave_mB = 0  # ave can't be negative
ave_rM = .7  # average relative match at rL=1: rate of ave_mB decay with relative distance, due to correlation between proximity and similarity

def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_

    for blob in blob_:  # each blob forms derBlob per compared adj_blob and accumulates adj_blobs'derBlobs:
        if not isinstance(blob.DerBlob, CderBlob):  # blob was not compared as adj_blob, forming derBlob
            blob.DerBlob = CderBlob()

        comp_blob_recursive(blob, blob.adj_blobs[0], derBlob_=[])
        # derBlob_ and derBlob_id_ are local and frame-wide

    bblob_ = form_bblob_(blob_)  # form blobs of blobs, connected by mutual match

    visualize_cluster_(bblob_, blob_, frame)

    return bblob_


def comp_blob_recursive(blob, adj_blob_, derBlob_):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    derBlob_pair_ = [ [derBlob.blob, derBlob._blob]  for derBlob in derBlob_]  # blob, adj_blob pair

    for adj_blob in adj_blob_:
        if [blob, adj_blob] in derBlob_pair_:  # blob was compared in prior function call
            break
        elif [adj_blob, blob] in derBlob_pair_:  # derBlob.blob=adj_blob, derBlob._blob=blob
            derBlob = derBlob_[derBlob_pair_.index([adj_blob,blob])]
            accum_derBlob(blob, derBlob)  # also adj_blob.rdn += 1?
        else:  # form new derBlob
            derBlob = comp_blob(blob, adj_blob)  # compare blob and adjacent blob
            accum_derBlob(blob, derBlob)         # from all compared blobs, regardless of mB sign
            derBlob_.append(derBlob)             # also frame-wide

        if derBlob.mB > 0:  # replace blob with adj_blob for continued adjacency search:

            if not isinstance(adj_blob.DerBlob, CderBlob):  # else DerBlob was formed in previous call
                adj_blob.DerBlob = CderBlob()
            comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], derBlob_)  # search depth could be different, compare anyway
            break
        elif blob.Dert.M + blob.neg_mB + derBlob.mB > ave_mB:  # neg mB but positive comb M,
            # extend blob comparison to adjacents of adjacent, depth-first
            blob.neg_mB += derBlob.mB  # mB and distance are accumulated over comparison scope
            blob.distance += np.sqrt(adj_blob.A)
            comp_blob_recursive(blob, adj_blob.adj_blobs[0], derBlob_)

    '''
    if blob.id not in blob_id_:
        blob_id_.append(blob.id)  # to prevent redundant (adj_blob, blob) derBlobs, local per blob
        _blob_id = [derBlob._blob.id for derBlob in blob.derBlob_] + \
                   [derBlob.blob.id for derBlob in blob.derBlob_] # list of derBlob.blobs & derBlob._blobs, local to current function
        for adj_blob in adj_blob_:
            if adj_blob.id not in _blob_id:  # adj_blob of adj_blob could be the blob itself
                # pairing function generates unique number from each pair of comparands, frame-wide:
                derBlob_id = (0.5 * (blob.id + adj_blob.id) * (blob.id + adj_blob.id + 1) + (blob.id * adj_blob.id))
                # if derBlob exists, it may be that derBlob.blob==blob, derBlob._blob==adj_blob, or derBlob.blob=adj_blob, derBlob._blob=blob
                if derBlob_id in derBlob_id_:  # derBlob exists, just accumulate it in blob.DerBlob
                    derBlob = derBlob_[derBlob_id_.index(derBlob_id)]
                    accum_derBlob(blob, derBlob)  # also adj_blob.rdn += 1?
                else:  # compute new derBlob
                    derBlob = comp_blob(blob, adj_blob)  # compare blob and adjacent blob
                    accum_derBlob(blob, derBlob)         # from all compared blobs, regardless of mB sign
                    derBlob_id_.append(derBlob_id)       # unique comparand_pair identifier
                    derBlob_.append(derBlob)             # also frame-wide
                if derBlob.mB > 0:
                    # replace blob with adj_blob for continuing adjacency search:
                    if not isinstance(adj_blob.DerBlob, CderBlob):  # if adj_blob.DerBlob: it's already searched in previous call,
                        adj_blob.DerBlob = CderBlob()  # but this search could be of different depth, so compare again:
                    comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], blob_id_, derBlob_, derBlob_id_)
                    break
                elif blob.Dert.M + blob.neg_mB + derBlob.mB > ave_mB:  # neg mB but positive comb M,
                    # extend blob comparison to adjacents of adjacent, depth-first
                    blob.neg_mB += derBlob.mB  # mB and distance are accumulated over comparison scope
                    blob.distance += np.sqrt(adj_blob.A)
                    comp_blob_recursive(blob, adj_blob.adj_blobs[0], blob_id_, derBlob_, derBlob_id_)
     '''

def comp_blob(blob, _blob):
    '''
    cross compare _blob and blob
    '''
    (_I, _Dy, _Dx, _G, _M, _Dyy, _Dyx, _Dxy, _Dxx, _Ga, _Ma, _Mdx, _Ddx), _A = _blob.Dert.unpack(), _blob.A
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, Mdx, Ddx), A = blob.Dert.unpack(), blob.A

    dI = _I - I  # d is always signed
    mI = min(_I, I)
    dA = _A - A
    mA = min(_A, A)
    dG = _G - G
    mG = min(_G, G)
    dM = _M - M
    mM = min(_M, M)

    mB = mI + mA + mG + mM - ave_mB * (ave_rM ** ((1+blob.distance) / np.sqrt(A)))
    # deviation from average blob match at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=blob, _blob=_blob, mB=mB, dB=dB)  # blob is core node, _blob is adjacent blob

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_):
    '''
    bblob is a cluster / graph of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    bblob_ = []
    for blob in blob_:
        if blob.DerBlob.mB > 0:  # init bblob with current blob

            bblob = CBblob(DerBlob=CderBlob())
            accum_bblob(bblob, blob)  # accum blob into bblob
            form_bblob_recursive(bblob_, bblob, bblob.blob_)

    # for debug purpose on duplicated blobs in bblob, not needed in actual code
    for bblob in bblob_:
        bblob_blob_id_ = [ blob.id for blob in bblob.blob_]
        if len(bblob_blob_id_) != len(np.unique(bblob_blob_id_)):
            raise ValueError("Duplicated blobs")

    return bblob_

def form_bblob_recursive(bblob_, bblob, blob_):

    blobs2check = []  # blobs to check for inclusion in bblob

    for blob in blob_:  # search new added blobs to get potential border clustering blob
        if (blob.DerBlob.mB > 0):  # positive mB
            for derBlob in blob.derBlob_:
                # blob is in bblob.blob_, but derBlob._blob is not in bblob_blob_ and (DerBlob.mB > 0 and blob.mB > 0):
                if (derBlob._blob not in bblob.blob_) and (derBlob._blob.DerBlob.mB + blob.DerBlob.mB > 0):
                    accum_bblob(bblob, derBlob._blob)  # pack derBlob._blob in bblob
                    blobs2check.append(derBlob._blob)
                elif (derBlob.blob not in bblob.blob_) and (derBlob.blob.DerBlob.mB + blob.DerBlob.mB > 0):
                    accum_bblob(bblob, derBlob.blob)
                    blobs2check.append(derBlob.blob)

    if blobs2check:
        form_bblob_recursive(bblob_, bblob, blobs2check)

    bblob_.append(bblob)  # pack bblob after scanning all accessible derBlobs


def accum_derBlob(blob, derBlob):

    blob.DerBlob.accumulate(**{param:getattr(derBlob, param) for param in blob.DerBlob.numeric_params})
    blob.derBlob_.append(derBlob)

def accum_bblob(bblob, blob):

    # accumulate derBlob
    bblob.DerBlob.accumulate(**{param:getattr(blob.DerBlob, param) for param in bblob.DerBlob.numeric_params})

    bblob.blob_.append(blob)

    for derBlob in blob.derBlob_: # pack adjacent blobs of blob into bblob
        if derBlob._blob not in bblob.blob_:
            bblob.blob_.append(derBlob._blob)

'''
    cross-comp among sub_blobs in nested sub_layers:
    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)
        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''


def visualize_cluster_(bblob_, blob_, frame):

    colour_list = []  # list of colours:
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([192, 192, 192])  # silver
    colour_list.append([255, 255, 0])  # blue2
    colour_list.append([34, 34, 178])  # red2
    colour_list.append([0, 215, 255])  # yellow2
    colour_list.append([50, 205, 50])  # green2
    colour_list.append([114, 128, 250])  # pink2
    colour_list.append([255, 255, 224])  # cyan2
    colour_list.append([0, 140, 255])  # orange2
    colour_list.append([204, 50, 153])  # purple2
    colour_list.append([63, 133, 205])  # brown2
    colour_list.append([128, 128, 128])  # silver2

    # initialization
    ysize, xsize = frame.dert__[0].shape
    blob_img = np.zeros((ysize, xsize,3)).astype('uint8')
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    img_separator = np.zeros((ysize,3,3)).astype('uint8')

    # create mask
    blob_mask = np.zeros_like(frame.dert__[0]).astype('uint8')
    cluster_mask = np.zeros_like(frame.dert__[0]).astype('uint8')

    blob_colour_index = 1
    cluster_colour_index = 1

    cv2.namedWindow('blobs & clusters', cv2.WINDOW_NORMAL)

    # draw blobs
    for blob in blob_:
        cy0, cyn, cx0, cxn = blob.box
        blob_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ * blob_colour_index).astype('uint8')
        blob_colour_index += 1

    # draw blob cluster of bblob
    for bblob in bblob_:
        for blob in bblob.blob_:
            cy0, cyn, cx0, cxn = blob.box
            blob_colour_index += 1
            cluster_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ *cluster_colour_index).astype('uint8')

        # increase colour index
        cluster_colour_index += 1

        # insert colour of blobs
        for i in range(1,blob_colour_index):
            blob_img[np.where(blob_mask == i)] = colour_list[i % 10]

        # insert colour of clusters
        for i in range(1,cluster_colour_index):
            cluster_img[np.where(cluster_mask == i)] = colour_list[i % 10]

        # combine images for visualization
        img_concat = np.concatenate((blob_img, img_separator,
                                    cluster_img, img_separator), axis=1)

        # plot cluster of blob
        cv2.imshow('blobs & clusters',img_concat)
        cv2.resizeWindow('blobs & clusters', 1920, 720)
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_unique_id(id1, id2):
    '''
    generate unique id based on id1 and id2, different order of id1 and id2 yields unique id in different sign
    '''
    # generate unique id with sign
    '''
    get sign based on order of id1 and id2, output would be +1 or -1
    id_sign = ((0.5*(id1+id2)*(id1+id2+1) + id1) - (0.5*(id2+id1)*(id2+id1+1) + id2)) / abs(id1-id2)
    modified pairing function, so that different order of a and b will generate same value
    unique_id = (0.5*(id1+id2)*(id1+id2+1) + (id1*id2)) * id_sign
    '''
    # generate unique id without sign
    unique_id = (0.5 * (id1 + id2) * (id1 + id2 + 1) + (id1 * id2))

    return unique_id