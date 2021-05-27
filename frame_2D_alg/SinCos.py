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
    Frame_blobs is a root function for all deeper processing in 2D alg.
    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/blob_params.drawio
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

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Sin, Cos, G, M, blob_, dert__')


class CDert(ClusterStructure):
    # Dert params, comp_pixel:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # comp_angle:
    Day = complex
    Dax = complex
    Ga = int
    Ma = int
    # comp_dx:
    Mdx = int
    Ddx = int

class CFlatBlob(ClusterStructure):  # from frame_blobs only, no sub_blobs

    # Dert params, comp_pixel:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # Dert params, comp_angle:
    Day = complex
    Dax = complex
    Ga = int
    Ma = int
    # Dert params, comp_dx:
    Mdx = int
    Ddx = int
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
    I = int
    Sin = int
    Cos = int
    G = int
    M = int
    # Dert params, comp_angle:
    Day = complex
    Dax = complex
    Ga = int
    Ma = int
    # Dert params, comp_dx:
    Mdx = int
    Ddx = int

    # blob params:
    A = int  # blob area
    sign = NoneType
    box = list
    mask__ = bool
    dert__ = tuple  # 2D array per param
    root_dert__ = tuple
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
    mB = int
    dB = int
    derBlob_ = list
    distance = int  # common per derBlob_
    neg_mB = int    # common per derBlob_
    bblob = object


def comp_pixel(image):  # 2x2 pixel cross-correlation within image, a standard edge detection operator
    # see comp_pixel_versions file for other versions and more explanation

    # input slices into sliding 2x2 kernel, each slice is a shifted 2D frame of grey-scale pixels:
    topleft__ = image[:-1, :-1]
    topright__ = image[:-1, 1:]
    bottomleft__ = image[1:, :-1]
    bottomright__ = image[1:, 1:]

    rot_Gy__ = bottomright__ - topleft__  # rotated to bottom__ - top__
    rot_Gx__ = topright__ - bottomleft__  # rotated to right__ - left__

    G__ = np.hypot(rot_Gy__, rot_Gx__)  # central gradient per kernel, between four vertex pixels
    G__[G__ == 0] = 1  # when G=0 , set to 1
    sin__, cos__ = (rot_Gy__, rot_Gx__) / G__
    vG__ = (G__ - ave).astype('int')  # deviation of gradient

    M__ = int(ave * 1.2) - (abs(rot_Gy__) + abs(rot_Gx__))  # inverse deviation of SAD (variation), ave * (ave_SAD / ave_G): 1.2?

    return (topleft__, sin__, cos__, vG__, M__)  # tuple of 2D arrays per param of dert (derivatives tuple), topleft__ is off by .5 pixels
    # renamed dert__ = (p__, dy__, dx__, g__, m__) for readability in functions below
'''
    rotate dert__ 45 degrees clockwise, convert diagonals into orthogonals to avoid summation, which degrades accuracy of Gy, Gx
    Gy, Gx are used in comp_a, which returns them (as well as day, dax) back to orthogonal orientation
    
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
        I, Sin, Cos, G, M = 0, 0, 0, 0, 0
        for blob in blob_:
            I += blob.I
            Sin += blob.Sin
            Cos += blob.Cos
            G += blob.G
            M += blob.M
        frame = FrameOfBlobs(I=I, Sin=Sin, Cos=Cos, G=G, M=M, blob_=blob_, dert__=dert__)

    assign_adjacents(adj_pairs)  # f_segment_by_direction=False

    if verbose: print(f"{len(frame.blob_)} blobs formed in {time() - start_time} seconds")
    if render: visualize_blobs(idmap, frame.blob_)

    return frame


def flood_fill(dert__, sign__, verbose=False, mask__=None, blob_cls=CBlob, fseg=False, prior_forks=[]):

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
                blob = blob_cls(sign=sign__[y, x], root_dert__=dert__)
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
                    blob.accumulate(I =dert__[0][y][x],
                                    Sin=dert__[1][y][x],
                                    Cos=dert__[2][y][x],
                                    G =dert__[3][y][x],
                                    M =dert__[4][y][x])

                    if len(dert__)>5: # comp_angle
                        blob.accumulate(Day =dert__[5][y][x],
                                        Dax =dert__[6][y][x],
                                        Ga  =dert__[7][y][x],
                                        Ma  =dert__[8][y][x])
                    if len(dert__)>10: # comp_dx
                        blob.accumulate(Mdx =dert__[9][y][x],
                                        Ddx =dert__[10][y][x])

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
    verbose = args.verbose
    intra = args.intra
    render = args.render

    start_time = time()
    dert__ = comp_pixel(image)
    frame = derts2blobs(dert__, verbose=args.verbose, render=args.render, use_c=args.clib)

    if intra:  # call to intra_blob, omit for testing frame_blobs only:

        if args.verbose: print("\rRunning intra_blob...")
        from intra_blob import intra_blob, aveB

        deep_frame = frame, frame  # 1st frame initializes summed representation of hierarchy, 2nd is individual top layer
        deep_blob_i_ = []  # index of a blob with deep layers
        deep_layers = [[]] * len(frame.blob_)  # for visibility only
        empty = np.zeros_like(frame.dert__[0])
        root_dert__ = (  # update root dert__
            frame.dert__[0],  # i
            frame.dert__[1],  # sin
            frame.dert__[2],  # cos
            frame.dert__[3],  # g
            frame.dert__[4],  # m
            )

        for i, blob in enumerate(frame.blob_):  # print('Processing blob number ' + str(bcount))
            '''
            Blob G: -|+ predictive value, positive value of -G blobs is lent to the value of their adjacent +G blobs. 
            +G "edge" blobs are low-match, valuable only as contrast: to the extent that their negative value cancels 
            positive value of adjacent -G "flat" blobs.
            '''
            G = blob.vG
            M = blob.M
            blob.root_dert__=root_dert__
            blob.prior_forks=['g']
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
    Blob structure, for all layers of blob hierarchy, see class CBlob:
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

            if (-blob.M * blob.Ma - AveB > 0):  # vs. G reduced by Ga: * (1 - Ga / (4.45 * A)), max_ga=4.45
                blob.f_comp_a = 0
                blob.prior_forks.extend('p')
                if kwargs.get('verbose'): print('\nslice_blob fork\n')
                segment_by_direction(blob, verbose=True)
                # derP_ = slice_blob(blob, [])  # cross-comp of vertically consecutive Ps in selected stacks
                # blob.PP_ = derP_2_PP_(derP_, blob.PP_)  # form vertically contiguous patterns of patterns
    else:
        # root fork is frame_blobs or comp_r
        ext_dert__, ext_mask__ = extend_dert(blob)  # dert__ boundaries += 1, for cross-comp in larger kernels

        if blob.G > AveB:  # comp_a fork, replace G with borrow_M when known

            adert__, mask__ = comp_a(ext_dert__, Ave, blob.prior_forks, ext_mask__)  # compute ma and ga
            blob.f_comp_a = 1
            if kwargs.get('verbose'): print('\na fork\n')
            blob.prior_forks.extend('a')

            if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, at least one dert in dert__
                sign__ = adert__[3] * adert__[8] > 0   # g * ma: not co-measurable with g
                dert__ = adert__  # already flat, adert__[[5, 6]] are now complex

                cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs)  # forms sub_blobs of sign in unmasked area
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.M > AveB * 1.2:  # comp_r fork, ave M = ave G * 1.2

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

    sub_blobs, idmap, adj_pairs = flood_fill(dert__, sign__, verbose=False, mask__=mask__, blob_cls=CBlob)
    assign_adjacents(adj_pairs, CBlob)

    if kwargs.get('render', False):
        visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (f_comp_a = {blob.f_comp_a}, f_root_a = {blob.f_root_a})")

    blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
    blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

    for sub_blob in sub_blobs:  # evaluate sub_blob
        '''
        adj_M = blob.adj_blobs[3]  # adj_M is incomplete, computed within current dert_only, use root blobs instead:
        adjacent valuable blobs of any sign are tracked from frame_blobs to form borrow_M?
        track adjacency of sub_blobs: wrong sub-type but right macro-type: flat blobs of greater range?
        G indicates or dert__ extend per blob G?
        borrow_M = min(G, adj_M / 2)
        '''
        sub_blob.prior_forks = blob.prior_forks.copy()  # increments forking sequence: g->a, g->a->p, etc.

        if sub_blob.G > AveB:  # G = blob.G  # Gr, Grr..' borrow_M, replace with known borrow_M if any
            # comp_a:
            sub_blob.f_root_a = 1
            sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub_blob, currently not used
            sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
            blob.sub_layers += intra_blob(sub_blob, **kwargs)

        elif sub_blob.M - aveG > AveB:
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


"""
Cross-comparison of pixels 3x3 kernels or gradient angles in 2x2 kernels
"""

import numpy as np
import functools

''' 
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
        majority_mask__ = (mask__[1:-1:2, 1:-1:2].astype(int)
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

    # i think this shouldn't be needed now, if root_fia, it will not be reaching comp_r, root_fia is always 0 here
    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        sin__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
        cos__ = dert__[2][1:-1:2, 1:-1:2].copy()
        m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs:

    dy__ = ((i__topleft - i__bottomright) * -1 +
            (i__top - i__bottom) * -2 +
            (i__topright - i__bottomleft) * -1 +
            (i__right - i__left) * 0)

    dx__ = ((i__topleft - i__bottomright) * -1 +
            (i__top - i__bottom) * 0 +
            (i__topright - i__bottomleft) * 1 +
            (i__right - i__left) * 2)

    g__ = np.hypot(dy__, dx__)  # gradient, recomputed at each comp_r
    g__[g__ == 0] = 1  # when g =0 , set g = 1

    sin__ += dy__ / g__  # accumulate sin
    cos__ += dx__ / g__  # accumulate cos

    vg__ = g__ - ave  # deviation of gradient
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    ave SAD = ave g * 1.2:
    '''
    m__ += int(ave * 1.2) - (abs(i__center - i__topleft)
                             + abs(i__center - i__top) * 2
                             + abs(i__center - i__topright)
                             + abs(i__center - i__right) * 2
                             + abs(i__center - i__bottomright)
                             + abs(i__center - i__bottom) * 2
                             + abs(i__center - i__bottomleft)
                             + abs(i__center - i__left) * 2
                             )

    return (i__center, sin__, cos__, vg__, m__), majority_mask__


def comp_a(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels
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

    i__, sin__, cos__, vg__, m__ = dert__[:5]  # day__,dax__,ga__,ma__ are recomputed

    # az__ = dx__ + 1j * dy__  # take the complex number (z), phase angle is now atan2(dy, dx)
    az__ = cos__ + 1j * sin__  # not so sure yet

    with np.errstate(divide='ignore', invalid='ignore'):  # suppress numpy RuntimeWarning
        az__ /= np.absolute(az__)  # normalized, cosine = a__.real, sine = a__.imag

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

    # sin and cos of a?
    sin_ga__ = dax__ / ga__
    cos_ga__ = day__ / ga__

    if (prior_forks[-1] == 'g') or (prior_forks[-1] == 'a'):  # root fork is frame_blobs, recompute orthogonal sin and cos
        i__topleft = i__[:-1, :-1]
        i__topright = i__[:-1, 1:]
        i__botright = i__[1:, 1:]
        i__botleft = i__[1:, :-1]
        dy__ = (i__botleft + i__botright) - (i__topleft + i__topright)  # decomposition of two diagonal differences
        dx__ = (i__topright + i__botright) - (i__topleft + i__botleft)  # decomposition of two diagonal differences
        g__ = np.hypot(dy__, dx__)
        g__[g__ == 0] = 1  # when g =0 , set g = 1
        sin__ = dy__ / g__  # recomputed sin__
        cos__ = dx__ / g__  # recomputed cos__

    else:
        sin__ = sin__[:-1, :-1]  # passed on as idy, not rotated
        cos__ = cos__[:-1, :-1]  # passed on as idx, not rotated

    i__ = i__[:-1, :-1]  # for summation in Dert
    vg__ = vg__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]

    return (i__, sin__, cos__, vg__, m__, sin_ga__, cos_ga__, ga__, ma__), majority_mask__  # dazx__, dazy__ may not be needed


def angle_diff(az2, az1):  # unpacked in comp_a
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


'''
old version:
'''


def comp_a_simple(dert__, ave, prior_forks, mask__=None):  # cross-comp of gradient angle in 2x2 kernels

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
    a__left = a__[:, :-1, :-1]  # was topleft
    a__top = a__[:, :-1, 1:]  # was topright
    a__right = a__[:, 1:, 1:]  # was botright
    a__bottom = a__[:, 1:, :-1]  # was botleft

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
    ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__)) - 2
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
        dy_ = dy__[:-1, :-1]  # passed on as idy, not rotated
        dx__ = dx__[:-1, :-1]  # passed on as idx, not rotated

    i__ = i__[:-1, :-1]  # for summation in Dert
    g__ = g__[:-1, :-1]  # for summation in Dert
    m__ = m__[:-1, :-1]

    return (i__, sin__, cos__, g__, m__, day__, dax__, ga__, ma__), majority_mask__


def angle_diff_simple(a2, a1):  # compare angle_1 to angle_2

    sin_1, cos_1 = a1[:]
    sin_2, cos_2 = a2[:]

    # sine and cosine of difference between angles:

    sin_da = (cos_1 * sin_2) - (sin_1 * cos_2)
    cos_da = (cos_1 * cos_2) + (sin_1 * sin_2)

    return [sin_da, cos_da]


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
'''

from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
# import warnings  # to detect overflow issue, in case of infinity loop
# warnings.filterwarnings('error')

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_dX = 10  # difference between median x coords of consecutive Ps
ave_Cos = 10
ave_mP = 8  # just a random number right now.
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
ave_da = 1  # da at 45 degree?
# comp_PP
ave_mPP = 0
ave_rM  = .7

'''
CP should be a nested class, including derP, possibly multi-layer:
if PP: CP contains P, 
   each param contains summed values of: param, (m,d),
   and each dert in dert_ is actually P
if PPP: CP contains P which also contains P
   each param contains summed values of: param, (m,d), ((mm,dm), (md,dd)) 
   and each dert in dert_ is actually PP
'''

class CP(ClusterStructure):

    # Dert params, comp_pixel:
    I = int
    Sin = int
    Cos = int
    G = int
    M = int
    # Dert params, comp_angle:
    Sin_ga = int
    Cos_ga = int
    Ga = int
    Ma = int
    # Dert params, comp_dx:
    Mdx = int
    Ddx = int

    L = int
    x0 = int
    x = int  # median x
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

class CderP(ClusterStructure):

    # derP params
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
    mDay = complex
    mDax = complex
    mGa = int
    mMa = int
    mMdx = int
    mDdx = int
    dDay = complex
    dDax = complex
    dGa = int
    dMa = int
    dMdx = int
    dDdx = int

    P = object   # lower comparand
    _P = object  # higher comparand
    PP = object  # FPP if flip_val, contains this derP
    # from comp_dx
    fdx = NoneType


class CderPP(ClusterStructure):

    PP = object
    _PP = object
    mmPP = int
    dmPP = int
    mdPP = int
    ddPP = int

class CPP(ClusterStructure):

    # Dert params, comp_pixel:
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # Dert params, comp_angle:
    Sin_ga = int
    Cos_ga = int
    Ga = int
    Ma = int
    # Dert params, comp_dx:
    Mdx = int
    Ddx = int

    # derP params
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
    mDay = complex
    mDax = complex
    mGa = int
    mMa = int
    mMdx = int
    mDdx = int
    dDay = complex
    dDax = complex
    dGa = int
    dMa = int
    dMdx = int
    dDdx = int

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

    # comp_PP
    derPPm_ = []
    derPPd_ = []
    distance = int
    mmPP = int
    dmPP = int
    mdPP = int
    ddPP = int
    neg_mmPP = int
    neg_mdPP = int

    PPPm = object
    PPPd = object

class CPPP(ClusterStructure):

    PPm_ = list
    PPd_ = list
    mmPP = int
    dmPP = int
    mdPP = int
    ddPP = int

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

        comp_PP_(blob,fPPd)

        # yet to be updated
        # draw PPs
        #    if not isinstance(blob, CPP):
        #        draw_PP_(blob)

def form_P_(idert_, mask_, y):  # segment dert__ into P__ in horizontal ) vertical order, sum dert params into P params

    P_ = []                # rows of derPs
    _dert = list(idert_[0]) # first dert
    dert_ = [_dert]         # pack 1st dert
    _mask = mask_[0]       # mask bit per dert

    if ~_mask:
        # initialize P with first dert
        P = CP(I=_dert[0], Sin=_dert[1], Cos=_dert[2], G=_dert[3], M=_dert[4], Sin_ga=_dert[5], Cos_ga=_dert[6], Ga=_dert[7], Ma=_dert[8], x0=0, L=1, y=y, dert_=dert_)

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # pixel mask

        if mask:  # masks: if 1,_0: P termination, if 0,_1: P initialization, if 0,_0: P accumulation:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P.x = P.x0 + (P.L-1) // 2
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                # initialize P with first dert
                P = CP(I=dert[0], Sin=dert[1], Cos=dert[2], G=dert[3], M=dert[4], Sin_ga=dert[5], Cos_ga=dert[6], Ga=dert[7], Ma=dert[8],
                       x0=x, L=1, y=y, dert_=dert_)
            else:
                # _dert is not masked, accumulate P params with (p, dy, dx, g, m, day, dax, ga, ma) = dert
                P.accumulate(I=dert[0], Sin=dert[1], Cos=dert[2], G=dert[3], M=dert[4], Sin_ga=dert[5], Cos_ga=dert[6], Ga=dert[7], Ma=dert[8], L=1)
                P.dert_.append(dert)

        _mask = mask

    if ~_mask:  # terminate last P in a row
        P.x = P.x0 + (P.L-1) // 2
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
            _sign = _dert[2] > 0
            # initialize P with first dert
            P = CP(I=_dert[0], Sin=_dert[1], Cos=_dert[2], G=_dert[3], M=_dert[4], Sin_ga=_dert[5], Cos_ga=_dert[6], Ga=_dert[7], Ma=_dert[8],
                   x0=iP.x0, dert_=dert_, L=1, y=iP.y, sign=_sign, Pm=iP)
            x = 1  # relative x within P

            for dert in iP.dert_[1:]:
                sign = dert[2] > 0
                if sign == _sign: # same Dx sign
                    # accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                    P.accumulate(I=dert[0], Sin=dert[1], Cos=dert[2], G=dert[3], M=dert[4], Sin_ga=dert[5], Cos_ga=dert[6], Ga=dert[7], Ma=dert[8],L=1)
                    P.dert_.append(dert)

                else:  # sign change, terminate P
                    if P.Cos > ave_Cos:
                        # cross-comp of dx in P.dert_
                        comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
                    P.x = P.x0 + (P.L-1) // 2
                    Pd_.append(P)
                    # reinitialize params
                    P = CP(I=dert[0], Sin=dert[1], Cos=dert[2], G=dert[3], M=dert[4], Sin_ga=dert[5], Cos_ga=dert[6], Ga=dert[7], Ma=dert[8],
                           x0=iP.x0+x, dert_=[dert], L=1, y=iP.y, sign=sign, Pm=iP)
                _sign = sign
                x += 1
            # terminate last P
            if P.Cos > ave_Cos:
                comp_dx(P); P_Ddx += P.Ddx; P_Mdx += P.Mdx
            P.x = P.x0 + (P.L-1) // 2
            Pd_.append(P)
            # update Pd params in P
            iP.Pd_ = Pd_; iP.Ddx = P_Ddx; iP.Mdx = P_Mdx
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
                    derP = comp_slice(_P, P)  # form vertical and directional derivatives
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
                            derPd = comp_slice(_Pd, Pd)
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
        derP_2_PP_(blob.derP__, blob.PPdm_,  1)   # cluster by derPm dP sign
        derP_2_PP_(blob.derPd__, blob.PPdd_,  1)  # cluster by derPd dP sign, not used
    else:
        derP_2_PP_(blob.derP__, blob.PPmm_, 0)   # cluster by derPm mP sign
        derP_2_PP_(blob.derPd__, blob.PPmd_, 0)  # cluster by derPd mP sign, not used


def derP_2_PP_(derP_, PP_,  fPPd):
    '''
    first row of derP_ has downconnect_cnt == 0, higher rows may also have them
    '''
    for derP in reversed(derP_):  # bottom-up to follow upconnects, derP is stored top-down
        if not derP.P.downconnect_cnt and not isinstance(derP.PP, CPP):  # root derP was not terminated in prior call
            PP = CPP()  # init
            accum_PP(PP,derP)

            if derP._P.upconnect_:  # derP has upconnects
                upconnect_2_PP_(derP, PP_, fPPd)  # form PPs across _P upconnects
            else:
                PP_.append(derP.PP)


def upconnect_2_PP_(iderP, PP_,  fPPd):
    '''
    compare sign of lower-layer iderP to the sign of its upconnects to form contiguous same-sign PPs
    '''
    confirmed_upconnect_ = []

    for derP in iderP._P.upconnect_:  # potential upconnects from previous call
        if derP not in iderP.PP.derP__:  # derP should not in current iPP derP_ list, but this may occur after the PP merging

            if fPPd: same_sign = (iderP.dP > 0) == (derP.dP > 0)  # comp dP sign
            else: same_sign = (iderP.mP > 0) == (derP.mP > 0)  # comp mP sign

            if same_sign:  # upconnect derP has different PP, merge them
                if isinstance(derP.PP, CPP) and (derP.PP is not iderP.PP):
                    merge_PP(iderP.PP, derP.PP, PP_)
                else:  # accumulate derP in current PP
                    accum_PP(iderP.PP, derP)
                    confirmed_upconnect_.append(derP)
            else:
                if not isinstance(derP.PP, CPP):  # sign changed, derP is root derP unless it already has FPP/PP
                    PP = CPP()
                    accum_PP(PP,derP)
                    derP.P.downconnect_cnt = 0  # reset downconnect count for root derP

                iderP.PP.upconnect_.append(derP.PP) # add new initialized PP as upconnect of current PP
                derP.PP.downconnect_cnt += 1        # add downconnect count to newly initialized PP

            if derP._P.upconnect_:
                upconnect_2_PP_(derP, PP_, fPPd)  # recursive compare sign of next-layer upconnects

            elif derP.PP is not iderP.PP and derP.P.downconnect_cnt == 0:
                PP_.append(derP.PP)  # terminate PP (not iPP) at the sign change

    iderP._P.upconnect_ = confirmed_upconnect_

    if not iderP.P.downconnect_cnt:
        PP_.append(iderP.PP)  # iPP is terminated after all upconnects are checked


def merge_PP(_PP, PP, PP_):  # merge PP into _PP

    for derP in PP.derP__:
        if derP not in _PP.derP__:
            _PP.derP__.append(derP) # add derP to PP
            derP.PP = _PP           # update reference
            _PP.accum_from(derP)    # accumulate params
    if PP in PP_:
        PP_.remove(PP)  # remove merged PP



def accum_PP(PP, derP):  # accumulate params in PP

    PP.accum_from(derP)    # accumulate params
    PP.derP__.append(derP) # add derP to PP
    derP.PP = PP           # update reference


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
    P.Ddx = Ddx
    P.Mdx = Mdx


def comp_slice(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    x, M, L =  P.x, P.M, P.L # params per comp branch
    _x, _M, _dX, _L =  _P.x, _P.M, _P.dX, _P.L

    dX = x - _x  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?
    hyp = np.hypot(dX, 1)  # ratio of local segment of long (vertical) axis to dY = 1

    L /= hyp  # orthogonal L is reduced by hyp
    dL = L - _L; mL = min(L, _L)  # L: positions / sign, dderived: magnitude-proportional value
    M /= hyp  # orthogonal M is reduced by hyp
    dM = M - _M; mM = min(M, _M)  # use abs M?  no Mx, My: non-core, lesser and redundant bias?

    sin_da = (P.Cos * _P.Sin) - (P.Sin * _P.Cos)   # using formula : sin(α − β) = sin α cos β − cos α sin β
    cos_da = (P.Cos * _P.Cos) + (P.Sin * _P.Sin)   # using formula : cos(α − β) = cos α cos β + sin α sin β
    da = np.arctan2( sin_da, cos_da )
    ma = ave_da - abs(da)

    dP = dL + dM + da  # -> directional PPd, equal-weight params, no rdn?
    mP = mL + mM + ma  # -> complementary PPm, rdn *= Pd | Pm rolp?
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    derP = CderP(mP=mP, dP=dP, dX=dX, mL=mL, dL=dL, P=P, _P=_P)
    P.derP = derP

    return derP


def comp_slice_full(_P, P):  # forms vertical derivatives of derP params, and conditional ders from norm and DIV comp

    x0, Dx, Dy, L, = P.x0, P.Dx, P.Dy, P.L
    # params per comp branch, add angle params
    _x0, _Dx, _Dy,_dX, _L = _P.x0, _P.Dx, _P.Dy, _P.dX, _P.L

    dX = (x0 + (L-1) / 2) - (_x0 + (_L-1) / 2)  # x shift: d_ave_x, or from offsets: abs(x0 - _x0) + abs(xn - _xn)?

    if dX > ave_dX:  # internal comp is higher-power, else two-input comp not compressive?
        xn = x0 + L - 1
        _xn = _x0 + _L - 1
        mX = min(xn, _xn) - max(x0, _x0)  # overlap = abs proximity: summed binary x match
        rX = dX / mX if mX else dX*2  # average dist / prox, | prox / dist, | mX / max_L?

    ddX = dX - _dX  # long axis curvature, if > ave: ortho eval per P, else per PP_dX?
    mdX = min(dX, _dX)  # dX is inversely predictive of mP?

    # is this looks better? or it would better if we stick to the old code?
    difference = P.difference(_P)   # P - _P
    match = P.min_match(_P)         # min of P and _P
    abs_match = P.abs_min_match(_P) # min of abs(P) and abs(_P)

    dL = difference['L'] # L: positions / sign, dderived: magnitude-proportional value
    mL = match['L']
    dM = difference['M'] # use abs M?  no Mx, My: non-core, lesser and redundant bias?
    mM = match['M']

    # pending update
    # min is value distance for opposite-sign comparands, vs. value overlap for same-sign comparands
    dDy = difference['Dy']  # Dy per sub_P by intra_comp(dx), vs. less vertically specific dI
    mDy = abs_match['Dy']
    # no comp G: Dy, Dx are more specific:
    dDx = difference['Dx']  # same-sign Dx if Pd
    mDx = abs_match['Dx']

    if dX * P.G > ave_ortho:  # estimate params of P locally orthogonal to long axis, maximizing lateral diff and vertical match
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
        # recompute difference and match
        dL = _L - L
        mL = min(_L, L)
        dDy = _Dy - Dy
        mDy = min(abs(_Dy), abs(Dy))
        dDx = _Dx - Dx
        mDx = min(abs(_Dx), abs(Dx))

    if (Dx > 0) != (_Dx > 0): mDx = -mDx
    if (Dy > 0) != (_Dy > 0): mDy = -mDy

    dDdx, dMdx, mDdx, mMdx = 0, 0, 0, 0
    if P.dxdert_ and _P.dxdert_:  # from comp_dx
        fdx = 1
        dDdx = difference['Ddx']
        mDdx = abs_match['Ddx']
        if (P.Ddx > 0) != (_P.Ddx > 0): mDdx = -mDdx
        # Mdx is signed:
        dMdx = match['Mdx']
        mMdx = -abs_match['Mdx']
        if (P.Mdx > 0) != (_P.Mdx > 0): mMdx = -mMdx
    else:
        fdx = 0


    da = np.arctan2(difference['Sin'], difference['Cos'] )
    ma = ave_da - abs(da)

    # coeff = 0.7 for semi redundant parameters, 0.5 for fully redundant parameters:
    dP = da + ddX + dL + 0.7*(dM + dDx + dDy)  # -> directional PPd, equal-weight params, no rdn?
    # correlation: dX -> L, oDy, !oDx, ddX -> dL, odDy ! odDx? dL -> dDx, dDy?
    if fdx: dP += 0.7*(dDdx + dMdx)

    mP = ma + mdX + mL + 0.7*(mM + mDx + mDy)  # -> complementary PPm, rdn *= Pd | Pm rolp?
    if fdx: mP += 0.7*(mDdx + mMdx)
    mP -= ave_mP * ave_rmP ** (dX / L)  # dX / L is relative x-distance between P and _P,

    derP = CderP(P=P, _P=_P, mP=mP, dP=dP, dX=dX, mL=mL, dL=dL, mDx=mDx, dDx=dDx, mDy=mDy, dDy=dDy)
    P.derP = derP

    if fdx:
        derP.fdx=1; derP.dDdx=dDdx; derP.mDdx=mDdx; derP.dMdx=dMdx; derP.mMdx=mMdx

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


# draft of comp_PP, following structure of comp_blob
def comp_PP_(blob, fPPd):

    for fPd in [0,1]:
        if fPPd: # cluster by d sign
            if fPd: # using derPd (PPdd)
                PP_ = blob.PPdd_
            else: # using derPm (PPdm)
                PP_ = blob.PPdm_
            for PP in PP_:
                if len(PP.derPPd_) == 0: # PP doesn't perform any searching in prior function call
                    comp_PP_recursive(PP, PP.upconnect_, derPP_=[], fPPd=fPPd)

            form_PPP_(PP_, fPPd)

        else: # cluster by m sign
            if fPd: # using derPd (PPmd)
                PP_ = blob.PPmd_
            else: # using derPm (PPmm)
                PP_ = blob.PPmm_
            for PP in PP_:
                if len(PP.derPPm_) == 0: # PP doesn't perform any searching in prior function call
                    comp_PP_recursive(PP, PP.upconnect_, derPP_=[], fPPd=fPPd)

            form_PPP_(PP_, fPPd)

def comp_PP_recursive(PP, upconnect_, derPP_, fPPd):

    derPP_pair_ = [ [derPP.PP, derPP._PP]  for derPP in derPP_]

    for _PP in upconnect_:
        if [_PP, PP] in derPP_pair_ : # derPP.PP = _PP, derPP._PP = PP
            derPP = derPP_[derPP_pair_.index([_PP,PP])]

        elif [PP, _PP] not in derPP_pair_ : # same pair of PP and _PP doesn't checked prior this function call
            derPP = comp_PP(PP, _PP) # comp_PP
            derPP_.append(derPP)

        if "derPP" in locals(): # derPP exists
            accum_derPP(PP, derPP, fPPd)    # accumulate derPP
            if fPPd:                  # PP cluster by d
                mPP = derPP.mdPP      # match of PPs' d
            else:                     # PP cluster by m
                mPP = derPP.mmPP      # match of PPs' m

            if mPP>0: # _PP replace PP to continue the searching
                comp_PP_recursive(_PP, _PP.upconnect_, derPP_, fPPd)

            elif fPPd and PP.neg_mdPP + PP.mdPP > ave_mPP: # evaluation to extend PPd comparison
                PP.distance += len(_PP.Pd__) # approximate using number of Py, not so sure
                PP.neg_mdPP += derPP.mdPP
                comp_PP_recursive(PP, _PP.upconnect_, derPP_, fPPd)

            elif not fPPd and PP.neg_mmPP + PP.mmPP > ave_mPP: # evaluation to extend PPm comparison
                PP.distance += len(_PP.P__) # approximate using number of Py, not so sure
                PP.neg_mmPP += derPP.mmPP
                comp_PP_recursive(PP, _PP.upconnect_, derPP_, fPPd)

# draft
def form_PPP_(PP_, fPPd):

    PPP_ = []
    for PP in PP_:

        if fPPd:
            mPP = PP.mdPP # match of PP's d
            PPP = PP.PPPd
        else:
            mPP = PP.mmPP # match of PP's m
            PPP = PP.PPPm

        if mPP > 0 and not isinstance(PPP, CPPP):
            PPP = CPPP()              # init new PPP
            accum_PPP(PPP, PP, fPPd)  # accum PP into PPP
            form_PPP_recursive(PPP_, PPP, PP.upconnect_, checked_ids=[PP.id], fPPd=fPPd)
            PPP_.append(PPP) # pack PPP after scanning all upconnects

    return PPP_

def form_PPP_recursive(PPP_, PPP, upconnect_,  checked_ids, fPPd):

    for _PP in upconnect_:
        if _PP.id not in checked_ids:
            checked_ids.append(_PP.id)

            if fPPd: _mPP = _PP.mdPP   # match of _PPs' d
            else:    _mPP = _PP.mmPP   # match of _PPs' m

            if _mPP>0 :  # _PP.mPP >0

                if fPPd: _PPP = _PP.PPPd
                else:    _PPP = _PP.PPPm

                if isinstance(_PPP, CPPP):     # _PP's PPP exists, merge with current PPP
                    PPP_.remove(_PPP)    # remove the merging PPP from PPP_
                    merge_PPP(PPP, _PPP, fPPd)
                else:
                    accum_PPP(PPP, _PP, fPPd)  # accum PP into PPP
                    if _PP.upconnect_:         # continue with _PP upconnects
                        form_PPP_recursive(PPP_, PPP, _PP.upconnect_,  checked_ids, fPPd)


def accum_PPP(PPP, PP, fPPd):

    PPP.accum_from(PP) # accumulate parameter
    if fPPd:
        PPP.PPd_.append(PP) # add PPd to PPP's PPd_
        PP.PPPd = PPP       # update PPP reference of PP
    else:
        PPP.PPm_.append(PP) # add PPm to PPP's PPm_
        PP.PPPm = PPP       # update PPP reference of PP


def merge_PPP(PPP, _PPP, fPPd):
    if fPPd:
        for _PP in _PPP.PPd_:
            if _PP not in PPP.PPd_:
                accum_PPP(PPP, _PP, fPPd)
    else:
        for _PP in _PPP.PPm_:
            if _PP not in PPP.PPm_:
                accum_PPP(PPP, _PP, fPPd)


def comp_PP(PP, _PP):

    # match and difference of _PP and PP
    difference = _PP.difference(PP)
    match = _PP.min_match(PP)

    # match of compared PPs' m components
    mmPP = match['mP'] + match['mx'] + match['mL'] + match['mDx'] + match['mDy'] - ave_mPP
    # difference of compared PPs' m components
    dmPP = difference['mP'] + difference['mx'] + difference['mL'] + difference['mDx'] + difference['mDy'] - ave_mPP

    # match of compared PPs' d components
    mdPP = match['dP'] + match['dx'] + match['dL'] + match['dDx'] + match['dDy']
    # difference of compared PPs' d components
    ddPP = difference['dP'] + difference['dx'] + difference['dL'] + difference['dDx'] + difference['dDy']

    derPP = CderPP(PP=PP, _PP=_PP, mmPP=mmPP, dmPP = dmPP,  mdPP=mdPP, ddPP=ddPP)

    return derPP

def accum_derPP(PP, derPP, fPPd):

    if fPPd: # PP cluster by d
        PP.derPPd_.append(derPP)
        PP.accumulate(mdPP=derPP.mdPP, ddPP=derPP.ddPP)
    else:    # PP cluster by m
        PP.derPPm_.append(derPP)
        PP.accumulate(mmPP=derPP.mmPP, dmPP=derPP.dmPP)