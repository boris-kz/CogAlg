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

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, G, M, blob_, dert__')


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
            I += blob.I
            Dy += blob.Dy
            Dx += blob.Dx
            G += blob.G
            M += blob.M
        frame = FrameOfBlobs(I=I, Dy=Dy, Dx=Dx, G=G, M=M, blob_=blob_, dert__=dert__)

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
                                    Dy=dert__[1][y][x],
                                    Dx=dert__[2][y][x],
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