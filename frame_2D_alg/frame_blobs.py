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
from itertools import product
import numpy as np
# hyper-parameters, set as a guess, latter adjusted by feedback:
ave = 30  # base filter, directly used for comp_r fork
ave_a = 1.5  # coef filter for comp_a fork
aveB = 50
aveBa = 1.5
ave_mP = 100
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

def frame_blobs_root(i__):
    der__t = comp_pixel(i__)  # compare all in parallel -> i__, dy__, dx__, g__, s__
    frame = i__, I, Dy, Dx, blob_ = [i__, 0, 0, 0, []]  # init frame as output

    # Flood-fill 1 pixel at a time
    Y, X = i__.shape  # get i__ height and width
    fill_yx_ = list(product(range(1,Y-1), range(1,X-1)))  # set of pixel coordinates to be filled (fill_yx_)
    root__ = {}  # id map pixel to blob
    perimeter_ = []  # perimeter pixels
    while fill_yx_:  # fill_yx_ is popped per filled pixel, in form_blob
        if not perimeter_:  # init blob
            blob = [frame, None, 0, 0, 0, [], [], []]  # root (frame), sign, I, Dy, Dx, yx_, dert_, link_ (up-links)
            perimeter_ += [fill_yx_[0]]

        form_blob(blob, fill_yx_, perimeter_, root__, der__t)  # https://en.wikipedia.org/wiki/Flood_fill

        if not perimeter_:  # term blob
            frame[1] += blob[2]  # I
            frame[2] += blob[3]  # Dy
            frame[3] += blob[4]  # Dx
            blob_ += [blob]

    return frame

def comp_pixel(i__):
    # compute directional derivatives:
    dy__ = (
        (i__[2:,  :-2] - i__[:-2, 2:  ]) * 0.25 +
        (i__[2:, 1:-1] - i__[:-2, 1:-1]) * 0.50 +
        (i__[2:, 2:  ] - i__[:-2, 2:  ]) * 0.25
    )
    dx__ = (
        (i__[ :-2, 2:] - i__[2:  ,  :-2]) * 0.25 +
        (i__[1:-1, 2:] - i__[1:-1,  :-2]) * 0.50 +
        (i__[2:  , 2:] - i__[ :-2, 2:  ]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)                          # compute gradient magnitude
    s__ = ave - g__ > 0  # sign is positive for below-average g

    return i__, dy__, dx__, g__, s__


def form_blob(blob, fill_yx_, perimeter_, root__, der__t):
    # unpack structures
    root, sign, I, Dy, Dx, yx_, dert_, link_ = blob
    i__, dy__, dx__, g__, s__ = der__t
    Y, X = g__.shape

    # get and check coord
    y, x = perimeter_.pop()  # get pixel coord
    if y < 1 or y > Y or x < 1 or x > X: return  # out of bound
    i = i__[y, x]; dy = dy__[y-1, x-1]; dx = dx__[y-1, x-1]; s = s__[y-1, x-1] # get dert from arrays, -1 coords for shrunk arrays
    if (y, x) not in fill_yx_:  # if adjacent filled, this is pixel of an adjacent blob
        _blob = root__[y, x]
        if _blob not in link_: link_ += [_blob]
        return
    if sign is None: sign = s  # assign sign to new blob
    if sign != s: return   # different sign, stop

    # fill coord, proceed with form_blob
    fill_yx_.remove((y, x))  # remove from yx_
    root__[y, x] = blob  # assign root, for link forming
    I += i; Dy += dy; Dx += dx  # update params
    yx_ += [(y, x)]; dert_ += [(i, dy, dx)]  # update elements

    # update perimeter_
    perimeter_ += [(y-1,x), (y,x+1), (y+1,x), (y,x-1)]  # extend perimeter
    if sign: perimeter_ += [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]  # ... include diagonals for +blobs

    blob[:] = root, sign, I, Dy, Dx, yx_, dert_, link_ # update blob


if __name__ == "__main__":
    # standalone script, frame_blobs doesn't import from higher modules (like intra_blob).
    # Instead, higher modules will import from frame_blobs and will have their own standalone scripts like below.
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    args = argument_parser.parse_args()
    image = imread(args.image)

    frame = frame_blobs_root(image)

    # verification/visualization:
    import matplotlib.pyplot as plt
    _, I, Dy, Dx, blob_ = frame  # ignore

    i__ = np.zeros_like(image, dtype=np.float32)
    dy__ = np.zeros_like(image, dtype=np.float32)
    dx__ = np.zeros_like(image, dtype=np.float32)
    s__ = np.zeros_like(image, dtype=np.float32)
    line_ = []

    for blob in blob_:
        root, sign, I, Dy, Dx, yx_, dert_, link_ = blob
        for yx, (i, dy, dx) in zip(yx_, dert_):
            i__[yx] = i
            dy__[yx] = dy
            dx__[yx] = dx
            s__[yx] = sign
        y, x = map(np.mean, zip(*yx_))  # blob center of gravity
        for _blob in link_:  # show links
            _, _, _, _, _, _yx_, _, _ = _blob
            _y, _x = map(np.mean, zip(*_yx_))  # _blob center of gravity
            line_ += [((_x, x), (_y, y))]

    plt.imshow(i__, cmap='gray'); plt.show()    # show reconstructed i__
    plt.imshow(dy__, cmap='gray'); plt.show()   # show reconstructed dy__
    plt.imshow(dx__, cmap='gray'); plt.show()   # show reconstructed dx__

    # show blobs and links
    plt.imshow(s__, cmap='gray')
    for line in line_:
        plt.plot(*line, "b-")
    plt.show()