'''
    Intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
'''
import numpy as np
from frame_blobs import frame_blobs_root
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
# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
aveR = 10  # for range+, fixed overhead per blob
aveG = 10  # for vectorize
ave_nsub = 4  # ave n sub_blobs per blob: 4x higher costs? or eval costs only, separate clustering ave = aveB?
# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob_root(frame): # init rng+ recursion
    i__, I, Dy, Dx, blob_ = frame

    for blob in blob_:
        blob += [1, 1, []]     # extend blob with rdn, rng, sub_blob_
        intra_blob(blob, i__)

def intra_blob(root_blob, i__):  # recursive evaluation of cross-comp rng+ per blob
    # unpack root_blob
    root, sign, I, Dy, Dx, yx_, dert_, link_, rdn, rng, blob_ = root_blob

    if not sign: return  # only for below-average G
    G = np.hypot(Dy, Dx)
    if G >= aveR * rdn: return  # eval for comp_r
    rdn += 1.5  # update rdn
    rng += 1  # update rng
    dert__ = comp_r(i__, yx_, dert_, rdn, rng)  # return None if blob is too small
    if not dert__: return

    # proceed to form new fork, flood-fill 1 pixel at a time
    fyx_ = list(dert__.keys())  # set of pixel coordinates to be filled, from root_blob
    root__ = {}  # id map pixel to blob
    perimeter_ = []  # perimeter pixels
    while cell_:  # cell_ is popped per filled pixel, in form_blob
        if not perimeter_:  # init blob
            blob = [root_blob, None, 0, 0, 0, [], [], rdn, rng, []]  # root, sign, I, Dy, Dx, yx_, link_, rdn, rng, blob_
            perimeter_ += [fyx_[0]]

        form_blob(blob, fyx_, perimeter_, root__, dert__)

        if not perimeter_:  # term blob
            blob_ += [blob]
            intra_blob(blob, i__)  # recursive eval cross-comp per blob


def comp_r(i__, yx_, dert_, rng, rdn):
    Y, X = i__.shape

    # compute kernel
    ky__, kx__ = compute_kernel(rng)

    # loop through root_blob's pixels
    dert__ = {}     # mapping from y, x to dert
    for (y, x), (p, dy, dx) in zip(yx_, dert_):
        if y-rng < 0 or y+rng >= Y or x-rng < 0 or x+rng >= X: continue # boundary check

        # comparison. i,j: relative coord within kernel 0 -> rng*2+1
        for i, j in zip(*ky__.nonzero()):
            dy += ky__[i, j] * i__[y+i-rng, x+j-rng]    # -rng to get i__ coord
        for i, j in zip(*kx__.nonzero()):
            dx += kx__[i, j] * i__[y+i-rng, x+j-rng]

        g = np.hypot(dy, dx)
        s = ave*(rdn + 1) - g > 0

        dert__[y, x] = p, dy, dx, s

    return dert__


def compute_kernel(rng):
    # kernel_coefficient = projection_coefficient / distance
    #                    = [sin(angle), cos(angle)] / distance
    # With: distance = sqrt(x*x + y*y)
    #       sin(angle) = y / sqrt(x*x + y*y) = y / distance
    #       cos(angle) = x / sqrt(x*x + y*y) = x / distance
    # Thus:
    # kernel_coefficient = [y / sqrt(x*x + y*y), x / sqrt(x*x + y*y)] / sqrt(x*x + y*y)
    #                    = [y, x] / (x*x + y*y)
    ksize = rng*2+1  # kernel size
    y, x = k = np.indices((ksize, ksize)) - rng  # kernel span around (0, 0)
    sqr_dist = x*x + y*y  # squared distance
    sqr_dist[rng, rng] = 1  # avoid division by 0
    coeff = k / sqr_dist  # kernel coefficient
    coeff[1:-1, 1:-1] = 0  # non-rim = 0

    return coeff


def form_blob(blob, fyx_, perimeter_, root__, dert__):
    # unpack structures
    root, sign, I, Dy, Dx, yx_, dert_, link_, *other_params = blob

    # get and check coord
    y, x = perimeter_.pop()  # get pixel coord
    if (y, x) not in dert__: return  # out of bound
    i, dy, dx, s = dert__[y, x]
    if (y, x) not in fyx_:  # if adjacent filled, this is pixel of an adjacent blob
        _blob = root__[y, x]
        if _blob not in link_: link_ += [_blob]
        return
    if sign is None: sign = s  # assign sign to new blob
    if sign != s: return   # different sign, stop

    # fill coord, proceed with form_blob
    fyx_.remove((y, x))  # remove from yx_
    root__[y, x] = blob  # assign root, for link forming
    I += i; Dy += dy; Dx += dx  # update params
    yx_ += [(y, x)]; dert_ += [(i, dy, dx)]  # update elements

    # update perimeter_
    perimeter_ += [(y-1,x), (y,x+1), (y+1,x), (y,x-1)]  # extend perimeter
    if sign: perimeter_ += [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]  # ... include diagonals for +blobs

    blob[:] = root, sign, I, Dy, Dx, yx_, dert_, link_, *other_params # update blob



if __name__ == "__main__":
    import argparse
    from utils import imread
    # Parse arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    args = argument_parser.parse_args()
    image = imread(args.image)

    frame = frame_blobs_root(image)
    intra_blob_root(frame)