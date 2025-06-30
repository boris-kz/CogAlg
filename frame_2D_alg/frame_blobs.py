import weakref
import numpy as np
from matplotlib import pyplot as plt
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
    Frame_blobs is a start for all deeper processing in 2D alg.
    -
    Please see illustrations:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/blob_params.drawio
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs.png
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/frame_blobs_intra_blob.drawio

    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''
class CBase:
    refs = []
    def __init__(obj):
        obj._id = len(obj.refs)
        obj.refs.append(weakref.ref(obj))
    def __hash__(obj): return obj.id
    @property
    def id(obj): return obj._id
    @classmethod
    def get_instance(cls, _id):
        inst = cls.refs[_id]()
        if inst is not None and inst.id == _id:
            return inst
    def __repr__(obj): return f"{obj.__class__.__name__}(id={obj.id})"
    '''
    def __getattribute__(ave,name):
        coefs =   object.__getattribute__(ave, "coefs")
        if name == "coefs":
            return object.__getattribute__(ave, name)
        elif name == "md":
            return [ave.m * coefs["m"], ave.d *  coefs["d"]]  # get updated md
        else:
            return object.__getattribute__(ave, name)  * coefs[name]  # always return ave * coef
    '''
# hyper-parameters, init a guess, adjusted by feedback
ave  = 30  # base filter, directly used for comp_r fork
aveR = 10  # for range+, fixed overhead per blob

class CN(CBase):
    name = "node"
    def __init__(n, **kwargs):
        super().__init__()
        n.N_ = kwargs.get('N_',[])  # N_| nrim
        n.L_ = kwargs.get('L_',[])  # L_| rim
        n.nH = kwargs.get('nH',[])  # top-down: feedback of summed sub-node levels: CN with single added-layer derH, no H
        n.lH = kwargs.get('lH',[])  # bottom-up: higher link graphs hierarchy, also CN levs
        n.Et = kwargs.get('Et', np.zeros(3))  # sum from L_ or rims
        n.olp = kwargs.get('olp',1)  # overlap to other Ns, same for links?
        n.derH  = kwargs.get('derH',[])  # sum from L_ or rims
        n.derTT = kwargs.get('derTT',np.zeros((2,8)))  # sum from derH
        n.baseT = kwargs.get('baseT',np.zeros(4))
        n.yx    = kwargs.get('yx', np.zeros(2))  # [(y+Y)/2,(x,X)/2], from nodet, then ave node yx
        n.rng   = kwargs.get('rng',1)  # or med: loop count in comp_node_|link_
        n.box   = kwargs.get('box',np.array([np.inf, np.inf, -np.inf, -np.inf]))  # y0, x0, yn, xn
        n.span  = kwargs.get('span',0) # distance in nodet or aRad, comp with baseT and len(N_) but not additive?
        n.angle = kwargs.get('angle',np.zeros(2))  # dy,dx
        # nested CNs:
        n.root= kwargs.get('root',[])  # not in ext_
        n.rim = kwargs.get('rim',[])  # nrim, rim and their attrs, replaces CG
        n.alt = kwargs.get('alt',[])  # adjacent (contour) gap+overlap alt-fork graphs, converted to CG, empty alt.alt_: select+?
        n.fi  = kwargs.get('fi', 1)  # if G else 0, fd_: list of forks forming G?
        n.fin = kwargs.get('fin',0)  # in cluster, temporary?
        # n.fork_tree: list = z([[]])  # indices in all layers(forks, if no fback merge, G.fback_=[] # node fb buffer, n in fb[-1]
    def __bool__(n): return bool(n.N_)

class CBlob(CBase):

    def __init__(blob, root):
        super().__init__()
        blob.root = root
        blob.sign = None
        blob.area = 0
        blob.latuple = [0, 0, 0, 0, 0, 0]  #I, G, Dy, Dx, Y, X
        blob.dert_ = {}  # keys: (y, x). values: (i, g, dy, dx)
        blob.adj_ = []  # adjacent blobs
        blob.yx = np.zeros(2)

    def fill_blob(blob, fill_yx_, perimeter_, root__, dert__):
        y, x = perimeter_.pop()  # pixel coord
        if (y, x) not in dert__: return  # out of bound
        i, g, dy, dx, s = dert__[y,x]
        if (y, x) not in fill_yx_:  # there is a bug with last blob here
            if blob.sign != s:  # adjacent blobs have opposite sign
                _blob = root__[y, x]
                if _blob not in blob.adj_: blob.adj_ += [_blob]
            return
        if blob.sign is None: blob.sign = s  # assign sign to new blob
        if blob.sign != s: return  # different blob.sign, stop
        fill_yx_.remove((y,x))
        root__[y,x] = blob  # assign root, for link forming
        blob.area += 1
        I, G, Dy, Dx, Y, X  = blob.latuple
        Y += y; X += x; I += i; Dy += dy; Dx += dx; G += g  # update params
        blob.latuple = I, G, Dy, Dx, Y, X
        blob.dert_[y, x] = i, g, dy, dx  # update elements

        perimeter_ += [(y-1,x), (y,x+1), (y+1,x), (y,x-1)]  # extend perimeter
        if blob.sign: perimeter_ += [(y-1,x-1), (y-1,x+1), (y+1,x+1), (y+1,x-1)]  # ... include diagonals for +blobs

    def term(blob):
        blob.yx = np.array(list(map(np.mean, zip(*blob.yx_))))
        frame = blob.root
        I, G, Dy, Dx = frame.baseT
        *_, i, dy, dx, g = blob.latuple
        I += i; G += g; Dy += dy; Dx += dx
        frame.baseT = I, G, Dy, Dx
        frame.N_ += [blob]
        if blob.sign:   # transfer adj_ from +blob to -blobs and remove
            for _blob in blob.adj_:
                if blob not in _blob.adj_: _blob.adj_ += [blob]
            del blob.adj_  # prevents circular assignment
        # edges are unpacked so adjacents have to be assigned to slices) PPs as higher-order Alt_, probably not worth it
    @property
    def G(blob): return blob.latuple[-1]
    @property
    def yx_(blob): return list(blob.dert_.keys())


def frame_blobs_root(image, rV=1, fintra=0):
    global ave, aveR
    ave *= rV; aveR *= rV

    dert__ = comp_pixel(image) if isinstance(image[0],int) else image  # precomputed
    i__, g__, dy__, dx__, s__ = dert__  # convert to dict for flood-fill
    y__, x__ = np.indices(i__.shape)
    dert__ = dict(zip(
        zip(y__.flatten(), x__.flatten()),
        zip(i__.flatten(), g__.flatten(), dy__.flatten(), dx__.flatten(), s__.flatten()),
    ))
    frame = CN(box = np.array([0,0,image.shape[0],image.shape[1]]),
               yx  = np.array([image.shape[0]//2, image.shape[1]//2]),
               L_  = image)  # temporary
    flood_fill(frame, dert__)  # flood-fill 1 pixel at a time
    if fintra and frame.baseT[1] > ave:
        intra_blob_root(frame)  # kernel size extension
    return frame

def comp_pixel(i__):  # compare all in parallel -> i__, g__, dy__, dx__, s__
    # compute directional derivatives:
    dy__ = (
            (i__[2:, :-2] - i__[:-2, 2:]) * 0.25 +
            (i__[2:, 1:-1] - i__[:-2, 1:-1]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25
    )
    dx__ = (
            (i__[:-2, 2:] - i__[2:, :-2]) * 0.25 +
            (i__[1:-1, 2:] - i__[1:-1, :-2]) * 0.50 +
            (i__[2:, 2:] - i__[:-2, 2:]) * 0.25
    )
    g__ = np.hypot(dy__, dx__)  # compute gradient magnitude, -> separate G because it's not signed, dy,dx cancel out in Dy,Dx
    s__ = ave - g__ > 0  # sign, positive = below-average g
    dert__ = np.stack([i__[:-2,:-2],g__,dy__,dx__,s__])

    return dert__

def flood_fill(frame, dert__):
    # Flood-fill 1 pixel at a time
    fill_yx_ = list(dert__.keys())  # set of pixel coordinates to be filled (fill_yx_)
    root__ = {}  # map pixel to blob
    perimeter_ = []  # perimeter pixels
    while fill_yx_:  # fill_yx_ is popped per filled pixel, in form_blob
        if not perimeter_:  # init blob
            blob = CBlob(frame); perimeter_ += [fill_yx_[0]]
        blob.fill_blob(fill_yx_, perimeter_, root__, dert__)  # https://en.wikipedia.org/wiki/Flood_fill
        if not perimeter_ or not fill_yx_:
            blob.term()
'''
intra_blob recursively segments each blob for two forks of extended internal cross-comp and sub-clustering:
- comp_range: incremental range cross-comp in low-variation blobs: >ave negative gradient
- vectorize_root: slice_edge -> comp_slice -> agg_recursion
blobs that terminate on frame edge will have to be spliced across frames
'''

class CrNode_(CN):
    def __init__(rnode_, blob):
        super().__init__(blob.root.L_)  # init params, extra params init below:
        rnode_.root = blob
        rnode_.olp= blob.root.olp + 1.5
        rnode_.rng = blob.root.rng + 1

def intra_blob_root(frame, rV=1):

    global aveR
    aveR *= rV
    frame.olp = frame.rng = 1
    for blob in frame.N_:
        rblob(blob)

def rblob(blob):
    if not blob.sign or blob.G >= ave*blob.area + aveR*blob.root.olp:
        return

    # sign and G < ave*L + aveR*olp:
    rnode_ = CrNode_(blob)
    dert__ = comp_r(rnode_)     # return None if blob is too small
    if dert__ is None: return   # terminate if blob is too small

    # rnode_ is added dynamically, only positive blobs may have rnode_:
    blob.rnode_ = rnode_
    flood_fill(rnode_, dert__)

    for bl in rnode_.N_: # recursive eval cross-comp per blob
        rblob(bl)

def comp_r(rnode_):   # rng+ comp
    # compute kernel
    ky__, kx__ = compute_kernel(rnode_.rng)
    # loop through root_blob's pixels
    dert__ = {}     # mapping from y, x to dert
    for (y, x), (p, g, dy, dx) in rnode_.root.dert_.items():
        try:
            # comparison. i,j: relative coord within kernel 0 -> rng*2+1
            for i, j in zip(*ky__.nonzero()):
                dy += ky__[i, j] * rnode_.L_[y+i-rnode_.rng, x+j-rnode_.rng]    # -rng to get i__ coord
            for i, j in zip(*kx__.nonzero()):
                dx += kx__[i, j] * rnode_.L_[y+i-rnode_.rng, x+j-rnode_.rng]
        except IndexError: continue     # out of bound
        g = np.hypot(dy, dx)
        s = ave*(rnode_.olp + 1) - g > 0
        dert__[y, x] = p, g, dy, dx, s
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
    dy, dx = k = np.indices((ksize, ksize)) - rng  # kernel span around (0, 0)
    sqr_dist = dx*dx + dy*dy  # squared distance
    sqr_dist[rng, rng] = 1  # avoid division by 0
    coeff = k / sqr_dist  # kernel coefficient
    coeff[1:-1, 1:-1] = 0  # non-rim = 0

    return coeff

def imread(filename, raise_if_not_read=True):  # Read an image in grayscale, return array
    try: return np.mean(plt.imread(filename), axis=2).astype(float)
    except AttributeError:
        if raise_if_not_read: raise SystemError('image is not read')
        else: print('Warning: image is not read')

def unpack_blob_(frame):
    blob_ = []
    q_ = list(frame.N_)
    while q_:
        blob = q_.pop(0)
        blob_ += [blob]
        if hasattr(blob, "rnode_") and blob.rnode_.N_:  # if blob is extended with rnode_
            q_ += blob.rnode_.N_
    return blob_

if __name__ == "__main__":

    # image_file = './images//raccoon_eye.jpeg'
    image_file = './images//toucan_small.jpg'
    image = imread(image_file)
    frame = frame_blobs_root(image)
    # verification (intra):
    for blob in unpack_blob_(frame):
        print(f"{blob}'s parent is {blob.root}", end="")
        if hasattr(blob, "rnode_") and blob.rnode_.N_:  # if blob is extended with rnode_
            cnt = len(blob.rnode_.N_)
            print(f", has {cnt} sub-blob{'' if cnt == 1 else 's'}")
        else: print()  # the blob is not extended, skip

    I, G, Dy, Dx = frame.baseT
    # verification:
    i__ = np.zeros_like(image, dtype=np.float32)
    g__ = np.zeros_like(image, dtype=np.float32)
    dy__ = np.zeros_like(image, dtype=np.float32)
    dx__ = np.zeros_like(image, dtype=np.float32)
    s__ = np.zeros_like(image, dtype=np.float32)
    line_ = []
    for blob in frame.N_:
        for (y, x), (i, g, dy, dx) in blob.dert_.items():
            i__[y, x] = i; g__[y, x] = g; dy__[y, x] = dy; dx__[y, x] = dx; s__[y, x] = blob.sign
        y,x = blob.yx
        if not blob.sign:
            for _blob in blob.adj_:  # show adjacents
                _y, _x = _blob.yx  # _blob center of gravity
                line_ += [((_x, x), (_y, y))]

    plt.imshow(i__, cmap='gray'); plt.show()  # show reconstructed i__
    plt.imshow(dy__,cmap='gray'); plt.show()  # show reconstructed dy__
    plt.imshow(dx__,cmap='gray'); plt.show()  # show reconstructed dx__
    plt.imshow(dx__,cmap='gray'); plt.show()  # show reconstructed dx__
    plt.imshow(g__, cmap='gray'); plt.show()  # show reconstructed g__

    # show blobs and links
    plt.imshow(s__, cmap='gray')
    for line in line_:
        plt.plot(*line, "b-")

    plt.show()