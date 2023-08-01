'''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
'''
import numpy as np
from itertools import zip_longest
from class_cluster import ClusterStructure
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_r, comp_a
from vectorize_edge_blob.root import vectorize_root

class CEdge(ClusterStructure):  # edge blob

    sign: bool = None
    I: float = 0.0
    Dy: float = 0.0
    Dx: float = 0.0
    G: float = 0.0
    A: float = 0.0  # blob area
    M: float = 0.0  # summed PP.M, for both types of recursion?
    # composite params:
    box: tuple = (0, 0, 0, 0)  # y0, yn, x0, xn
    mask__ : object = None
    der__t : object = None
    der__t_roots: object = None  # map to dir__t
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
# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
ave_a = 2*(2**0.5) / 8  # 1/8 maximum ga possible
aveR = 10  # for range+, fixed overhead per blob
aveG = 10  # for vectorize
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
            if blob.G < aveR * blob.rdn and blob.sign:  # below-average G, eval for comp_r
                blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                # comp_r 5x5?:
                new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                sign__ = ave * (blob.rdn+1) - new_der__t[3] > 0  # m__ = ave - g__
                # if min Ly and Lx, der__t>=1: form, splice sub_blobs:
                if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                    spliced_layers[:] =\
                        cluster_fork_recursive( blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=0)

            if blob.G > aveG * blob.rdn and not blob.sign:  # above-average G, vectorize blob
                blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                blob.prior_forks += 'v'
                if verbose: print('fork: v')
                # convert Cblob to Cedge:
                edge = CEdge(der__t=blob.der__t, der__t_roots=[[[] for col in row] for row in blob.der__t[0]], mask__=blob.mask__,
                             I=blob.I)
                vectorize_root(edge, verbose=verbose)

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
    ext_der__t = type(blob.der__t)(
        *(par__[y0e:yne, x0e:xne] for par__ in blob.root_der__t))

    # extend mask__:
    ext_mask__ = np.pad(blob.mask__,
                        ((y0 - y0e, yne - yn),
                         (x0 - x0e, xne - xn)),
                        constant_values=True, mode='constant')
    blob.der__t = ext_der__t
    blob.mask__ = ext_mask__
    blob.der__t_roots = [[[] for _ in range(x0e, xne)] for _ in range(y0e, yne)]
    blob.box = (y0e, yne, x0e, xne)

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