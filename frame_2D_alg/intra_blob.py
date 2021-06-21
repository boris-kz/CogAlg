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
from slice_utils import *
from segment_by_direction import segment_by_direction

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering
ave_ga = .78
ave_ma = 2

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

            adert__, mask__ = comp_a(ext_dert__, ave_ma, ave_ga, blob.prior_forks, ext_mask__)  # compute ma and ga
            blob.f_comp_a = 1
            blob.rng = 0
            if kwargs.get('verbose'): print('\na fork\n')
            blob.prior_forks.extend('a')

            if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, least one dert in dert__
                sign__ = adert__[3] * adert__[8] > 0   # g * (ma / ave: deviation rate, no independent value, not co-measurable with g)
                dert__ = adert__  # already flat, adert__[[5, 6]] are now complex

                cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs)  # forms sub_blobs of sign in unmasked area
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.M > AveB * 1.2:  # comp_r fork, ave M = ave G * 1.2
            blob.rng += 1

            dert__, mask__ = comp_r(ext_dert__, Ave, blob.rng, blob.f_root_a, ext_mask__)
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
        G = blob.G  # Gr, Grr...
        adj_M = blob.adj_blobs[3]  # adj_M is incomplete, computed within current dert_only, use root blobs instead:
        adjacent valuable blobs of any sign are tracked from frame_blobs to form borrow_M?
        track adjacency of sub_blobs: wrong sub-type but right macro-type: flat blobs of greater range?
        G indicates or dert__ extend per blob G?
        borrow_M = min(G, adj_M / 2): usually not available, use average
        '''
        sub_blob.prior_forks = blob.prior_forks.copy()  # increments forking sequence: g->a, g->a->p, etc.

        if sub_blob.G > AveB:  # replace with borrow_M when known
            # comp_a:
            sub_blob.f_root_a = 1
            sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub_blob, currently not used
            sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
            blob.sub_layers += intra_blob(sub_blob, **kwargs)

        elif sub_blob.M - aveG > AveB:
            # comp_r:
            sub_blob.rng = blob.rng + 1  # correct?
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