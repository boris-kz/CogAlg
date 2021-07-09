'''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - comp_slice_ forms roughly edge-orthogonal Ps, their stacks evaluated for rotation, comp_d, and comp_slice
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
'''

import numpy as np
from frame_blobs import assign_adjacents, flood_fill, CBlob
from intra_comp import comp_r, comp_a
from draw_frame_blobs import visualize_blobs
from itertools import zip_longest
from comp_slice_ import *
from segment_by_direction import segment_by_direction

# filters, All *= rdn:
ave = 50  # comp_a cost per dert, from average m, reflects blob definition cost
aveB = 50  # comp_a cost per intra_blob comp and clustering
# no ave_ga = .78, ave_ma = 2: no indep eval
pcoef = 2  # ave_comp_P / ave: relative cost of p fork
rcoef = 1  # ave_comp_r / ave: relative cost of r fork

# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob(blob, **kwargs):  # slice_blob or recursive input rng+ | angle cross-comp within input blob

    Ave = int(ave * blob.rdn)
    AveB = int(aveB * blob.rdn)
    verbose = kwargs.get('verbose')

    if kwargs.get('render') is not None:  # don't render small blobs
        if blob.A < 100: kwargs['render'] = False
    spliced_layers = []  # to extend root_blob sub_layers

    # root fork is frame_blobs or comp_r
    ext_dert__, ext_mask__ = extend_dert(blob)  # dert__ boundaries += 1, for cross-comp in larger kernels

    G = np.hypot(blob.Dy,blob.Dx)

    if G > AveB:  # comp_a fork, replace G with borrow_M if known
        adert__, mask__ = comp_a(ext_dert__, ext_mask__)  # compute abs ma, no indep eval
        blob.f_comp_a = 1
        blob.rng = 0
        if kwargs.get('verbose'): print('\na fork\n')
        blob.prior_forks.extend('a')

        if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, least one dert in dert__
            sign__ = ((np.hypot(adert__[1],adert__[2])) * adert__[8]) > ave * pcoef  # variable value of comp_P
            # g * (ma / ave: deviation rate, no independent value, not co-measurable with g)

            cluster_sub_eval(blob, adert__, sign__, mask__, **kwargs)  # forms sub_blobs of fork p sign in unmasked area
            spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                              zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    elif blob.M > AveB * rcoef:  # comp_r fork
        blob.rng += 1  # rng counter for comp_r

        dert__, mask__ = comp_r(ext_dert__, Ave, blob.rng, ext_mask__)
        blob.f_comp_a = 0
        if kwargs.get('verbose'): print('\na fork\n')
        blob.prior_forks.extend('r')

        if mask__.shape[0] > 2 and mask__.shape[1] > 2 and False in mask__:  # min size in y and x, at least one dert in dert__
            sign__ = dert__[3] > 0  # m__ is inverse deviation of SAD

            cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs)  # forms sub_blobs of sign in unmasked area
            spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                              zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    return spliced_layers


def cluster_sub_eval(blob, dert__, sign__, mask__, **kwargs):  # comp_r or comp_a eval per sub_blob:

    AveB = aveB * blob.rdn
    sub_blobs, idmap, adj_pairs = flood_fill(dert__, sign__, verbose=False, mask__=mask__, blob_cls=CBlob)
    assign_adjacents(adj_pairs, CBlob)

    if kwargs.get('render', False):
        visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (f_comp_a = {blob.f_comp_a}, f_root_a = {blob.prior_forks[-1] == 'a'})")

    blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
    blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

    for sub_blob in sub_blobs:  # evaluate sub_blob
        sub_blob.prior_forks = blob.prior_forks.copy()  # increments forking sequence: m->r, g->a, a->p
        if sub_blob.mask__.shape[0] > 2 and sub_blob.mask__.shape[1] > 2 and False in sub_blob.mask__:  # min size in y and x, at least one dert in dert__

            sub_G = np.hypot(sub_blob.Dy,sub_blob.Dx)
            if sub_blob.prior_forks[-1] == 'a':  # p fork
                if (sub_G * sub_blob.Ma - AveB * pcoef > 0):  # vs. G reduced by Ga: * (1 - Ga / (4.45 * A)), max_ga=4.45
                    sub_blob.prior_forks.extend('p')
                    if kwargs.get('verbose'): print('\nslice_blob fork\n')
                    segment_by_direction(sub_blob, verbose=True)

            else: # a fork or r fork
                '''
                G = blob.G  # Gr, Grr...
                adj_M = blob.adj_blobs[3]  # adj_M is incomplete, computed within current dert_only, use root blobs instead:
                adjacent valuable blobs of any sign are tracked from frame_blobs to form borrow_M?
                track adjacency of sub_blobs: wrong sub-type but right macro-type: flat blobs of greater range?
                G indicates or dert__ extend per blob G?
                borrow_M = min(G, adj_M / 2): usually not available, use average
                '''
                if sub_G > AveB:  # replace with borrow_M when known
                    # comp_a:
                    sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub_blob, currently not used ( do we want to keep this?)
                    sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                    blob.sub_layers += intra_blob(sub_blob, **kwargs)

                elif sub_blob.M > AveB * mB_coef:
                    # comp_r:
                    sub_blob.rng = blob.rng
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