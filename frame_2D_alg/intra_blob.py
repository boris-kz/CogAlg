'''
    Intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
'''
import numpy as np
from itertools import zip_longest
from frame_blobs import assign_adjacents, flood_fill, idert
from vectorize_edge_blob.root import vectorize_root
from utils import kernel_slice_3x3 as ks
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

def intra_blob_root(root_blob, render, verbose):  # recursive evaluation of cross-comp slice| range| per blob

    spliced_layers = []

    for blob in root_blob.rlayers[0]:
        # <--- extend der__t: cross-comp in larger kernels
        y0, yn, x0, xn = blob.box  # dert box
        Y, X = blob.root_der__t[0].shape  # root dert size
        # set pad size, e is for extended
        y0e = max(0, y0 - 1); yne = min(Y, yn + 1)
        x0e = max(0, x0 - 1); xne = min(X, xn + 1)
        # take extended der__t from part of root_der__t:
        blob.der__t = idert(
            *(par__[y0e:yne, x0e:xne] for par__ in blob.root_der__t))
        # extend mask__:
        blob.mask__ = np.pad(
            blob.mask__, ((y0 - y0e, yne - yn), (x0 - x0e, xne - xn)),
            constant_values=True, mode='constant')
        # extends blob box
        blob.box = (y0e, yne, x0e, xne)
        Ye = yne - y0e; Xe = xne - x0e
        # ---> end extend der__t

        # increment forking sequence: g -> r|v
        if Ye > 3 and Xe > 3:  # min blob dimensions: Ly, Lx
            # <--- r fork
            if blob.G < aveR * blob.rdn and blob.sign:  # below-average G, eval for comp_r
                blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                    if verbose: print('fork: r')
                    sign__ = ave * (blob.rdn + 1) - new_der__t.g > 0  # m__ = ave - g__
                    # form sub_blobs:
                    sub_blobs, idmap, adj_pairs = flood_fill(
                        new_der__t, sign__, prior_forks=blob.prior_forks + 'r', verbose=verbose, mask__=new_mask__)
                    '''
                    adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
                    sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
                    + if sub_blob_val > alt_blob_val, else -?  
                    '''
                    # adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
                    # blob.rdn += adj_rdn
                    # for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
                    assign_adjacents(adj_pairs)

                    sublayers = blob.rlayers
                    sublayers += [sub_blobs]  # next level sub_blobs, then add deeper layers of sub_blobs:
                    sublayers += intra_blob_root(blob, render, verbose)  # recursive eval cross-comp per blob
                    spliced_layers[:] += [spliced_layer + sublayer for spliced_layer, sublayer in
                                          zip_longest(spliced_layers, sublayers, fillvalue=[])]
            # ---> end r fork
            # <--- v fork
            if blob.G > aveG * blob.rdn and not blob.sign:  # above-average G, vectorize blob
                blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                blob.prior_forks += 'v'
                if verbose: print('fork: v')
                # vectorize_root(blob, verbose=verbose)
            # ---> end v fork
    return spliced_layers


def comp_r(dert__, rng, mask__=None):
    '''
    Selective sampling: skipping current rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel.
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
    # unmask all derts in kernels with only one masked dert (can be set to any number of masked derts),
    # to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    # unmasked derts were computed due to extend_dert() in intra_blob
    if mask__ is not None:
        smsk__ = mask__[::2, ::2]     # sparse_mask__
        majority_mask__ = (
                smsk__[ks.tl].astype(int) + smsk__[ks.tc].astype(int) + smsk__[ks.tr].astype(int) +
                smsk__[ks.ml].astype(int) + smsk__[ks.mc].astype(int) + smsk__[ks.mr].astype(int) +
                smsk__[ks.bl].astype(int) + smsk__[ks.bc].astype(int) + smsk__[ks.br].astype(int)
            ) > 1
    else:
        majority_mask__ = None  # returned at the end of function

    # unpack dert__:
    i__, dy__, dx__, g__ = dert__  # i is pixel intensity, g is gradient
    si__ = i__[::2, ::2]  # sparse_i, i is pixel intensity
    sdy__ = dy__[::2, ::2].copy()  # sparse_dy, for accumulation
    sdx__ = dx__[::2, ::2].copy()  # sparse_dx, for accumulation

    # compare opposed pairs of rim pixels, project onto x, y:
    dist = 2**rng

    sdy__[ks.mc] += (
        (si__[ks.bl] - si__[ks.tr]) * 0.5 +
        (si__[ks.bc] - si__[ks.tc]) +
        (si__[ks.br] - si__[ks.tl]) * 0.5
    ) / dist

    sdx__[ks.mc] += (
        (si__[ks.tr] - si__[ks.bl]) * 0.5 +
        (si__[ks.mr] - si__[ks.mc]) +
        (si__[ks.br] - si__[ks.tl]) * 0.5
    ) / dist
    sg__ = np.hypot(sdy__, sdx__)  # gradient, recomputed at each comp_r

    return idert(si__[ks.mc], sdy__, sdx__, sg__), majority_mask__