'''
    Intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
'''
import numpy as np
from itertools import zip_longest
from frame_blobs import assign_adjacents, flood_fill, idert
from vectorize_edge_blob.root import vectorize_root
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
                # TODO: revise comp_r:
                new_der__t, new_mask__ = comp_r(blob.der__t, blob.rng, blob.mask__)
                sign__ = ave * (blob.rdn+1) - new_der__t.g > 0  # m__ = ave - g__
                if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                    # TODO: revise layers:
                    spliced_layers[:] =\
                        cluster_fork_recursive( blob, spliced_layers, new_der__t, sign__, new_mask__, verbose, render, fBa=0)

            if blob.G > aveG * blob.rdn and not blob.sign:  # above-average G, vectorize blob
                blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
                blob.prior_forks += 'v'
                if verbose: print('fork: v')
                vectorize_root(blob, verbose=verbose)

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


def comp_r(dert__, ave, rng, root_fia, mask__=None):
    '''
    Selective sampling: skipping current rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 3: 9x9 kernel,
    ...
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
    '''
    can't happen:
    if root_fia:  # initialize derivatives:  
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)
    else: 
    '''
     # root fork is comp_r, accumulate derivatives:
    dy__ = dert__[1][1:-1:2, 1:-1:2].copy()  # sparse to align with i__center
    dx__ = dert__[2][1:-1:2, 1:-1:2].copy()
    m__ = dert__[4][1:-1:2, 1:-1:2].copy()

    # compare four diametrically opposed pairs of rim pixels, with Sobel coeffs * rim skip ratio:

    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 9x9, *4 for 17x17

    dy__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * -2  * rngSkip +
             (i__topright - i__bottomleft) * -1 * rngSkip +
             (i__right - i__left) * 0)

    dx__ += ((i__topleft - i__bottomright) * -1 * rngSkip +
             (i__top - i__bottom) * 0 +
             (i__topright - i__bottomleft) * 1 * rngSkip+
             (i__right - i__left) * 2 * rngSkip)

    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    '''
    m__ += ( abs(i__center - i__topleft) * 1 * rngSkip
           + abs(i__center - i__top) * 2 * rngSkip
           + abs(i__center - i__topright) * 1 * rngSkip
           + abs(i__center - i__right) * 2 * rngSkip
           + abs(i__center - i__bottomright) * 1 * rngSkip
           + abs(i__center - i__bottom) * 2 * rngSkip
           + abs(i__center - i__bottomleft) * 1 * rngSkip
           + abs(i__center - i__left) * 2 * rngSkip
           )
    return idert(i__center, dy__, dx__, g__), majority_mask__


def comp_r_2x2(dert__, rng, mask__=None):
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient in shorter-range cross-comp: input intensity didn't vary much.
    Such input is predictable enough for selective sampling: skipping current rim in following comparison kernels.
    Skipping forms increasingly sparse dert__ for next-range cross-comp,
    hence kernel width increases as 2^rng: 1: 2x2 kernel, 2: 4x4 kernel, 3: 8x8 kernel
    There is also skipping within greater-rng rims, so configuration of compared derts is always 2x2
    '''
    i__ = dert__[0]  # pixel intensity, should be separate from i__sum
    # sparse aligned rim arrays:
    i__topleft = i__[:-1:2, :-1:2]  # also assignment to new_dert__[0]
    i__topright = i__[:-1:2, 1::2]
    i__bottomleft = i__[1::2, :-1:2]
    i__bottomright = i__[1::2, 1::2]
    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__ = ( mask__[:-1:2, :-1:2].astype(int)
                          + mask__[:-1:2, 1::2].astype(int)
                          + mask__[1::2, 1::2].astype(int)
                          + mask__[1::2, :-1:2].astype(int)
                          ) > 1
    else:
        majority_mask__ = None  # returned at the end of function

    d_upleft__ = dert__[1][:-1:2, :-1:2].copy()  # sparse step=2 sampling
    d_upright__= dert__[2][:-1:2, :-1:2].copy()
    rngSkip = 1
    if rng>2: rngSkip *= (rng-2)*2  # *2 for 8x8, *4 for 16x16
    # combined distance and extrapolation coeffs, or separate distance coef: ave * (rave / dist), rave = ave abs d / ave i?
    # compare pixels diagonally:
    d_upright__+= (i__bottomleft - i__topright) * rngSkip
    d_upleft__ += (i__bottomright - i__topleft) * rngSkip

    g__ = np.hypot(d_upright__, d_upleft__)  # match = inverse of abs gradient (variation), recomputed at each comp_r
    ri__ = i__topleft + i__topright + i__bottomleft + i__bottomright

    return idert(i__topleft, d_upleft__, d_upright__, g__), majority_mask__

