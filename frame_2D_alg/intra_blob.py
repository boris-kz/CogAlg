'''
    Intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - vectorize_root: forms roughly edge-orthogonal Ps, evaluated for rotation, comp_slice, etc.
'''
import numpy as np
from scipy.signal import convolve2d
from itertools import zip_longest
from frame_blobs import assign_adjacents, flood_fill, Tdert
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
        # increment forking sequence: g -> r|v
        # <--- r fork
        if (blob.sign and blob.G < aveR * blob.rdn):  # below-average G, eval for comp_r
            ret = comp_r(blob)  # return None if blob is too small
            if ret is not None: # if True, proceed to form new fork
                blob.rdn = root_blob.rdn + 1.5  # sub_blob root values
                if verbose: print('fork: r')
                new_der__t, new_mask__, fork_ibox = ret  # unpack comp_r output
                sign__ = ave * (blob.rdn + 1) - new_der__t.g > 0  # m__ = ave - g__
                fork_data = 'r', fork_ibox, new_der__t, sign__, new_mask__ # for flood_fill
                # form sub_blobs:
                sub_blobs, idmap, adj_pairs = flood_fill(blob, fork_data, verbose=verbose)
                '''
                adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
                sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
                + if sub_blob_val > alt_blob_val, else -?  
                adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
                blob.rdn += adj_rdn
                for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
                '''
                assign_adjacents(adj_pairs)

                sublayers = blob.rlayers
                sublayers += [sub_blobs]  # next level sub_blobs, then add deeper layers of sub_blobs:
                sublayers += intra_blob_root(blob, render, verbose)  # recursive eval cross-comp per blob
                spliced_layers[:] += [spliced_layer + sublayer for spliced_layer, sublayer in
                                      zip_longest(spliced_layers, sublayers, fillvalue=[])]
        # ---> end r fork
        # <--- v fork
        if not blob.sign and blob.G > aveG * blob.rdn:  # above-average G, vectorize blob
            blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn, sub_blob root values
            blob.prior_forks += 'v'
            if verbose: print('fork: v')
            vectorize_root(blob, verbose=verbose)
        # ---> end v fork
    return spliced_layers


def comp_r(blob):
    # increment rng and compute kernel
    rng = blob.rng + 1
    ky, kx, km = compute_kernel(rng)

    # expand ibox to reduce shrink if possible
    y0, x0, yn, xn = blob.ibox
    y0e, x0e, yne, xne = eibox = blob.ibox.expand(rng, *blob.i__.shape) # 'e' stands for 'expanded'
    pad = ((y0-y0e, yne-yn), (x0-x0e, xne-xn))  # pad of emask__: expanded size per side

    # compute majority mask, at this point, mask__ can't be None
    emask__ = np.pad(blob.mask__, pad, 'constant', constant_values=False).astype(int)
    if emask__.shape[0] < 2+2*rng or emask__.shape[1] < 2+2*rng: # blob is too small
        return None

    majority_mask__ = convolve2d(emask__, km, mode='valid') > 1

    if not majority_mask__.any():  # no dert
        return None

    # fork is valid, update blob's rng
    blob.rng = rng

    # compute shrink box of der__t, if shrink = 0 on all size, sdbox size is the same as blob.ibox
    fork_ibox = eibox.shrink(rng)  # fork_ibox, with size = that of der__t after comparison
    sdbox = blob.ibox.box2sub_box(fork_ibox)    # shrinked der__t box

    dy__, dx__, g__ = blob.der__t   # unpack der__t
    i__ = blob.i__[eibox.slice()]   # expanded i__ for comparison
    # compare opposed pairs of rim pixels, project onto x, y:
    new_dy__ = dy__[sdbox.slice()] + convolve2d(i__, ky, mode='valid')
    new_dx__ = dx__[sdbox.slice()] + convolve2d(i__, kx, mode='valid')
    new_g__ = np.hypot(new_dy__, new_dx__)  # gradient, recomputed at each comp_r

    return Tdert(new_dy__, new_dx__, new_g__), majority_mask__, fork_ibox


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
    km = np.ones((ksize, ksize), dtype=int)  # kernel of mask
    sqr_dist = x*x + y*y  # squared distance
    sqr_dist[rng, rng] = 1  # avoid division by 0
    coeff = k / sqr_dist # kernel coefficient
    coeff[ks.mc] = 0  # take the rim
    km[ks.mc] = 0  # take the rim

    return (*coeff, km)