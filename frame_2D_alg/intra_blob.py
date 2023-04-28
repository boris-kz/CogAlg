'''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_angle: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - comp_slice_ forms roughly edge-orthogonal Ps, their stacks evaluated for rotation, comp_d, and comp_slice
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
'''
import numpy as np
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_r, comp_a
from itertools import zip_longest
from comp_slice import comp_slice_root

# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
aveB = 50  # cost / blob loop: fixed syntactic overhead
pcoef = 2  # ave_comp_slice / ave: relative cost of p fork;  no ave_ga = .78, ave_ma = 2: no eval for comp_aa..
aveBa = 1.5  # ave (rblob value / ablob value), replaced by pcoef?
ave_nsub = 4  # ave n sub_blobs per blob: 4x higher costs? or eval costs only, separate clustering ave = aveB?
# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob_root(root_blob, render, verbose, fBa):  # recursive evaluation of cross-comp slice| range| angle per blob

    # deep_blobs = []  # for visualization
    spliced_layers = []
    if fBa: blob_ = root_blob.dlayers[0]
    else:   blob_ = root_blob.rlayers[0]

    for blob in blob_:  # fork-specific blobs, print('Processing blob number ' + str(bcount))

        blob.prior_forks = root_blob.prior_forks.copy()  # increment forking sequence: g -> r|a, a -> p
        blob.root_dert__= root_blob.dert__
        blob_height = blob.box[1] - blob.box[0];  blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions: Ly, Lx
            if root_blob.fBa:
                # comp_slice fork in angle blobs
                # add evaluate splice_blobs(root_blob), normally in frame_bblobs?
                AveB = aveB * (blob.rdn+1)  # comp_slice is doubling the costs, likely higher, adjust per nsub_blobs?
                if blob.G * (1/blob.Ga) > AveB * pcoef:  # value of comp_slice_blob is proportional to angle stability?
                    blob.fBa = 0; blob.rdn = root_blob.rdn+1
                    comp_slice_root(blob, verbose=True)
                    blob.prior_forks.extend('p')
                    if verbose: print('\nslice_blob fork\n')  # if render and blob.A < 100: deep_blobs.append(blob)
            else:
                ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
                ''' 
                comp_r || comp_a, gap or overlap version:
                if aveBa < 1: blobs of ~average G are processed by both forks
                if aveBa > 1: blobs of ~average G are not processed:
                '''
                if blob.G < aveB * blob.rdn:  # below-average G, eval for comp_r
                    # root values for sub_blobs:
                    blob.fBa = 0; blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn
                    # comp_r 4x4:
                    new_dert__, new_mask__ = comp_r(ext_dert__, blob.rng, ext_mask__)
                    sign__ = ave * (blob.rdn+1) - new_dert__[3] > 0  # m__ = ave - g__
                    blob.prior_forks.extend('r')
                    # if min Ly and Lx, dert__>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = cluster_fork_recursive( blob, spliced_layers, new_dert__, sign__, new_mask__, verbose, render, fBa=0)

                if blob.G > aveB*aveBa * blob.rdn:  # above-average G, eval for comp_a
                    # root values for sub_blobs:
                    blob.fBa = 1; blob.rdn = root_blob.rdn + 1.5  # comp cost * fork rdn
                    # comp_a 2x2:
                    new_dert__, new_mask__ = comp_a(ext_dert__, ext_mask__)
                    Ave = ave * (blob.rdn + 1)
                    sign__ = (new_dert__[1] - Ave) + (Ave * pcoef - new_dert__[2]) > 0  # val_comp_slice_= dev_gr + inv_dev_ga
                    blob.prior_forks.extend('a')
                    # if min Ly and Lx, dert__>=1: form, splice sub_blobs:
                    if new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:
                        spliced_layers[:] = cluster_fork_recursive( blob, spliced_layers, new_dert__, sign__, new_mask__, verbose, render, fBa=1)
            '''
            exclusive forks version:
            
            vG = blob.G - ave_G  # deviation of gradient, from ave per blob, combined max rdn = blob.rdn+1:
            vvG = abs(vG) - ave_vG * blob.rdn  # 2nd deviation of gradient, from fixed costs of if "new_dert__" loop below
            # vvG = 0 maps to max G for comp_r if vG < 0, and to min G for comp_a if vG > 0:
            
            if blob.sign:  # sign of pixel-level g, which corresponds to sign of blob vG, so we don't need the later
                if vvG > 0:  # below-average G, eval for comp_r...
                elif vvG > 0:  # above-average G, eval for comp_a...
            '''
    # if verbose: print("\rFinished intra_blob")  # print_deep_blob_forking(deep_blobs)

    return spliced_layers


def cluster_fork_recursive(blob, spliced_layers, new_dert__, sign__, new_mask__, verbose, render, fBa):

    if verbose:
        if fBa: print('fork:', '-'.join(blob.prior_forks))
        else:   print('fork:', '-'.join(blob.prior_forks))
    # form sub_blobs:
    sub_blobs, idmap, adj_pairs = flood_fill(new_dert__, sign__, verbose=False, mask__=new_mask__.fill(False))
    '''
    adjust per average sub_blob, depending on which fork is weaker, or not taken at all:
    sub_blob.rdn += 1 -|+ min(sub_blob_val, alt_blob_val) / max(sub_blob_val, alt_blob_val):
    + if sub_blob_val > alt_blob_val, else -?  
    '''
    adj_rdn = ave_nsub - len(sub_blobs)  # adjust ave cross-layer rdn to actual rdn after flood_fill:
    blob.rdn += adj_rdn
    for sub_blob in sub_blobs: sub_blob.rdn += adj_rdn
    assign_adjacents(adj_pairs)
    # if render: visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (froot_Ba = {blob.fBa}, froot_Ba = {blob.prior_forks[-1] == 'a'})")

    if fBa: sublayers = blob.dlayers
    else:   sublayers = blob.rlayers

    sublayers += [sub_blobs]  # r|a fork- specific sub_blobs, then add deeper layers of mixed-fork sub_blobs:
    sublayers += intra_blob_root(blob, render, verbose, fBa)  # recursive eval cross-comp range| angle| slice per blob

    new_spliced_layers = [spliced_layer + sublayer for spliced_layer, sublayer in
                          zip_longest(spliced_layers, sublayers, fillvalue=[])]
    return new_spliced_layers


def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_dert__[0].shape  # higher dert size
    # set pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended
    # take ext_dert__ from part of root_dert__:
    ext_dert__ = []
    for dert in blob.root_dert__:
        if isinstance(dert,list):  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
            ext_dert__.append(dert[0][y0e:yne, x0e:xne])
            ext_dert__.append(dert[1][y0e:yne, x0e:xne])
        else:
            ext_dert__.append(dert[y0e:yne, x0e:xne])
    ext_dert__ = tuple(ext_dert__)  # change list to tuple
    # extend mask__:
    ext_mask__ = np.pad(blob.mask__,
                        ((y0 - y0e, yne - yn),
                         (x0 - x0e, xne - xn)),
                        constant_values=True, mode='constant')

    return ext_dert__, ext_mask__

def print_deep_blob_forking(deep_layers):

    def check_deep_blob(deep_layer,i):
        for deep_blob_layer in deep_layer:
            if isinstance(deep_blob_layer,list):
                check_deep_blob(deep_blob_layer,i)
            else:
                print('blob num = '+str(i)+', forking = '+'->'.join(deep_blob_layer.prior_forks))

    for i, deep_layer in enumerate(deep_layers):
        if len(deep_layer)>0:
            check_deep_blob(deep_layer,i)