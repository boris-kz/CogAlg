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
from frame_blobs import assign_adjacents, flood_fill, CBlob
from intra_comp import comp_r, comp_a
from draw_frame_blobs import visualize_blobs
from itertools import zip_longest
from comp_slice_ import *
from segment_by_direction import segment_by_direction

# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
ave_G = 50  # 
ave_vG = 50  # cost / blob loop: fixed syntactic overhead
pcoef = 2  # ave_comp_slice / ave: relative cost of p fork;  no ave_ga = .78, ave_ma = 2: no eval for comp_aa..
# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob_root(root_blob, render, verbose):  # recursive evaluation of cross-comp slice| range| angle per blob

    deep_blobs = []  # for visualization
    spliced_layers = []  # to extend root_blob sublayers

    for blob in root_blob.sublayers[0]:  # print('Processing blob number ' + str(bcount))

        blob.prior_forks = root_blob.prior_forks.copy()  # increment forking sequence: g -> r|a, a -> p
        blob.root_dert__= root_blob.dert__
        blob_height = blob.box[1] - blob.box[0];  blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions: Ly, Lx
            if root_blob.fBa:
                # p fork in angle blobs
                if blob.G * blob.Ga > ave_G * (blob.rdn+1) * pcoef:  # updated rdn
                    blob.fBa = 0; blob.rdn = root_blob.rdn+1  # double the costs
                    # comp_slice root:
                    segment_by_direction(blob, verbose=True)
                    blob.prior_forks.extend('p')
                    if verbose: print('\nslice_blob fork\n')
                    if render and blob.A < 100: deep_blobs.append(blob)
            else:
                vG = blob.G - ave_G  # deviation of gradient, from ave per blob
                vvG = abs(vG) - ave_vG * blob.rdn  # 2nd deviation of gradient, from fixed costs of if "new_dert__" loop below
                '''
                from both: combined max rdn = blob.rdn+1, adjust to actual after flood_fill: rdn -= 1 - (1 / len(sub_blob_))
                vvG = 0 maps to max G for comp_r if vG < 0, and to min G for comp_a if vG > 0
                '''
                if blob.sign:  # sign of pixel-level g, which corresponds to sign of blob vG, so we don't need the later
                    if vvG > 0:  # below-average G, eval for comp_r
                        # root values for sub_blobs:
                        blob.fBa = 0; blob.rng = root_blob.rng + 1; blob.rdn = root_blob.rdn + 1
                        ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
                        # comp_r 4x4:
                        new_dert__, new_mask__ = comp_r(ext_dert__, blob.rng, ext_mask__)
                        sign__ = ave * (blob.rdn + 1) - new_dert__[3] > 0  # m__ = ave - g__
                        blob.prior_forks.extend('r')
                        if verbose: print('\na fork\n')
                        if render and blob.A < 100: deep_blobs.append(blob)

                elif vvG > 0:  # above-average G, eval for comp_a
                    # root values for sub_blobs:
                    blob.fBa = 1; blob.rdn = root_blob.rdn+1
                    ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
                    # comp_a 2x2:
                    new_dert__, new_mask__ = comp_a(ext_dert__, ext_mask__)  # no vgr * vga: deviations can't be combined in a product
                    sign__ = ave * (blob.rdn+2) * pcoef - new_dert__[3] * new_dert__[9] > 0  # val_comp_slice_, rdn+2: -2 gs: gr * ga
                    blob.prior_forks.extend('a')
                    if verbose: print('\na fork\n')
                    if render and blob.A < 100: deep_blobs.append(blob)
                '''
                Separate ave_Gs may select comp_r and comp_a without vvG, then blobs may be processed by both or neither of the forks:
                if ave_G_r > ave_G_a: overlapped part of blob.G spectrum is processed by both forks,
                if ave_G_r < ave_G_a: the gap part of blob.G spectrum is processed by neither fork
                '''
                if "new_dert__" in locals() and new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:  # min Ly and Lx, dert__>=1
                    # form sub_blobs:
                    sub_blobs, idmap, adj_pairs = flood_fill(new_dert__, sign__, verbose=False, mask__=new_mask__.fill(False), blob_cls=CBlob)
                    assign_adjacents(adj_pairs, CBlob)
                    del new_dert__
                    if render:
                        visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (froot_Ba = {blob.fBa}, froot_Ba = {blob.prior_forks[-1] == 'a'})")

                    blob.sublayers = [sub_blobs]  # sublayers[0]
                    blob.sublayers += intra_blob_root(blob, render, verbose)  # recursive evaluation of cross-comp slice| range| angle per blob

                    spliced_layers = [spliced_layers + sublayers for spliced_layers, sublayers in
                                      zip_longest(spliced_layers, blob.sublayers, fillvalue=[])]
    if verbose:
        print_deep_blob_forking(deep_blobs); print("\rFinished intra_blob")

    return spliced_layers


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
        if type(dert) == list:  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
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