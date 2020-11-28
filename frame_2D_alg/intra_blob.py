'''
    intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:

    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - P_blob forms roughly edge-orthogonal Ps, evaluated for comp_d, then evaluates their stacks for comp_P
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png

    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = A, Ly, I, Dy, Dx, G, M, Day, Dax, Ga, Ma
    # A: area, Ly: vertical dimension, I: input; Dy, Dx: renamed Gy, Gx; G: gradient; M: match; Day, Dax, Ga, Ma: angle Dy, Dx, G, M
    sign,
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, dy, dx, g, m, day, dax, ga, ma
    # next fork:
    fia,  # flag: input is from comp angle
    fca,  # flag: current fork is comp angle
    rdn,  # redundancy to higher layers
    rng,  # comparison range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''

import numpy as np
from frame_blobs import assign_adjacents, flood_fill, CDeepBlob
from intra_comp import comp_r, comp_a
from frame_blobs_imaging import visualize_blobs
from itertools import zip_longest
from slice_blob import slice_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering

# --------------------------------------------------------------------------------------------------------------
# functions:


def intra_blob(blob, **kwargs):  # recursive input rng+ | angle cross-comp within blob

    Ave = int(ave * blob.rdn); AveB = int(aveB * blob.rdn)

    if kwargs.get('render') is not None:  # stop rendering sub-blobs when blob is too small
        if blob.A < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)

    if blob.fia:  # input from comp_a -> P_blobs

        dert__= tuple([root_dert[blob.box[0]:blob.box[1],blob.box[2]:blob.box[3]] for root_dert in blob.root_dert__])
        mask = blob.mask

        if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, at least one dert in dert__

            # slice_blobs eval, tentative:
            if blob.G * blob.Ma - AveB > 0:  # ma =/ ave_ma: ratio deviation: no independent value and mag is not co-measurable?
                # Ma vs. (1 - blob.Ga / (4.45 * blob.A)), max_ga=4.45
                # G reduced by relative Ga value, base G is second deviation or specific borrow value
                crit__ = dert__[3] * dert__[8] - Ave  # record separately from g and ga?
                '''
                # flip_yx on whole blob
                for blob in frame['blob__']:
                    for stack in blob.stack_:
                        if stack.f_gstack:
                            for istack in stack.Py_:
                                y0 = istack.y0
                                yn = istack.y0 + stack.Ly
                                x0 = min([P.x0 for P in istack.Py_])
                                xn = max([P.x0 + P.L for P in istack.Py_])
                                L_bias = (xn - x0 + 1) / (yn - y0 + 1)  # elongation: width / height, pref. comp over long dimension
                                G_bias = abs(istack.Dy) / abs(istack.Dx)  # ddirection: Gy / Gx, preferential comp over low G
                                if istack.G * L_bias * G_bias > flip_ave:  # y_bias = L_bias * G_bias: projected PM net gain:
                                    flipped_Py_ = flip_yx(istack.Py_)  # rotate stack.Py_ by 90 degree, rescan blob vertically -> comp_slice_
                #                return stack, f_istack  # comp_slice if G + M + fflip * (flip_gain - flip_cost) > Ave_comp_slice?
                        # evaluate for arbitrary-angle rotation here,
                        # to replace flip if both vertical and horizontal dimensions are significantly different from the angle of blob axis.
                        else:
                            y0 = stack.y0
                            yn = stack.y0 + stack.Ly
                            x0 = min([P.x0 for P in stack.Py_])
                            xn = max([P.x0 + P.L for P in stack.Py_])
                            L_bias = (xn - x0 + 1) / (yn - y0 + 1)  # elongation: width / height, pref. comp over long dimension
                            G_bias = abs(stack.Dy) / abs(stack.Dx)  # ddirection: Gy / Gx, preferential comp over low G
                            if stack.G * L_bias * G_bias > flip_ave:  # y_bias = L_bias * G_bias: projected PM net gain:
                                flipped_Py_ = flip_yx(stack.Py_)  # rotate stack.Py_ by 90 degree, rescan blob vertically -> comp_slice_
                '''
                blob.fca = 0
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                # includes re-clustering by P_blobs
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]
    else:
        # input from frame_blobs or comp_r -> comp_r or comp_a
        if blob.M > AveB:
            if kwargs.get('verbose'):
                print(' ')
                print('r fork')
            blob.prior_forks.extend('r')
            dert__, mask = comp_r(ext_dert__, Ave, blob.fia, ext_mask)
            crit__ = dert__[4]  # m__ is inverse deviation of SAD

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.G > AveB:
            if kwargs.get('verbose'):
                print(' '); print('a fork')

            blob.prior_forks.extend('a')

            adert__, mask = comp_a(ext_dert__, Ave, ext_mask)  # -> m sub_blobs
            crit__ = adert__[3]  # deviation of g

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__

                # flatten adert
                dert__ = tuple([adert__[0], adert__[1], adert__[2], adert__[3], adert__[4],
                                adert__[5][0], adert__[5][1], adert__[6][0], adert__[6][1],
                                adert__[7], adert__[8]])
                blob.fia = 1
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    return spliced_layers


def sub_eval(blob, dert__, crit__, mask, **kwargs):
    AveB = aveB * blob.rdn

    if blob.fia and not blob.fca:  # terminal P_blobs
        if kwargs.get('verbose'):
            print(' '); print('dert_P fork')

        blob.prior_forks.extend('p')
        sub_frame = slice_blob(blob, dert__, mask, crit__, AveB, verbose=kwargs.get('verbose'))

        sub_blobs = sub_frame['blob__']

        blob.Ls = len(sub_blobs)  # for visibility and next-fork rd
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

    else:  # comp_r, comp_a
        sub_blobs, idmap, adj_pairs = flood_fill(dert__,
                                                 sign__=crit__ > 0,
                                                 verbose=False,
                                                 mask=mask,
                                                 blob_cls=CDeepBlob,
                                                 accum_func=accum_blob_Dert)
        assign_adjacents(adj_pairs, CDeepBlob)
        if kwargs.get('render', False):
            visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (fca = {blob.fca}, fia = {blob.fia})")


        for i, sub_blob in enumerate(sub_blobs):
            # generate instance of deep blob from flood fill's blobs
            sub_blobs[i] = CDeepBlob(I=sub_blob.I, Dy=sub_blob.Dy, Dx=sub_blob.Dx, G=sub_blob.G, M=sub_blob.M, A=sub_blob.A, box=sub_blob.box, sign=sub_blob.sign,
                                     mask=sub_blob.mask, root_dert__=dert__, adj_blobs=sub_blob.adj_blobs, fopen=sub_blob.fopen, prior_forks = blob.prior_forks.copy())

        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate sub_blob
            G = blob.G  # or Gr, Grr..
            adj_M = blob.adj_blobs[3]
            borrow_M = min(G, adj_M / 2)

            if borrow_M > AveB:
                # comp_a:
                sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                sub_blob.fia = 1
                sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub blob
                blob.sub_layers += intra_blob(sub_blob, **kwargs)

            elif sub_blob.M - borrow_M > AveB:
                # comp_r:
                sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                sub_blob.fia = 0
                sub_blob.rng = blob.rng * 2
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

    # extended mask
    ext_mask = np.pad(blob.mask,
                      ((y0 - y0e, yne - yn),
                       (x0 - x0e, xne - xn)),
                      constant_values=True, mode='constant')

    return ext_dert__, ext_mask


def accum_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dyy += dert__[5][y, x]
        blob.Dyx += dert__[6][y, x]
        blob.Dxy += dert__[7][y, x]
        blob.Dxx += dert__[8][y, x]
        blob.Ga += dert__[9][y, x]
        blob.Ma += dert__[10][y, x]
