'''
    intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:

    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - P_blobs forms roughly edge-orthogonal Ps, evaluated for comp_d, then evaluates their stacks for comp_P

    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png

    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = S, Ly, I, Dy, Dx, G, M, Day, Dax, Ga, Ma
    # S: area, Ly: vertical dimension
    # I: input; Dy, Dx: renamed Gy, Gx; G: gradient; M: match; Day, Dax, Ga, Ma: angle Dy, Dx, G, M
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
from collections import deque, defaultdict
from frame_blobs_defs import CDeepBlob
from class_bind import AdjBinder
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_r, comp_a
from frame_blobs_imaging import visualize_blobs
from itertools import zip_longest
from utils import pairwise
import numpy as np
from P_blobs import P_blobs

# from comp_P_draft import comp_P_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering


# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob(blob, **kwargs):  # recursive input rng+ | angle cross-comp within blob

    Ave = int(ave * blob.rdn); AveB = int(aveB * blob.rdn)

    if kwargs.get('render') is not None:  # stop rendering sub-blobs when blob is too small
        if blob.S < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)

    if blob.fia:  # input from comp_a -> P_blobs or comp_aga

        dert__, mask = comp_a(ext_dert__, Ave, ext_mask)  # -> ga sub_blobs -> P_blobs (comp_g, comp_P)
        if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, at least one dert in dert__

            # P_blobs eval, tentative:
            if blob.G * (1 - blob.Ga / (4.45 * blob.S)) - AveB > 0:  # max_ga=4.45
                # G reduced by relative Ga value, base G is second deviation or specific borrow value
                crit__ = dert__[3] * (1 - dert__[7] / 4.45) - Ave  # max_ga=4.45, record separately from g and ga?
                # ga is not signed, use Ave_ga?
                blob.fca = 0
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                # includes re-clustering by P_blobs
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    else:  # input from comp_r -> comp_r or comp_a
        if blob.M > AveB:
            if kwargs.get('verbose'):
                print(' ')
                print('r fork')
            dert__, mask = comp_r(ext_dert__, Ave, blob.fia, ext_mask)
            crit__ = dert__[4]  # m__ is inverse deviation of SAD

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.G > AveB:
            if kwargs.get('verbose'):
                print(' '); print('a fork')

            dert__, mask = comp_a(ext_dert__, Ave, ext_mask)  # -> m sub_blobs
            crit__ = dert__[3]  # deviation of g

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    return spliced_layers


def sub_eval(blob, dert__, crit__, mask, **kwargs):
    Ave = ave * blob.rdn; AveB = aveB * blob.rdn

    if blob.fia and not blob.fca:  # terminal P_blobs
        if kwargs.get('verbose'):
            print(' '); print('dert_P fork')

        sub_frame = P_blobs(dert__, mask, crit__, Ave, verbose=kwargs.get('verbose'))
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

        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate sub_blob
            G = blob.G  # or Gr, Grr..
            adj_M = blob.adj_blobs[3]
            borrow_M = min(G, adj_M / 2)

            if blob.fia and blob.fca:
                # comp_aga
                Ga = blob.Ga
                adj_Ma = blob.adj_blobs[4]
                borrow_Ma = min(Ga, adj_Ma / 2)
                if borrow_M / (1 - borrow_Ma / (4.45 * blob.S)) > AveB:  # combine G with Ga, need to re-check
                    sub_blob.fia = 1
                    sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                    sub_blob.a_depth += blob.a_depth  # accumulate a depth from blob to sub blob
                    blob.sub_layers += intra_blob(sub_blob, **kwargs)  # comp_aga, not correct, need to fix nested day and dax
            else:
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
    for derts in blob.root_dert__:
        ext_dert__.append(derts[y0e:yne, x0e:xne])
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

        blob.Dyy += dert__[5][0][y, x]
        blob.Dyx += dert__[5][1][y, x]
        blob.Dxy += dert__[6][0][y, x]
        blob.Dxx += dert__[6][1][y, x]
        blob.Ga += dert__[7][y, x]
        blob.Ma += dert__[8][y, x]


