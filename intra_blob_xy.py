'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:

    - comp_r: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_a: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - xy_blobs in low gradient of angle areas: forms edge-orthogonal Ps, evaluated for comp_d, then evaluates their stacks for comp_P
    Each adds a layer of sub_blobs per blob.
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_xy_scheme.png

    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = I, iDy, iDx, G, Dy, Dx, M, S (area), Ly (vertical dimension)
    # I: input, (iDy, iDx): angle of input gradient, G: gradient, (Dy, Dx): vertical and lateral Ds, M: match
    sign,
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, idy, idx, g, dy, dx, m
    # next fork:
    fcr,  # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork?
    fig,  # flag input is gradient
    rdn,  # redundancy to higher layers
    rng,  # comp range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
from collections import deque, defaultdict
from frame_blobs_defs import CDeepBlob
from class_bind import AdjBinder
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_g, comp_r, comp_a
from frame_blobs_imaging import visualize_blobs
from itertools import zip_longest
from utils import pairwise
import numpy as np
from xy_blobs import image_to_blobs

# from comp_P_draft import comp_P_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering


# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def intra_blob(blob, **kwargs):  # recursive input rng+ | der+ cross-comp within blob
    # fig: flag input is g | p, fcr: flag comp over rng+ | der+
    if kwargs.get('render') is not None:  # stop rendering sub-blobs when blob is too small
        if blob.S < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)
    if blob.fca:  # comp_a
        dert__, mask = comp_a(ext_dert__, blob.figa, ext_mask)  # -> xy_blos (comp_d, comp_P)
    else:  # comp_r
        dert__, mask = comp_r(ext_dert__, blob.fig, blob.fcr, ext_mask)  # -> m sub_blobs

    if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
        sub_blobs = cluster_derts(dert__, mask, ave * blob.rdn, blob.fcr, blob.fig, blob.fca, blob.figa, verbose=False, **kwargs)

        # fork params:
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate for intra_blob comp_a | comp_r | xy_blobs:

            G = blob.G;
            adj_G = blob.adj_blobs[2]
            borrow = min(abs(G), abs(adj_G) / 2)  # or adjacent M if negative sign?

            # +Ga, +Gaga, +Gr, +Gagr, +Grr and so on
            if sub_blob.G + borrow > ave * blob.rdn:  # comp_a
                sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                sub_blob.figa = 1
                sub_blob.fig = 1
                blob.sub_layers += intra_blob(sub_blob, **kwargs)

            # +Ma, +Maga, +Magr and so on (root fork of xy_blobs always = comp_a)
            elif blob.fca == 1 and sub_blob.M > ave * blob.rdn:  # xy_blobs
                image_to_blobs(sub_blob.root_dert__, verbose=False, render=False)

            # +Mr, +Mrr and so on
            elif sub_blob.M > ave * blob.rdn:  # comp_r
                sub_blob.rdn = sub_blob.rdn + 1 + 1 / blob.Ls
                sub_blob.fcr = 1
                sub_blob.rng = blob.rng * 2
                sub_blob.fig = blob.fig
                blob.sub_layers += intra_blob(sub_blob, **kwargs)

        spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                          zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]
    return spliced_layers


def cluster_derts(dert__, mask, Ave, fcr, fig, fca, fga, verbose=False, **kwargs):
    if fca:
        if fga:
            crit__ = Ave - dert__[1]  # temporary crit
        else:
            crit__ = Ave - dert__[3]  # temporary crit

    elif fcr:  # comp_r output;  form clustering criterion:
        if fig:
            crit__ = dert__[0] + dert__[6] - Ave  # eval by i + m, accum in rng; dert__[:,:,0] if not transposed
        else:
            crit__ = Ave - dert__[3]  # eval by -g, accum in rng
    else:  # comp_g output
        crit__ = dert__[6] - Ave  # comp_g output eval by m, or clustering is always by m?

    if kwargs.get('use_c'):
        raise NotImplementedError
        (_, _, _, blob_, _), idmap, adj_pairs = flood_fill()
    else:
        blob_, idmap, adj_pairs = flood_fill(dert__,
                                             sign__=crit__ > 0,
                                             verbose=verbose,
                                             mask=mask,
                                             blob_cls=CDeepBlob,
                                             accum_func=accum_blob_Dert)

    assign_adjacents(adj_pairs, CDeepBlob)
    if kwargs.get('render', False):
        visualize_blobs(idmap, blob_,
                        winname=f"Deep blobs (fcr = {fcr}, fig = {fig})")

    return blob_


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
        if derts is not None:
            if type(derts) == tuple:  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
                ext_dert__.append(derts[0][y0e:yne, x0e:xne])
                ext_dert__.append(derts[1][y0e:yne, x0e:xne])
            else:
                ext_dert__.append(derts[y0e:yne, x0e:xne])
        else:
            ext_dert__.append(None)
    ext_dert__ = tuple(ext_dert__)  # change list to tuple

    # extended mask
    ext_mask = np.pad(blob.mask,
                      ((y0 - y0e, yne - yn),
                       (x0 - x0e, xne - xn)),
                      constant_values=True, mode='constant')

    return ext_dert__, ext_mask


def accum_blob_Dert(blob, dert__, y, x):
    if len(dert__) < 10:  # comp_g, comp_r fork
        blob.I += dert__[0][y, x]
        blob.iDy += dert__[1][y, x]
        blob.iDx += dert__[2][y, x]
        blob.G += dert__[3][y, x]
        blob.Dy += dert__[4][y, x]
        blob.Dx += dert__[5][y, x]
        blob.M += dert__[6][y, x]
    else:  # comp_a fork
        blob.I += dert__[0][y, x]
        blob.iDy += dert__[2][y, x]
        blob.iDx += dert__[3][y, x]
        blob.G += dert__[4][y, x]
        blob.Dyy += dert__[5][0][y, x]
        blob.Dyx += dert__[5][1][y, x]
        blob.Dxy += dert__[6][0][y, x]
        blob.Dxx += dert__[6][1][y, x]
        blob.Dx += dert__[3][y, x]
        blob.M += dert__[7][y, x]
