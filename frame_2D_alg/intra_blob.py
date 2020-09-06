'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:

    der+: incremental derivation cross-comp in high-variation edge areas of +vg: positive deviation of gradient triggers comp_g,
    rng+: incremental range cross-comp in low-variation flat areas of +v--vg: positive deviation of negated -vg triggers comp_r.
    Each adds a layer of sub_blobs per blob.
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_2_fork_scheme.png

    Blob structure, for all layers of blob hierarchy:
    root_dert__,
    Dert = I, iDy, iDx, G, Dy, Dx, M, S (area), Ly (vertical dimension)
    # I: input, (iDy, iDx): angle of input gradient, G: gradient, (Dy, Dx): vertical and lateral Ds, M: match
    sign,
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, idy, idx, g, dy, dx, m
    stack_[ stack_params, Py_ [(P_params, dert_)]]: refs down blob formation tree, in vertical (horizontal) order
    # next fork:
    fcr,  # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork?
    fig,  # flag input is gradient
    rdn,  # redundancy to higher layers
    rng,  # comp range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
from collections import deque, defaultdict
from frame_blobs_defs import CDeepBlobs
from class_bind import AdjBinder
from frame_blobs import assign_adjacents, flood_fill
from intra_comp import comp_g, comp_r
from itertools import zip_longest
from utils import pairwise
import numpy as np

# from comp_P_draft import comp_P_blob

# filters, All *= rdn:
ave = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 50  # fixed cost per intra_blob comp and clustering


# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def intra_blob(blob, rdn, rng, fig, fcr, **kwargs):  # recursive input rng+ | der+ cross-comp within blob
    # fig: flag input is g | p, fcr: flag comp over rng+ | der+
    if kwargs.get('render', None) is not None:  # stop rendering sub-blobs when blob is too small
        if blob.Dert['S'] < 100:
            kwargs['render'] = False

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__, ext_mask = extend_dert(blob)
    if fcr:
        dert__, mask = comp_r(ext_dert__, fig, fcr, ext_mask)  # -> m sub_blobs
    else:
        dert__, mask = comp_g(ext_dert__, ext_mask)  # -> g sub_blobs:

    if dert__[0].shape[0] > 2 and dert__[0].shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
        sub_blobs = cluster_derts(dert__, mask, ave * rdn, fcr, fig, **kwargs)
        # fork params:
        blob.fcr = fcr
        blob.fig = fig
        blob.rdn = rdn
        blob.rng = rng
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rdn
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate for intra_blob comp_g | comp_r:

            G = blob.Dert['G'];  adj_G = blob.adj_blobs[2]
            borrow = min(abs(G), abs(adj_G) / 2)  # or adjacent M if negative sign?

            if sub_blob.sign:
                if sub_blob.Dert['M'] - borrow > aveB * rdn:  # M - (intra_comp value lend to edge blob)
                    # comp_r fork:
                    blob.sub_layers += intra_blob(sub_blob, rdn + 1 + 1 / blob.Ls, rng * 2, fig=fig, fcr=1, **kwargs)
                # else: comp_P_
            elif sub_blob.Dert['G'] + borrow > aveB * rdn:  # G + (intra_comp value borrow from flat blob)
                # comp_g fork:
                blob.sub_layers += intra_blob(sub_blob, rdn + 1 + 1 / blob.Ls, rng=rng, fig=1, fcr=0, **kwargs)
            # else: comp_P_

        spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                          zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]
    return spliced_layers


def cluster_derts(dert__, mask, Ave, fcr, fig, **kwargs):
    if fcr:  # comp_r output;  form clustering criterion:
        if fig:
            crit__ = dert__[0] + dert__[6] - Ave  # eval by i + m, accum in rng; dert__[:,:,0] if not transposed
        else:
            crit__ = Ave - dert__[3]  # eval by -g, accum in rng
    else:  # comp_g output
        crit__ = dert__[6] - Ave  # comp_g output eval by m, or clustering is always by m?

    excluded_derts = set(zip(*mask.nonzero()))

    blob_, idmap, adj_pairs = flood_fill(dert__,
                                         sign__=crit__ > 0,
                                         verbose=verbose,
                                         excluded_derts=excluded_derts,
                                         blob_cls=CDeepBlobs,
                                         accum_func=accum_blob_Dert,
                                         **kwargs)

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
    ext_dert__ = [derts[y0e:yne, x0e:xne] if derts is not None else None
                  for derts in blob.root_dert__]

    # TODO: build mask
    mask = None

    return ext_dert__, mask


def accum_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.iDy += dert__[1][y, x]
    blob.iDx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.Dy += dert__[4][y, x]
    blob.Dx += dert__[5][y, x]
    blob.M += dert__[6][y, x]