from copy import deepcopy as dcopy
import numpy as np
from intra_comp import angle_diff


def intra_blob_nested_a(blob, **kwargs):  # recursive input rng+ | angle cross-comp within blob

    Ave = int(ave * blob.rdn)
    AveB = int(aveB * blob.rdn)

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
                dert__ = list(dert__)
                dert__ = (dert__[0], dert__[1], dert__[2], dert__[3], dert__[4],
                          dert__[5][0], dert__[5][1], dert__[6][0], dert__[6][1],  # flatten day and dax, no nested yet
                          dert__[7], dert__[8])
                crit__ = dert__[3] * (1 - dert__[7] / 4.45) - Ave  # max_ga=4.45, record separately from g and ga?
                # ga is not signed, use Ave_ga?
                blob.fca = 0
                sub_eval(blob, dert__, crit__, mask, **kwargs)  # includes re-clustering by P_blobs

            # comp_aga eval, tentative: # why condition is the same as P_blobs eval?
            elif blob.G / (1 - blob.Ga / (4.45 * blob.S)) - AveB < 0:  # max_ga=4.45, init G is 2nd deviation or borrow value
                if kwargs.get('verbose'):
                    print(' ')
                    print('aga fork')
                    print("a depth=" + str(blob.a_depth + 1))

                # G increased by relative Ga value,
                # flatten day and dax?
                blob.a_depth += 1  # increase a depth
                crit__ = dert__[3] / (1 - dert__[7] / 4.45) - Ave  # ~ eval per blob, record separately from g and ga?
                # ga is not signed, use Ave_ga?
                blob.fca = 1
                sub_eval(blob, dert__, crit__, mask, **kwargs)

            spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                              zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

    else:  # input from comp_r -> comp_r or comp_a
        if blob.M > AveB:
            if kwargs.get('verbose'):
                print(' ')
                print('r fork')

            dert__, mask = comp_r(ext_dert__, Ave, blob.fia, ext_mask)
            crit__ = dert__[4]  # m__: inverse deviation of SAD

            if mask.shape[0] > 2 and mask.shape[1] > 2 and False in mask:  # min size in y and x, least one dert in dert__
                sub_eval(blob, dert__, crit__, mask, **kwargs)
                spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                                  zip_longest(spliced_layers, blob.sub_layers, fillvalue=[])]

        elif blob.G > AveB:
            if kwargs.get('verbose'):
                print(' ')
                print('a fork')
                print("a depth=" + str(blob.a_depth + 1))

            blob.a_depth += 1  # increase a depth
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
            print(' ')
            print('dert_P fork')

        sub_frame = P_blobs(dert__, mask, crit__, Ave, verbose=kwargs.get('verbose'))
        sub_blobs = sub_frame['blob__']
        blob.Ls = len(sub_blobs)  # for visibility and next-fork rd
        blob.sub_layers = [sub_blobs]  # 1st layer of sub_blobs


    else:  # comp_r, comp_a, comp_aga
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


def cluster_derts(dert__, crit__, mask, verbose=False, **kwargs):
    # sub_blobs = cluster_derts(dert__, crit__, mask, verbose=False, **kwargs)  # -> m sub_blobs:
    # replaced by flood_fill

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
                        winname=f"Deep blobs (fca = {fca}, fia = {fia})")

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

            params = [y0e, yne, x0e, xne]
            ext_dert__.append(nested(derts, nested_crop, params))

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
    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]

    if blob.a_depth > 0:  # past comp_a fork

        nested(dert__[5][0], nested_accum_blob_Dert, blob.Dyy, y, x)
        nested(dert__[5][1], nested_accum_blob_Dert, blob.Dyx, y, x)
        nested(dert__[6][0], nested_accum_blob_Dert, blob.Dxy, y, x)
        nested(dert__[6][1], nested_accum_blob_Dert, blob.Dxx, y, x)
        nested(dert__[7], nested_accum_blob_Dert, blob.Ga, y, x)
        nested(dert__[8], nested_accum_blob_Dert, blob.Ma, y, x)


def comp_aga(dert__, ave, mask=None):  # prior fork is comp_a, cross-comp of angle in 2x2 kernels

    if mask is not None:
        majority_mask = (mask[:-1, :-1].astype(int) +
                         mask[:-1, 1:].astype(int) +
                         mask[1:, 1:].astype(int) +
                         mask[1:, :-1].astype(int)
                         ) > 1
    else:
        majority_mask = None

    i__ = [dert__[1],dert__[2]] # i = [dy,dx]
    dy__, dx__, g__, m__ = dert__[5:]  # day__,dax__,ga__,ma__ are recomputed
    g__ = nested(g__, replace_zero_nested)  # to avoid / 0

    # compute g + ave
    day__ = calc_a(dy__, g__, ave)  # sin, restore g to abs
    dax__ = calc_a(dx__, g__, ave)  # cos, restore g to abs
    a__ = [day__, dax__]

    # shift directions
    a__topleft = nested(dcopy(a__), shift_topleft) # use deep copy to create a new memory location
    a__topright = nested(dcopy(a__), shift_topright)
    a__botright = nested(dcopy(a__), shift_botright)
    a__botleft = nested(dcopy(a__), shift_botleft)

    # diagonal angle differences:
    sin_da0__, cos_da0__ = nested2(a__topleft, a__botright, angle_diff)
    sin_da1__, cos_da1__ = nested2(a__topright, a__botleft, angle_diff)

    ma1__ = nested2(sin_da0__, cos_da0__, hypot_add1_nested)
    ma2__ = nested2(sin_da0__, cos_da0__, hypot_add1_nested)
    ma__ = nested2(ma1__, ma2__, add_nested)
    # ma = inverse angle match = SAD: covert sin and cos da to 0->2 range

    # negative nested sin_da0
    sin_da0_nested__ = nested(dcopy(sin_da0__), negative_nested)

    # day__ = (-sin_da0__ - sin_da1__), (cos_da0__ + cos_da1__)
    day__ = [nested2(sin_da0_nested__, cos_da0__, subtract_nested),
             nested2(cos_da0__, cos_da1__, add_nested)]
    # angle change in y, sines are sign-reversed because da0 and da1 are top-down, no reversal in cosines

    # dax__ = (-sin_da0__ + sin_da1__), (cos_da0__ + cos_da1__)
    dax__ = [nested2(sin_da0_nested__, cos_da0__, add_nested),
             nested2(cos_da0__, cos_da1__, add_nested)]
    # angle change in x, positive sign is right-to-left, so only sin_da0__ is sign-reversed

    # np.arctan2(*day__)
    arctan_day__ = nested2(day__[0], day__[1], arctan2_nested)
    # np.arctan2(*dax__)
    arctan_dax__ = nested2(dax__[0], dax__[1], arctan2_nested)

    # ga__ = np.hypot(np.arctan2(*day__), np.arctan2(*dax__))
    ga__ = nested2(arctan_day__, arctan_dax__, hypot_nested)
    # angle gradient, a scalar evaluated for comp_aga

    i__ = nested(i__, shift_topleft)  # for summation in Dert
    g__ = nested(g__, shift_topleft)  # for summation in Dert
    m__ = nested(m__, shift_topleft)
    dy__ = nested(dy__, shift_topleft)  # passed on as idy
    dx__ = nested(dx__, shift_topleft)  # passed on as idy

    return (i__, dy__, dx__, g__, m__, day__, dax__, ga__, ma__), majority_mask


# ----------------------------------------------------------------------------
# General purpose functions for nested operation


def nested(element__, function, *args):  # provided function operates on nested variable

    if isinstance(element__, list):
        if len(element__) > 1 and isinstance(element__[0], list):
            for i, element_ in enumerate(element__):
                element__[i] = nested(element_, function, *args)
        else:
            element__ = function(element__, *args)
    else:
        element__ = function(element__, *args)
    return element__


def nested2(element1__, element2__, function):  # provided function operates on 2 nested variables

    element__ = dcopy(element1__)
    if isinstance(element1__[0], list):
        for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
            element__[i] = nested2(element1_, element2_, function)
    else:
        element__ = function(element1__, element2__)
    return element__


# ----------------------------------------------------------------------------
# Single purpose function for nested operation in intra_comp


def calc_a(element1__, element2__, ave):  # nested compute a from gy,gx, g and ave

    element__ = dcopy(element1__)
    if isinstance(element2__[0], list):
        for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
            element__[i] = calc_a(element1_, element2_, ave)
    else:
        if isinstance(element2__, list):
            for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
                element__[i] = [element1_[0] / element2_, element1_[1] / element2_]
        else:
            element__ = [element1__[0] / element2__, element1__[1] / element2__]

    return element__


def shift_topleft(element_):  # shift variable in top left direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, :-1]
    else:
        element_ = element_[:-1, :-1]
    return element_


def shift_topright(element_):  # shift variable in top right direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, 1:]
    else:
        element_ = element_[:-1, 1:]
    return element_


def shift_botright(element_):  # shift variable in bottom right direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, 1:]
    else:
        element_ = element_[1:, 1:]
    return element_


def shift_botleft(element_):  # shift variable in bottom left direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, :-1]
    else:
        element_ = element_[1:, :-1]
    return element_


def negative_nested(element_):  # complement all values in the variable

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = -element
    else:
        element_ = -element_
    return element_


def replace_zero_nested(element_):  # replace all 0 values in the variable with 1

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element[np.where(element == 0)] = 1
            element_[i] = element
    else:
        element_[np.where(element_ == 0)] = 1
    return element_


def hypot_nested(element1, element2):  # hypot of 2 elements

    return [np.hypot(element1[0], element2[0]), np.hypot(element1[1], element2[1])]


def hypot_add1_nested(element1, element2):  # hypot of 2 (elements+1)

    return [np.hypot(element1[0] + 1, element2[0] + 1), np.hypot(element1[1] + 1, element2[1] + 1)]


def add_nested(element1, element2):  # sum of 2 variables

    return [element1[0] + element2[0], element1[1] + element2[1]]


def subtract_nested(element1, element2):  # difference of 2 variables

    return [element1[0] - element2[0], element1[1] - element2[1]]


def arctan2_nested(element1, element2):  # arctan of 2 variables

    return [np.arctan2(element1[0], element2[0]), np.arctan2(element1[1], element2[1])]


# ----------------------------------------------------------------------------
# Single purpose function for nested operation in intra_blob


def nested_crop(element_, *args):  # crop element based on coordinates in box

    y0e = args[0][0]
    yne = args[0][1]
    x0e = args[0][2]
    xne = args[0][3]

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[y0e:yne, x0e:xne]
    else:
        element_ = element_[y0e:yne, x0e:xne]

    return element_


def nested_accum_blob_Dert(element_, *args):  # accumulate parameters based on the param value and their x,y coordinates

    param = args[0]
    y = args[1]
    x = args[2]

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i][y, x] += param

    else:
        element_[y, x] += param

    return element_
