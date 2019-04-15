import numpy as np
import numpy.ma as ma

from filters import get_filters
get_filters(globals())  # imports all filters at once

def comp_gradient(blob):  # compare g within sub blob, a component of intra_blob

    dert__ = ma.empty(shape=blob.dert__.shape, dtype=int)   # initialize dert__
    g__ = ma.array(blob.dert__[:, :, 3], mask=~blob.map)    # apply mask = ~map

    dy__ = g__[2:, 1:-1] - g__[:-2, 1:-1]  # vertical comp between rows -> dy, (1:-1): first and last column are discarded
    dx__ = g__[1:-1, 2:] - g__[1:-1, :-2]  # lateral comp between columns -> dx, (1:-1): first and last row are discarded
    gg__ = np.hypot(dy__, dx__) - ave      # deviation of gradient

    # pack all derts into dert__
    dert__[:, :, 0] = g__
    dert__[1:-1, 1:-1, 1] = dy__  # first row, last row, first column and last-column are discarded
    dert__[1:-1, 1:-1, 2] = dx__
    dert__[1:-1, 1:-1, 3] = gg__

    blob.new_dert__[0] = dert__  # pack dert__ into blob
    return 1  # comp rng


def hypot_g_map(blob):  # redefine master blob by reduced g and increased ave * 2: variable costs of comp_angle

    mask = ~blob.map[:, :, np.newaxis].repeat(4, axis=2)  # stack map 4 times to fit the shape of dert__: (width, height, number of params)
    blob.new_dert__[0] = ma.array(blob.dert__, mask=mask)  # initialize dert__ with mask for selective comp

    # redefine g = hypot(dx, dy), ave * 2 assuming that added cost of angle calc = cost of hypot_g calc
    blob.new_dert__[0][:, :, 3] = np.hypot(blob.new_dert__[0][:, :, 1], blob.new_dert__[0][:, :, 2]) - ave * 2

    return 1  # comp rng

    # ---------- hypot_g() end -----------------------------------------------------------------------------------

def overlap(blob, box, map):  # returns number of overlapping pixels between blob.map and map

    y0, yn, x0, xn = blob.box
    _y0, _yn, _x0, _xn = box

    olp_y0 = max(y0, _y0)
    olp_yn = min(yn, _yn)
    if olp_yn - olp_y0 <= 0:  # no overlapping y coordinate span
        return 0
    olp_x0 = max(x0, _x0)
    olp_xn = min(xn, _xn)
    if olp_xn - olp_x0 <= 0:  # no overlapping x coordinate span
        return 0

    # master_blob coordinates olp_y0, olp_yn, olp_x0, olp_xn are converted to local coordinates before slicing:

    map1 = box.map[(olp_y0 - y0):(olp_yn - y0), (olp_x0 - x0):(olp_xn - x0)]
    map2 = map[(olp_y0 - _y0):(olp_yn - _y0), (olp_x0 - _x0):(olp_xn - _x0)]

    olp = np.logical_and(map1, map2).sum()  # compute number of overlapping pixels
    return olp

'''
for box, map in map_:  # of higher-value blobs in the layer, incrementally nested, with root_blobs per blob?

    olp = overlap(blob, box, map)  # or no lateral overlap between xblobs?
    rdn += 1 * (olp / blob.Derts[-1][-1])  # rdn += 1 * (olp / G):
    # redundancy to higher and stronger-branch overlapping blobs, * specific branch cost ratio? 
'''