from math import hypot
import numpy as np
import numpy.ma as ma

# Sobel coefficients to decompose ds into dy and dx:

YCOEFs = [[1, -2, -1],
          [0,  1,  0],
          [1,  2,  1]]

XCOEFs = [[-1, 0, 1],
          [-2, 1, 2],
          [-1, 0, 1]]

max = 55

# change dert__ into list
# where length of list = length of y
# and number of element in list = length of x
def dert_lists(dert__):

    mask__ = dert__[0].mask.astype('int').tolist()
    list_dert__ = [mask__]

    for param in dert__:
        param.mask = ma.nomask
        param = param.tolist()
        list_dert__.append(param)

    return list_dert__


def sin_cos(g, dy, dx):

    if g != 0:
        sin = (dy / g)
        cos = (dx / g)
    else:
        if dy == 0: sin = 0
        else:       sin = max
        if dx == 0: cos = 0
        else:       cos = max

    return sin, cos


def comp_g(dert__):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__

    if isinstance(dert__, np.ndarray):
        dert__ = dert_lists(dert__)

    dert__ = shape_check_list(dert__)  # remove derts of incomplete kernels

    g__, dy__, dx__ = dert__[4:7]
    mask__ = dert__[0] # index 0 = mask, created in dert_lists

    dgy__, dgx__, gg__, mg__ = [],[],[],[]

    for y in range((len(g__) - 1)):
        dgy_, dgx_, gg_, mg_ = [],[],[],[]

        for x in range(len(g__[y]) - 1):
            dgy, dgx, gg, mg = [],[],[],[]
            '''
            no dgy = dgx = gg = mg = 0:  masked values should be skipped, here and in the future
            '''
            if mask__[y][x] == 0:

                sin0, cos0 = sin_cos(g__[y][x],     dy__[y][x],     dx__[y][x])
                sin1, cos1 = sin_cos(g__[y][x+1],   dy__[y][x+1],   dx__[y][x+1])
                sin2, cos2 = sin_cos(g__[y+1][x+1], dy__[y+1][x+1], dx__[y+1][x+1])
                sin3, cos3 = sin_cos(g__[y+1][x],   dy__[y+1][x],   dx__[y+1][x])

                # compute cosine difference

                cos_da0 = (sin0 * cos0) + (sin2 * cos2)  # top left to bottom right
                cos_da1 = (sin1 * cos1) + (sin3 * cos3)  # top right to bottom left

                # y-decomposed cosine difference between gs
                dgy = ((g__[y+1][x] + g__[y+1][x+1]) - (g__[y][x] * cos_da0 + g__[y][x+1] * cos_da1))

                # x-decomposed cosine difference between gs
                dgx = ((g__[y][x+1] + g__[y+1][x+1]) - (g__[y][x] * cos_da0 + g__[y+1][x] * cos_da1))

                # gradient of gradient
                gg = hypot(dgy,  dgx)

                mg0 = min(g__[y][x], g__[y+1][x+1]) * cos0  # g match = min(g, _g) *cos(da)
                mg1 = min(g__[y][x+1], g__[y+1][x]) * cos1
                mg = mg0 + mg1

                # pack computed values in row
                dgy_.append(dgy)
                dgx_.append(dgx)
                gg_.append(gg)
                mg_.append(mg)


        # remove last column from every row of input parameters
        g__[y].pop()
        dy__[y].pop()
        dx__[y].pop()
        mask__[y].pop() # we need remove last column for mask as well to enable a consistent dimension


        # add a new row into list of rows
        dgy__.append(dgy_)
        dgx__.append(dgx_)
        gg__.append(gg_)
        mg__.append(mg_)

    # remove last row from input parameters
    g__.pop()
    dy__.pop()
    dx__.pop()
    mask__.pop() # we need remove last row for mask as well to enable a consistent dimension

    return [mask__, g__, dy__, dx__, gg__, dgy__, dgx__, mg__]


def comp_r_loop(dert__, fig, root_fcr):

    if isinstance(dert__, np.ndarray):
        dert__ = dert_lists(dert__)

    i__ = dert__[1]  # i is ig if fig else pixel
    idy__ = dert__[2]
    idx__ = dert__[3]
    mask__ = dert__[0]

    i__center, idy__center, idx__center = [],[],[]
    g__, dy__, dx__, new_m__ = [],[],[],[]

    if root_fcr:  # root fork is comp_r, accumulate derivatives:
        dy__, dx__, m__ = dert__[5:8]

    for y in range(0, len(i__)- 2, 2):   # len(i__) - 2 instead of len(i__-2)?

        i_cent_, idy_cent_, idx_cent_ = [],[],[]
        g_, dy_, dx_, m_ = [],[],[],[]

        for x in range(0, len(i__[y]) - 2, 2):
            '''
            masked values should be skipped, here and in the future
            '''
            if mask__[y][x] == 0:

                i_center = i__[y + 1][x + 1]
                idy_center = idy__[y + 1][x + 1]
                idx_center = idx__[y + 1][x + 1]

                if root_fcr:
                    m = m__[y][x]
                else:
                    m = 0

                if not fig:

                    dt1 = i__[y][x]     - i__[y + 2][x + 2]    # i__topleft - i__bottomright
                    dt2 = i__[y][x + 1] - i__[y + 2][x + 1]    # i__top - i__bottom
                    dt3 = i__[y][x + 2] - i__[y + 2][x]        # i__topright - i__bottomleft
                    dt4 = i__[y + 1][x  + 2] - i__[y + 1][x]   # i__right - i__left

                    # dx and dy computation
                    dy = dt1 * YCOEFs[0][1] + dt2 * YCOEFs[0][1] + dt3 * YCOEFs[0][2] + dt4 * YCOEFs[1][2]
                    dx = dt1 * XCOEFs[0][1] + dt2 * XCOEFs[0][1] + dt3 * XCOEFs[0][2] + dt4 * XCOEFs[1][2]

                    # gradient
                    g = hypot(dy, dx)
                    '''
                    inverse match = SAD, more precise measure of variation than g, direction-invariant)
                    '''
                    m += (abs(i_center - i__[y][x]) +
                          abs(i_center - i__[y][x + 1]) +
                          abs(i_center - i__[y][x + 2]) +
                          abs(i_center - i__[y + 1][x  + 2]) +
                          abs(i_center - i__[y + 2][x + 2]) +
                          abs(i_center - i__[y + 2][x + 1]) +
                          abs(i_center - i__[y + 2][x]) +
                          abs(i_center - i__[y + 1][x]))

                else: # fig is TRUE, compare angle and then magnitude of 8 center-rim pairs
                    # center
                    a_sin0, a_cos0 = sin_cos(i_center, idy_center, idx_center)
                    # topleft
                    a_sin1, a_cos1 = sin_cos(i__[y][x], idy__[y][x], idx__[y][x])
                    # top
                    a_sin2, a_cos2 = sin_cos(i__[y][x + 1], idy__[y][x + 1], idx__[y][x + 1])
                    # topright
                    a_sin3, a_cos3 = sin_cos(i__[y][x + 2], idy__[y][x + 2], idx__[y][x + 2])
                    # right
                    a_sin4, a_cos4 = sin_cos(i__[y + 1][x  + 2], idy__[y + 1][x  + 2], idx__[y + 1][x  + 2])
                    # bottomright
                    a_sin5, a_cos5 = sin_cos(i__[y + 2][x + 2], idy__[y + 2][x + 2], idx__[y + 2][x + 2])
                    # bottom
                    a_sin6, a_cos6 = sin_cos(i__[y + 2][x + 1], idy__[y + 2][x + 1], idx__[y + 2][x + 1])
                    # bottomleft
                    a_sin7, a_cos7 = sin_cos(i__[y + 2][x], idy__[y + 2][x], idx__[y + 2][x])
                    # left
                    a_sin8, a_cos8 = sin_cos(i__[y + 1][x], idy__[y + 1][x], idx__[y + 1][x])

                    # differences between center dert angle and rim dert angle

                    cos_da1 = (a_sin0 * a_cos0) + (a_sin1 * a_cos1)
                    cos_da2 = (a_sin0 * a_cos0) + (a_sin2 * a_cos2)
                    cos_da3 = (a_sin0 * a_cos0) + (a_sin3 * a_cos3)
                    cos_da4 = (a_sin0 * a_cos0) + (a_sin4 * a_cos4)
                    cos_da5 = (a_sin0 * a_cos0) + (a_sin5 * a_cos5)
                    cos_da6 = (a_sin0 * a_cos0) + (a_sin6 * a_cos6)
                    cos_da7 = (a_sin0 * a_cos0) + (a_sin7 * a_cos7)
                    cos_da8 = (a_sin0 * a_cos0) + (a_sin8 * a_cos8)

                    # cosine matches per direction

                    m += (min(i_center, i__[y][x])          * cos_da1) +  \
                         (min(i_center, i__[y][x + 1])      * cos_da2) +  \
                         (min(i_center, i__[y][x + 2])      * cos_da3) +  \
                         (min(i_center, i__[y + 1][x  + 2]) * cos_da4) +  \
                         (min(i_center, i__[y + 2][x + 2])  * cos_da5) +  \
                         (min(i_center, i__[y + 2][x + 1])  * cos_da6) +  \
                         (min(i_center, i__[y + 2][x])      * cos_da7) +  \
                         (min(i_center, i__[y + 1][x])      * cos_da8)

                    # cosine differences per direction

                    dt1 = i_center - i__[y][x]          * cos_da1
                    dt2 = i_center - i__[y][x + 1]      * cos_da2
                    dt3 = i_center - i__[y][x + 2]      * cos_da3
                    dt4 = i_center - i__[y + 1][x + 2]  * cos_da4
                    dt5 = i_center - i__[y + 2][x + 2]  * cos_da5
                    dt6 = i_center - i__[y + 2][x + 1]  * cos_da6
                    dt7 = i_center - i__[y + 2][x]      * cos_da7
                    dt8 = i_center - i__[y + 1][x]      * cos_da8

                    dy = dt1 * YCOEFs[0][0] + dt2 * YCOEFs[0][1] + \
                         dt3 * YCOEFs[0][2] + dt4 * YCOEFs[1][2] + \
                         dt5 * YCOEFs[2][2] + dt6 * YCOEFs[2][1] + \
                         dt7 + YCOEFs[2][0] + dt8 * YCOEFs[1][0]

                    dx = dt1 * XCOEFs[0][0] + dt2 * XCOEFs[0][1] + \
                         dt3 * XCOEFs[0][2] + dt4 * XCOEFs[1][2] + \
                         dt5 * XCOEFs[2][2] + dt6 * XCOEFs[2][1] + \
                         dt7 + XCOEFs[2][0] + dt8 * XCOEFs[1][0]

                    g = hypot(dy, dx)

                i_cent_.append(i_center)
                idy_cent_.append(idy_center)
                idx_cent_.append(idx_center)
                g_.append(g)
                dy_.append(dy)
                dx_.append(dx)
                m_.append(m)


        i__center.append(i_cent_)
        idy__center.append(idy_cent_)
        idx__center.append(idx_cent_)
        g__.append(g_)
        dy__.append(dy_)
        dx__.append(dx_)
        new_m__.append(m_)

    return [ i__center, idy__center, idx__center, g__, dy__, dx__, new_m__]

# should we set condition for shape check where size of x or y is >=2?
# in some cases,length of  y = 1, and if we delete the line y, the length would be 0
# shape check for numpy input
def shape_check(dert__):
    # remove derts of 2x2 kernels that are missing some other derts

    if dert__[0].shape[0] % 2 != 0 and dert__[0].shape[0]>2:
        dert__ = dert__[:, :-1, :]
    if dert__[0].shape[1] % 2 != 0 and dert__[0].shape[1]>2:
        dert__ = dert__[:, :, :-1]

    return dert__

# shape check for list input
def shape_check_list(dert__):
    # remove derts of 2x2 kernels that are missing some other derts


    # if length of y is not multiple of 2 and >2
    if len(dert__[0]) % 2 != 0 and len(dert__[0]) >2:
        # remove last y elemet
        dert__ = [ydert[:-1] for ydert in dert__]

    # if length of x is not multiple of 2 and >2
    if len(dert__[0][0]) % 2 != 0 and len(dert__[0][0])>2:
        # remove last x element
        dert__ = [[xdert[:-1] for xdert in ydert] for ydert in dert__]

    return dert__
