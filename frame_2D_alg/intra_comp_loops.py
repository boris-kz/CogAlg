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

''' 
    |--(clockwise)--+  |--(clockwise)--+
    YCOEF: -1  -2  -1  ¦   XCOEF: -1   0   1  ¦
            0       0  ¦          -2       2  ¦
            1   2   1  ¦          -1   0   1  ¦
'''
max = 55



def unpack_dert(dert__):

    mask__ = dert__[0].mask.astype('int').tolist()
    list_dert__ = [mask__]

    for param in dert__:
        param.mask = ma.nomask
        param = param.tolist()
        list_dert__.append(param)

    return list_dert__


def comp_sin_cos(g, dy, dx):
    # if element is unmasked and denominator is not 0
    if g != 0:
        sin = (dy / g)
        cos = (dx / g)

    # if element is unmasked but denominator is 0
    elif g == 0:
        if dy == 0:
            sin = 0
        else:
            sin = max

        if dx == 0:
            cos = 0
        else:
            cos = max

    return sin, cos


def diff_cos(sin0, cos0, sin1, cos1, sin2, cos2, sin3, cos3):
    cos_da0 = (sin0 * cos0) + (sin2 * cos2)  # top left to bottom right
    cos_da1 = (sin1 * cos1) + (sin3 * cos3)  # top right to bottom left

    return cos_da0, cos_da1


def decompose_difference(g1, g2, g3, g4, cos0, cos1):
    '''g3__, g2__, g0__ g1__ for dgy
       g1__, g2__, g0__ g3__ for dgx'''

    dec_diff = ((g1 + g2) - (g3 * cos0 + g4 * cos1))

    return dec_diff


def match_comp(g0, g1, g2, g3, cos0, cos1):
    # g match = min(g, _g) *cos(da)
    mg0 = min(g0, g2) * cos0
    mg1 = min(g1, g3) * cos1
    mg = mg0 + mg1

    return mg


def comp_g_loop(dert__):  # cross-comp of g in 2x2 kernels, between derts in ma.stack dert__

    if isinstance(dert__, np.ndarray):
        dert__ = unpack_dert(dert__)

    # remove derts of incomplete kernels
    dert__ = shape_check(dert__)

    g__, dy__, dx__ = dert__[4:7]
    default_mask = dert__[0]

    # initialise the lists of new parametes
    dgy__ = []
    dgx__ = []
    gg__ = []
    mg__ = []

    for y in range((len(g__) - 1)):
        # create lists for each sin and cos row to add into final list
        dgy_row = []
        dgx_row = []
        gg_row = []
        mg_row = []

        for x in range(len(g__[y]) - 1):

            # check for masking:
            if default_mask[y][x] == 1:
                dgy = dgx = gg = mg = 0

            # if the element is real
            else:
                # computing cos and sin
                sin0, cos0 = comp_sin_cos(g__[y][x],
                                          dy__[y][x],
                                          dx__[y][x])

                sin1, cos1 = comp_sin_cos(g__[y][x + 1],
                                          dy__[y][x + 1],
                                          dx__[y][x + 1])

                sin2, cos2 = comp_sin_cos(g__[y + 1][x + 1],
                                          dy__[y + 1][x + 1],
                                          dx__[y + 1][x + 1])

                sin3, cos3 = comp_sin_cos(g__[y + 1][x],
                                          dy__[y + 1][x],
                                          dx__[y + 1][x])
                # computing cosine difference
                cos_da0, cos_da1 = diff_cos(sin0, cos0, sin1, cos1, sin2, cos2, sin3, cos3)

                # y-decomposed cosine difference between gs
                dgy = decompose_difference(g__[y + 1][x], g__[y + 1][x + 1], g__[y][x],
                                           g__[y][x + 1], cos0, cos1)
                # x-decomposed cosine difference between gs
                dgx = decompose_difference(g__[y][x + 1], g__[y + 1][x + 1], g__[y][x],
                                           g__[y + 1][x], cos0, cos1)
                # gradient of gradient
                gg = hypot(dgy,  dgx)
                # match computation
                mg = match_comp(g__[y][x], g__[y][x + 1], g__[y + 1][x + 1], g__[y + 1][x],
                                cos_da0, cos_da1)

            # pack computed values in row
            dgy_row.append(dgy)
            dgx_row.append(dgx)
            gg_row.append(gg)
            mg_row.append(mg)

        # remove last column from every row of input parameters
        g__[y].pop()
        dy__[y].pop()
        dx__[y].pop()

        # add a new row into list of rows
        dgy__.append(dgy_row)
        dgx__.append(dgx_row)
        gg__.append(gg_row)
        mg__.append(mg_row)

    # remove last row from input parameters
    g__.pop()
    dy__.pop()
    dx__.pop()

    return [default_mask,
            g__,
            dy__,
            dx__,
            gg__,
            dgy__,
            dgx__,
            mg__]


def comp_r_loop(dert__, fig, root_fcr):

    if isinstance(dert__, np.ndarray):
        dert__ = unpack_dert(dert__)

    i__ = dert__[1]  # i is ig if fig else pixel
    idy__ = dert__[2]
    idx__ = dert__[3]
    default_mask = dert__[0]

    i__center = []
    idy__center = []
    idx__center = []
    g__ = []
    dy__ = []
    dx__ = []
    new_m__ = []

    if root_fcr:  # root fork is comp_r, accumulate derivatives:
        dy__, dx__, m__ = dert__[5:8]


    for y in range(0, len(i__ - 2), 2):
        i_cent_row = []
        idy_cent_row = []
        idx_cent_row = []
        g_row = []
        dy_row = []
        dx_row = []
        m_row = []

        for x in range(0, len(i__[y]) - 2, 2):

            # check for masking:
            if default_mask[y][x] == 1:
                i_center = idy_center = idx_center = m = g = dy = dx = 0

            else:
                i_center = i__[y + 1][x + 1]
                idy_center = idy__[y + 1][x + 1]
                idx_center = idx__[y + 1][x + 1]

                if root_fcr:
                    m = m__[y][col]
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

                    '''inverse match = SAD, more precise measure of variation than g, direction-invariant)'''
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
                    a_sin0, a_cos0 = comp_sin_cos(i_center, idy_center, idx_center)

                    # topleft
                    a_sin1, a_cos1 = comp_sin_cos(i__[y][x],
                                                  idy__[y][x], idx__[y][x])
                    # top
                    a_sin2, a_cos2 = comp_sin_cos(i__[y][x + 1],
                                                  idy__[y][x + 1], idx__[y][x + 1])
                    # topright
                    a_sin3, a_cos3 = comp_sin_cos(i__[y][x + 2],
                                                  idy__[y][x + 2], idx__[y][x + 2])
                    # right
                    a_sin4, a_cos4 = comp_sin_cos(i__[y + 1][x  + 2],
                                                  idy__[y + 1][x  + 2], idx__[y + 1][x  + 2])
                    # bottomright
                    a_sin5, a_cos5 = comp_sin_cos(i__[y + 2][x + 2],
                                                  idy__[y + 2][x + 2], idx__[y + 2][x + 2])
                    # bottom
                    a_sin6, a_cos6 = comp_sin_cos(i__[y + 2][x + 1],
                                                  idy__[y + 2][x + 1], idx__[y + 2][x + 1])
                    # bottomleft
                    a_sin7, a_cos7 = comp_sin_cos(i__[y + 2][x],
                                                  idy__[y + 2][x], idx__[y + 2][x])
                    # left
                    a_sin8, a_cos8 = comp_sin_cos(i__[y + 1][x],
                                                  idy__[y + 1][x], idx__[y + 1][x])

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
                    dt4 = i_center - i__[y + 1][x  + 2] * cos_da4
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

            i_cent_row.append(i_center)
            idy_cent_row.append(idy_center)
            idx_cent_row.append(idx_center)
            g_row.append(g)
            dy_row.append(dy)
            dx_row.append(dx)
            m_row.append(m)

        i__center.append(i_cent_row)
        idy__center.append(idy_cent_row)
        idx__center.append(idx_cent_row)
        g__.append(g_row)
        dy__.append(dy_row)
        dx__.append(dx_row)
        new_m__.append(m_row)

    return [
        i__center,
        idy__center,
        idx__center,
        g__,
        dy__,
        dx__,
        new_m__]



def shape_check(dert__):
    # remove derts of 2x2 kernels that are missing some other derts

    if dert__[0].shape[0] % 2 != 0:
        dert__ = dert__[:, :-1, :]
    if dert__[0].shape[1] % 2 != 0:
        dert__ = dert__[:, :, :-1]

    return dert__