'''
Kernel-based version of:
 Comparison of input param between derts at range=rng, summing derivatives from shorter + current range comps per pixel
Input is pixel brightness p or gradient g in dert[0] or angle a in dert[1]: g_dert = g, (dy, dx); ga_dert = g, a, (dy, dx)
 if fa: compute and compare angle from dy, dx in dert[-1], only for g_dert in 2nd intra_comp of intra_blob
else:  compare input param in dert[fia]: p|g in derts[cyc][0] or angle a in dert[1]
 flag ga: i_dert = derts[cyc][fga], both fga and fia are set for current intra_blob forks and potentially recycled
flag ia: i = i_dert[fia]: selects dert[1] for incremental-range comp angle only
'''

import numpy as np

from cmath import phase
from math import hypot
from itertools import chain, starmap

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_hypot_g        = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -compare_i()
# -construct_input_array()
# -convolve()
# -calc_g_fold_dert()
# -accumulated_d_()
# -calc_angle()
# ***********************************************************************************************************************

def compare_i(P_, _dert___, i__, bounds, indices, flags):    # comparison of input param between derts at range = rng
    # _dert___ in blob ( dert__ in P_ line ( dert_ in P
    rng = _dert___.maxlen
    fia = flags & f_angle
    fga = fia and not (flags & f_hypot_g)
    cyc = -rng - 1 + fia
    coords = set(indices[0] + bounds[0])         # for faster checking

    derts__, i_ = construct_input_array(P_, bounds, flags, cyc, fia, fga)   # construct input array with predetermined shape

    if flags & f_hypot_g:           # hypot_g, return current line derts__
        return derts__, i_

    _dert___.appendleft(derts__)

    if i__ == None:                                         # no i__:
        return [], i_                                       # return empty _derts__
    elif i__.shape[0] <= rng * 2:                           # incomplete _dert___:
        return [], np.concatenate((i__, i_), axis=0)        # return empty _derts__

    i__ = np.concatenate((i__[1:], i_), axis=0)             # discard top line, append last line i__

    d_ = convolve(i__, kernels[rng], indices, rng)          # convolve i__ with kernels

    if flags & f_inc_rng:               # accumulate with
        d_[:, indices] += accumulated_d_(_dert___[0], indices, coords, d_)   # derts on rng-higher line

    _derts__ = calc_g_fold_dert(_dert___.popleft(), d_, indices, coords, fia and not fga)

    return _derts__ , i__

    # ---------- compare_i() end --------------------------------------------------------------------------------------------

def construct_input_array(P_, bounds, flags, cyc, fia, fga):   # unfold P_

    if flags & f_hypot_g:   # do hypot_g():
        derts__ = [([[(p,), (int(hypot(dy, dx)), (dy, dx))] for p, g, dy, dx in P.dert_], P.x0) for P in P_]   # unfold into derts__
        i_ = None   # no comparison

    else:                   # not hypot_g
        start, end = bounds
        b_calc_a = fia and not fga

        derts__ = []
        i_ = np.empty(shape=(1, end - start), dtype=int)
        for P in P_:
            if not b_calc_a:
                derts_ = P.derts_
            else:  # compute angles
                derts_ = [calc_angle(derts) for derts in P.derts_]

            derts__.append((derts_, P.x0))  # unfold into derts__

            index = P.x0 - start
            i_[0, index: index + P.L] = [derts[cyc][fga][fia] for derts in
                                         derts_]  # construct input array for comparison

    return derts__, i_

    # ---------- construct_input_array() end --------------------------------------------------------------------------------

def convolve(a, k, indices, rng):   # apply kernel, return array of dx, dy
    # d_[0, :]: array of dy
    # d_[1, :]: array of dx

    d_ = np.empty((2, a.shape[1]))

    d_[:, indices] = [(a[:, i-rng:i+rng+1] * k).sum(axis=(1, 2)) for i in indices]

    return d_

    # ---------- convolve() end ---------------------------------------------------------------------------------------------

def calc_g_fold_dert(derts__, d_, indices, coords, calc_a):   # compute g using array of dx, dy. fold dert
    x_start, x_end = bounds

    g_ = np.empty((d_.shape[1],))

    g_[indices] = np.hypot(d_[0, indices], d_[1, indices]).astype(int)

    # append new dert into new_derts

    new_derts__ = []

    for derts_, x0 in derts__:
        new_derts_ = []
        new_x0 = 16777215  # max int
        for x, derts in enumerate(derts_, start=x0):
            if x in coords:
                new_x0 = min(new_x0, x)
                i = x - x_start
                if calc_a:
                    derts[-1] = (g_[i],) + derts[-1] + (d_[0], d_[1])
                else:
                    derts = derts + [(g_[i], d_[0], d_[1])]

                new_derts_.append(derts)

            elif new_derts_:
                new_derts__.append((new_derts_, new_x0))
                new_derts_ = []
                new_x0 = 16777215  # max int

        if new_derts_:
            new_derts__.append((new_derts_, new_x0))

    return new_derts__

    # ---------- calc_g_fold_dert() end -------------------------------------------------------------------------------------

def accumulated_d_(derts__, indices, coords, shape):    # construct the accumulated d array assuming that rng > 1

    d_ = np.empty(shape)

    d_[:, indices] = np.array([derts[-1][-2:] for x, derts in chain(*starmap(enumerate, derts__)) if x in coords]).swapaxes(0, 1)

    return d_[:, indices]

    # ---------- accumulated_d_() end ---------------------------------------------------------------------------------------


def calc_angle(derts):  # compute a, return derts with angle

    g = derts[-1][0]
    dy, dx = derts[-1][-2:]

    a = (dx + dy * 1j) / g
    a_radiant = phase(a)

    return derts + [(a, a_radiant)]

    # ---------- calc_angle() end -------------------------------------------------------------------------------------------