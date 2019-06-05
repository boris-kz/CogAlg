import numpy as np
from cmath import phase
from math import hypot

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_comp_g         = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -compare_i()
# -construct_input_array()
# -convolve()
# -calc_g_fold_dert()
# -accumulated_d_()
# -calc_a()
# ***********************************************************************************************************************

def compare_i(P_, _dert___, i__, bounds, indices, flags):    # comparison of input param between derts at range = rng
    # _dert___ in blob ( dert__ in P_ line ( dert_ in P
    rng = _dert___.maxlen
    fa = flags & f_angle
    fga = fa and (flags & f_comp_g)
    fia = fa and (flags & f_inc_rng)
    cyc = -rng - 1 + fia

    derts__, i_ = construct_input_array(P_, bounds, flags, cyc, fa, fga, fia)   # construct input array with predetermined shape

    if not flags:           # no flag: hypot_g, return current line derts__
        return derts__, i_

    _dert___.appendleft(derts__)

    if len(_dert___) == 0:                                      # no i__:
        return [], i_                                       # return empty _derts__
    if i__.shape[0] <= rng * 2:                             # incomplete _dert___:
        return [], np.concatenate((i__, i_), axis=0)        # return empty _derts__

    i__ = np.concatenate((i__[1:], i_), axis=0)             # discard top line, append last line i__

    d_ = convolve(i__, kernels[rng], indices, rng)          # convolve i__ with kernels

    if flags & f_inc_rng:               # accumulate with
        d_ += accumulated_d_(_dert___[0])   # derts on rng-higher line

    _derts__ = calc_g_fold_dert(_dert___.pop(), d_, indices, bounds, flags)

    return _derts__ , i__

    # ---------- compare_i() end --------------------------------------------------------------------------------------------

def construct_input_array(P_, bounds, flags, cyc, fa, fga, fia):   # unfold P_

    if flags:   # not for hypot_g
        start, end = bounds
        b_calc_a = fa and not fia

        derts__ = []
        i_ = np.empty(shape=(1, end - start), dtype=int)
        for P in P_:
            if not b_calc_a:
                derts_ = P.derts_
            else:                               # compute angles
                derts_ = [calc_a(derts) for derts in P.derts_]

            derts__.append((P.x0, derts_))      # unfold into derts__

            index = P.x0 - start
            i_[0, index: index + P.L] = [derts[cyc][fga][fia] for derts in derts_]   # construct input array for comparison

    else:           # do hypot_g():
        derts__ = [(P.x0, [[(p,), (int(hypot(dy, dx)), (dy, dx))] for p, g, dy, dx in P.dert_]) for P in P_]   # unfold into derts__
        i_ = None   # no comparison

    return derts__, i_

    # ---------- construct_input_array() end --------------------------------------------------------------------------------

def convolve(a, k, indices, rng):   # apply kernel, return array of dx, dy
    # d_[0, :]: array of dy
    # d_[1, :]: array of dx

    d_ = np.empty((2, a.shape[1]))

    d_[:, indices] = [(a[:, i-rng:i+rng+1] * k).sum(axis=(1, 2)) for i in indices]

    return d_

    # ---------- convolve() end ---------------------------------------------------------------------------------------------

def calc_g_fold_dert(derts__, d_, indices, bounds, flags):   # compute g using array of dx, dy. fold dert
    x0, xn = bounds

    g_ = np.empty((d_.shape[1],))

    g_[indices] = np.hypot(d_[0, indices], d_[1, indices])

    new_derts__ = []

    # append new dert into new_derts
    # ...

    return new_derts__

    # ---------- calc_g_fold_dert() end -------------------------------------------------------------------------------------

def accumulated_d_(derts__):
    return

    # ---------- accumulated_d_() end ---------------------------------------------------------------------------------------


def calc_a(derts):  # compute a for derts return derts appended with angle

    g = derts[-1][0]
    dy, dx = derts[-1][-1]

    a = (dx + dy * 1j) / g
    a_radiant = phase(a)

    return derts + [(a, a_radiant)]

    # ---------- calc_a() end -----------------------------------------------------------------------------------------------