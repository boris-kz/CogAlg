import numpy as np
from cmath import phase
from math import hypot

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_comp_g         = 0b00000100  # = not f_inc_rng?

# ************ FUNCTIONS ************************************************************************************************
# -compare_i()
# -construct_input_array()
# -convolve()
# -calc_g_fold_dert()
# -accumulated_d_()
# -calc_angle()
# ***********************************************************************************************************************

'''
Kernel-based version of:

Comparison of input param between derts at range=rng, summing derivatives from shorter + current range comps per pixel
Input is pixel brightness p or gradient g in dert[0] or angle a in dert[1]: g_dert = g, (dy, dx); ga_dert = g, a, (dy, dx)

if fa: compute and compare angle from dy, dx in dert[-1], only for g_dert in 2nd intra_comp of intra_blob
else:  compare input param in dert[fia]: p|g in derts[cyc][0] or angle a in dert[1]

flag ga: i_dert = derts[cyc][fga], both fga and fia are set for current intra_blob forks and potentially recycled
flag ia: i = i_dert[fia]: selects dert[1] for incremental-range comp angle only
'''

def compare_i(P_, _dert___, i__, bounds, indices, flags):    # comparison of input param between derts at range = rng
    # _dert___ in blob ( dert__ in P_ line ( dert_ in P

    rng = _dert___.maxlen
    fa = flags & f_angle

    fga = fa and (flags & f_comp_g)  # why comp_g only?
    fia = fa and (flags & f_inc_rng)  # why fa, it can be for gradient of angle as well as for angle of angle?
    cyc = -rng - 1 + fia

    derts__, i_ = construct_input_array(P_, bounds, flags, cyc, fa, fga, fia)   # construct input array with predetermined shape

    if not flags:  # hypot_g, returns current line derts__
        return derts__, i_

    _dert___.appendleft(derts__)

    if len(_dert___) == 0:                                  # no i__:
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

    if flags:  # not hypot_g()
        start, end = bounds
        calc_a = fa and not fia

        derts__ = []
        i_ = np.empty(shape=(1, end - start), dtype=int)
        for P in P_:
            if not calc_a:
                derts_ = P.derts_
            else:                               # compute angles
                derts_ = [calc_angle(derts) for derts in P.derts_]

            derts__.append((P.x0, derts_))      # unfold into derts__

            index = P.x0 - start
            i_[0, index: index + P.L] = [derts[cyc][fga][fia] for derts in derts_]   # construct input array for comparison

    else:   # do hypot_g():
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


def calc_angle(derts):  # compute a, return derts with angle

    g = derts[-1][0]
    dy, dx = derts[-1][-1]

    a = (dx + dy * 1j) / g
    a_radian = phase(a)

    derts[-1][1].insert(a, a_radian)  # a = dert[1], for i = dert[fia]

    return derts[-1]

    # ---------- calc_angle() end -----------------------------------------------------------------------------------------------