import numpy as np
from cmath import phase, rect

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_comp_g         = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -comp_derts()
# -construct_input_array()
# -convolve()
# -calc_g_fold_dert()
# -accumulated_d_()
# -calc_a()
# ***********************************************************************************************************************

def comp_derts(P_, dbuff, ibuff, bounds, indices, Ave, rng, flags):    # comparison of input param between derts at range = rng
    # dert_buff___ in blob ( dert__ in P_ line ( dert_ in P
    rng = dbuff.maxlen
    fa = flags & f_angle
    fga = fa and (flags & f_comp_g)
    fia = fa and (flags & f_inc_rng)
    cyc = -rng - 1 + fia

    derts__, i_ = construct_input_array(P_, bounds, cyc, fa, fga, fia)   # construct input array with predetermined shape

    if not flags:           # no flag: hypot_g, return current line derts__
        return derts__, i_

    dbuff.appendleft(derts__)

    if len(dbuff) == 0:                                     # no ibuff
        return [], i_                                       # return empty _derts__
    if i_.shape[0] <= rng * 2:                              # incomplete dbuff
        return [], np.concatenate((ibuff, i_), axis=0)      # return empty _derts__

    ibuff = np.concatenate((ibuff[1:], i_), axis=0)         # discard top line, append last line ibuff

    d_ = convolve(ibuff, kernels[rng])   # convolve ibuff with kernels

    if flags & f_inc_rng:               # accumulate with
        d_ += accumulated_d_(dbuff[0])    # derts on rng-higher line

    _derts__ = calc_g_fold_dert(dbuff.pop(), d_, flags)

    return _derts__ , ibuff

    # ---------- comp_derts() end -------------------------------------------------------------------------------------------

def construct_input_array(P_, bounds, cyc, fa, fga, fia):   # unfold P_

    if flags:   # not for hypot_g
        start, end = bounds
        b_calc_a = fa and not (fga or fia)

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

    else:       # for hypot_g
        derts__ = [(P.x0, [[(p,), (hypot(dy, dx), (dy, dx))] for p, g, dy, dx in P.dert_]) for P in P_]   # unfold into derts__
        i_ = None   # no comparison

    return derts__, i_

    # ---------- construct_input_array() end --------------------------------------------------------------------------------

def convolve(a, k):
    return

    # ---------- convolve() end ---------------------------------------------------------------------------------------------

def calc_g_fold_dert(derts__, d_, flags):
    return

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