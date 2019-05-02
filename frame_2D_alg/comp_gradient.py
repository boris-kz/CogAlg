from math import atan2, pi
from comp_range import scan_slice_

# ************ FUNCTIONS ************************************************************************************************
# -comp_gradient()
# -lateral_comp()
# -vertical_comp()
# ***********************************************************************************************************************

def comp_gradient(P_, buff___, i_dert):    # compare g in blob ( dert__ in P_ line ( dert_ in P

    derts__ = lateral_comp(P_, i_dert)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___, i_dert)     # vertical and diagonal comparison (return last line in buff___)

    return _derts__

    # ---------- comp_gradient() end ----------------------------------------------------------------------------------------

def lateral_comp(P_, i_dert):  # horizontal comparison

    derts__ = []

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]

        _derts = derts_[0]
        _g = _derts[i_dert][-1]     # take g from indicated dert
        _ncomp, _dx = 0, 0          # init ncomp, dx buffers

        for derts in derts_[1:]:
            # compute angle:
            g = derts[i_dert][-1]   # take g from indicated dert

            d = g - _g      # lateral comparison

            dx = d          # dx
            _ncomp += 1     # bilateral accumulation
            _dx += d        # bilateral accumulation

            _derts.append((_ncomp, 0, _dx))     # return, with _dy = 0

            _derts = derts                  # buffer last derts
            _g, _ncomp, _dx = g, 1, dx      # buffer last ncomp and dx


        _derts.append((_ncomp, 0, _dx))     # return last one

        derts__.append((x0, derts_))        # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___, i_dert):    # vertical comparison

    if not buff___:     # buff___ is empty on the first line
        _derts__ = []
    else:               # not the first line
        _derts__ = buff___[0]

        scan_slice_(_derts__, derts__, i_index=(i_dert, -1))

    buff___.appendleft(derts__)

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------