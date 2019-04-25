from math import atan2, pi
from comp_range import scan_slice_

angle_coef = 128 / pi   # to scale angle into (-128, 128)

# ************ FUNCTIONS ************************************************************************************************
# -comp_angle()
# -lateral_comp()
# -vertical_comp()
# ***********************************************************************************************************************

def comp_angle(P_, buff___):    # comp i at incremented range, dert_buff___ in blob ( dert__ in P_ line ( dert_ in P

    derts__ = lateral_comp(P_)                     # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___)     # vertical and diagonal comparison (return last line in buff___)

    buff___.appendleft(derts__)                         # buffer derts__ into buff___, after vertical_comp to preserve last derts__ in buff___

    return _derts__  # return i indices and derts__

    # ---------- comp_angle() end -------------------------------------------------------------------------------------------

def lateral_comp(P_):  # horizontal comparison between pixels at distance == rng

    derts__ = []

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]

        _derts = derts_[0]
        idx, idy = _derts[0][-3:-1]
        _a = int(atan2(idy, idx) * angle_coef) + 128    # angle: 0 -> 255
        _ncomp, _dx = 0, 0                              # buffer ncomp, dx

        for derts in derts_:
            # compute angle:
            idx, idy = derts[0][-3:-1]
            a = int(atan2(idy, idx) * angle_coef) + 128  # angle: 0 -> 255

            d = a - _a      # lateral comparison
            # correct d_angle:
            if d > 127:
                d -= 255
            elif d < -127:
                d += 255
            dx = d          # dx
            _ncomp += 1     # bilateral accumulation
            _dx += d        # bilateral accumulation

            _derts.append((_a, _ncomp, 0, _dx))     # return, with _dy = 0
            _derts = derts          # buffer last derts
            _ncomp, _dx = 1, dx     # buffer last ncomp and dx

        _derts.append((_a, _ncomp, 0, _dx))  # return last dert
        derts__.append((x0, derts_))   # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___):    # vertical comparison

    if not buff___:  # first line buff___ is empty
        _derts__ = []
    else:
        _derts__ = buff___[0]
        scan_slice_(_derts__, derts__, i_index=(-1, 0), fangle=True)

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------