from math import atan2, pi
from comp_range import scan_slice_

angle_coef = 128 / pi   # to scale angle into (-128, 128)

# ************ FUNCTIONS ************************************************************************************************
# -comp_angle()
# -lateral_comp()
# -vertical_comp()
# ***********************************************************************************************************************

def comp_angle(P_, buff___, i_dert):    # compute, compare angle

    derts__ = lateral_comp(P_, i_dert)              # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___)      # vertical and diagonal comparison (return last line in buff___)

    return _derts__

    # ---------- comp_angle() end -------------------------------------------------------------------------------------------

def lateral_comp(P_, i_dert):  # horizontal comparison between pixels at distance == rng

    derts__ = []

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]

        _derts = derts_[0]
        idx, idy = _derts[i_dert][-3:-1]                # take dy, dx from pre-indicated dert
        _a = int(atan2(idy, idx) * angle_coef) + 128    # angle: 0 -> 255
        _ncomp, _dx = 0, 0                              # init ncomp, dx buffers

        for derts in derts_[1:]:
            # compute angle:
            idx, idy = derts[i_dert][-3:-1]             # take dy, dx from pre-indicated dert
            a = int(atan2(idy, idx) * angle_coef) + 128     # angle: 0 -> 255

            d = a - _a      # lateral comparison
            # correct angle diff:
            if d > 127:
                d -= 255
            elif d < -127:
                d += 255
            dx = d          # dx
            _ncomp += 1     # bilateral accumulation
            _dx += d        # bilateral accumulation

            _derts.append((_a, _ncomp, 0, _dx))     # return, with _dy = 0

            _derts = derts          # buffer last derts
            _a, _ncomp, _dx = a, 1, dx     # buffer last ncomp and dx


        _derts.append((_a, _ncomp, 0, _dx))  # return last one

        derts__.append((x0, derts_))    # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___):    # vertical comparison

    if not buff___:     # buff___ is empty on the first line
        _derts__ = []
    else:               # not the first line
        _derts__ = buff___[0]

        scan_slice_(_derts__, derts__, i_index=(-1, 0), fangle=True)    # unlike other branches, i_dert in comp_angle is always -1

    buff___.appendleft(derts__)

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------