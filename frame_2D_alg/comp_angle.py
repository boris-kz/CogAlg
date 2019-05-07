from cmath import phase, rect
from math import hypot, pi
from comp_range import scan_slice_

two_pi = 2 * pi
angle_coef = 256 / pi   # to scale angle into (-128, 128)

# ************ FUNCTIONS ************************************************************************************************
# -comp_angle()
# -lateral_comp()
# -vertical_comp()
# ***********************************************************************************************************************

def comp_angle(P_, buff___, alt):    # compute, compare angle

    derts__ = lateral_comp(P_)       # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___)  # vertical and diagonal comparison (return last line in buff___)

    return _derts__

    # ---------- comp_angle() end -------------------------------------------------------------------------------------------

def lateral_comp(P_):  # horizontal comparison between pixels at distance == rng

    derts__ = []

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]
        new_derts_ = []     # new derts buffer

        _derts = derts_[0]
        gd, dxd, dyd, _ = _derts[-1]       # comp_angle always follows comp_gradient or hypot_g
        _a = complex(dxd, dyd)             # to complex number
        _a /= abs(_a)                      # normalize _a
        _aa = phase(_a)                    # angular value of _a in radiant
        _dx, _ncomp = 0j, 0                # init ncomp, dx(complex) buffers

        for derts in derts_[1:]:
            # compute angle:
            g, dx, dy, _ = derts[-1]       # derts_ and new_derts_ are separate
            a = complex(dx, dy)            # to complex number
            a /= abs(a)                    # normalize a
            aa = phase(a)                  # angular value of _a in radiant

            d = rect(1, aa - _aa)          # convert bearing difference into complex form (rectangular coordinate)
            # complex d doesn't need to correct angle diff: if d > pi: d -= 255; elif d < -127: d += 255
            dx = d
            _dx += d        # bilateral accumulation
            _ncomp += 1     # bilateral accumulation

            new_derts_.append(_derts + [(_a, _aa), (0j, _dx, _ncomp)])   # return, with _dy = 0 + 0j and a in a separate tuple
            _derts = derts                          # buffer derts
            _a, _aa, _dx, _ncomp = a, aa, dx, 1     # buffer last ncomp and dx

        new_derts_.append(_derts + [(_a, _aa), (0j, _dx, _ncomp)])  # return last derts

        derts__.append((x0, new_derts_))    # new line of P derts_ appended with new_derts_

    return derts__

    # ---------- lateral_comp() end -----------------------------------------------------------------------------------------

def vertical_comp(derts__, buff___):    # vertical comparison

    if not buff___:  # first line, if buff___ is empty
        _derts__ = []
    else:               
        _derts__ = buff___[0]
        scan_slice_(_derts__, derts__, i_index=(-2, 1), fangle=True)

    buff___.appendleft(derts__)

    return _derts__

    # ---------- vertical_comp() end ----------------------------------------------------------------------------------------

def ga_from_da(da_x, da_y, ncomp, Ave):
    " convert dx, dy to angular value then compute g"
    da_x = phase(da_x)
    da_y = phase(da_y)

    ga = hypot(da_x, da_y)
    if ga > pi: ga = two_pi - ga        # translate ga's scope into [0, pi) (g is unsigned)

    return int(ga * angle_coef) - Ave

def clamp_angle(angle, lower=-pi, upper=pi):    # keep angle between in scope [lower, upper). Not used
    span = upper - lower
    while angle < lower: angle += span
    while angle >= upper: angle -= span

    return angle
