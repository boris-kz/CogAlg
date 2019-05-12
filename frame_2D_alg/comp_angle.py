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
        gd, dx, dy, _ = _derts[-1]         # comp_angle always follows comp_gradient or hypot_g
        _a = complex(dx, dy)               # to complex number: _a = dx + dyj
        _a /= abs(_a)                      # normalize _a so that abs(_a) == 1 (hypot() from real and imaginary part of _a == 1)
        _a_radiant = phase(_a)             # angular value of _a in radiant: _a_radiant in (-pi, pi)
        _dax, _ncomp = 0j, 0               # init ncomp, dx(complex) buffers

        for derts in derts_[1:]:
            # compute angle:
            g, dx, dy, _ = derts[-1]       # derts_ and new_derts_ are separate
            a = complex(dx, dy)            # to complex number: a = dx + dyj
            a /= abs(a)                    # normalize a so that abs(a) == 1 (hypot() from real and imaginary part of a == 1)
            a_radiant = phase(a)           # angular value of a in radiant: aa in (-pi, pi)

            da = rect(1, a_radiant - _a_radiant)   # convert bearing difference into complex form (rectangular coordinate)
            # complex d doesn't need to correct angle diff: if d > pi: d -= 255; elif d < -127: d += 255
            dx = da
            _dax += da      # bilateral accumulation
            _ncomp += 1     # bilateral accumulation

            new_derts_.append(_derts + [(_a, _a_radiant), (0j, _dax, _ncomp)])  # return a and _dy = 0 + 0j in separate tuples
            _derts = derts                         # buffer derts
            _a, _aa, _dx, _ncomp = a, a_radiant, dx, 1    # buffer last ncomp and dx

        new_derts_.append(_derts + [(_a, _a_radiant), (0j, _dax, _ncomp)]) # return last derts

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

def ga_from_da(da_x, da_y, ncomp):
    " convert dx, dy to angular value then compute g"
    da_x = phase(da_x / ncomp)      # phase(da_x) is the same as phase(da_x / ncomp)
    da_y = phase(da_y / ncomp)      # phase(da_y) is the same as phase(da_y / ncomp)

    ga = hypot(da_x, da_y)
    if ga > pi: ga = two_pi - ga        # translate ga's scope into [0, pi) (g is unsigned)

    return int(ga * angle_coef)

def clamp_angle(angle, lower=-pi, upper=pi):    # keep angle between in scope [lower, upper). Not used
    span = upper - lower
    while angle < lower: angle += span
    while angle >= upper: angle -= span

    return angle
