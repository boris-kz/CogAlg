from comp_range import scan_slice_

# ************ FUNCTIONS ************************************************************************************************
# -comp_gradient()
# -lateral_comp()
# -vertical_comp()
# ***********************************************************************************************************************

def comp_gradient(P_, buff___, alt):    # compare g in blob ( dert__ in P_ line ( dert_ in P

    derts__ = lateral_comp(P_)                          # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___)          # vertical and diagonal comparison (return last line in buff___)

    return _derts__

    # ---------- comp_gradient() end ----------------------------------------------------------------------------------------

def lateral_comp(P_):  # horizontal comparison

    derts__ = []

    for P in P_:
        x0 = P[1]
        derts_ = P[-1]
        new_derts_ = []

        _derts = derts_[0]
        _g = _derts[-1][0]          # take g from indicated dert
        _dx, _ncomp = 0, 0          # init ncomp, dx buffers

        for derts in derts_[1:]:
            # compute angle:
            g = derts[-1][0]    # take g from indicated dert

            d = g - _g          # lateral comparison

            dx = d              # dx
            _dx += d            # bilateral accumulation
            _ncomp += 1         # bilateral accumulation

            new_derts_.append(_derts + [(0, _dx, _ncomp)])     # make new derts with addition dert, append it to new_derts_

            _derts = derts                  # buffer last derts
            _g, _dx, _ncomp = g, dx, 1      # buffer last ncomp and dx

            new_derts_.append(_derts + [(0, _dx, _ncomp)])     # return last one

        derts__.append((x0, new_derts_))    # new line of P derts_ appended with new_derts_

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