from collections import deque

def comp_range(P_, buff___, dert_index):  # compare i param at incremented range

    # P_: current line Ps
    rng = buff___.maxlen  # dert_buff___ per blob ( dert__ per line ( dert_ per P

    derts__ = lateral_comp(P_, rng)                  # horizontal comparison (return current line)
    _derts__ = vertical_comp(derts__, buff___)    # vertical and diagonal comparison (return last line in buff___)

    return _derts__


def lateral_comp(P_, rng):  # horizontal comparison between pixels at distance == rng

    new_derts__ = []                 # initialize output
    buff_ = deque(maxlen=rng)  # new dert's buffer
    max_index = rng - 1             # max_index of dert_buff_
    _x0 = 0                         # prior x0, or x0 of previous P

    for P in P_:
        new_derts_ = []
        x0 = P[1]
        derts_ = P[-1]
        for x in range(_x0, x0):    # through invalid coordinates
            buff_.append(None)      # buffer invalid derts as None

        for x, derts in enumerate(derts_, start=x0):
            i = derts[-1][-rng+1][0]        # +1 due to this being done before new dert is appended
            ncomp, dy, dx = derts[-1][-4:-1]
            if len(buff_) == rng and buff_[max_index] != None:  # xd == rng and valid coordinate

                _derts = buff_[max_index]
                _i = _derts[-rng][0]
                _ncomp, _dy, _dx = _derts[-1]

                d = i - _i      # lateral comparison
                ncomp += 1      # bilateral accumulation
                dx += d         # bilateral accumulation
                _ncomp += 1     # bilateral accumulation
                _dx += d        # bilateral accumulation

                _derts[-1] = _ncomp, _dy, _dx   # assign back into _dert

                new_derts_.append(_derts)       # buffer into line derts_

            derts.append((ncomp, dy, dx))       # new dert

            buff_.appendleft(derts)             # buffer for horizontal comp

        while buff_:
            new_derts_.append(buff_.pop())      # terminate last derts in line

        new_derts__.append((x0, new_derts_))    # each new_derts_ (span of horizontally continuous derts) is appended into new_derts__
        _x0 = x0

    return new_derts__

def vertical_comp(dert__, dert_buff___, dert___):    # vertical and diagonal comparison. Currently under revision

    # at len=maxlen, first line of derts in last element of buff___ is returned to comp_range()

    dert_ = [(x,) + dert for x, dert in enumerate(dert_, start=x0) for x0, dert_ in dert__]
    # flatten current line derts

    for yd, _dert__ in enumerate(dert_buff___, start=1):  # yd: vertical distance between dert_ and _dert_

        _dert_ = [(_x,) + _dert for _x, _dert in enumerate(_dert_, start=_x0) for _x0, _dert_ in _dert__]
        # flatten higher line derts

        if yd == rng:   # vertical comp
            i = 0  # index of dert
            _i = 0  # index of higher line dert

            while i < len(dert_) and _i < len(_dert_):  # while there's still comparison to be performed. Loop is per i
                dert = dert_[i]
                x = dert[0]

                while _i < len(_dert_) and _dert_[_i] < x:      # search for the right coordinate
                    _i += 1

                if _i < len(_dert_) and _dert_[_i] == x:    # if coordinate is valid
                # compare dert to _dert_[_li] here

        else:   # diagonal comp

            xd = rng - yd
            i = 0       # index of dert
            _li = 0     # index of upper-left dert
            _ri = 0     # index of upper-right dert

            while i < len(dert_) and (_li < len(_dert_) or _ri < len(_dert_)):  # while there is a comparand, Loop is per i
                dert = dert_[i]
                x = dert[0]

                _lx = x - xd    # upper-left x coordinate
                _lx = x + xd    # upper-right x coordinate

                # upper-left comp:

                while _li < len(_dert_) and _dert_[_li] < _lx:  # search for the right coordinate
                    _li += 1

                if _li < len(_dert_) and _dert_[_li] == _lx:    # if coordinate is valid
                # compare dert to _dert_[_li] here

                # upper-right comp:

                while _ri < len(_dert_) and _dert_[_ri] < _rx:  # search for the right coordinate
                    _ri += 1

                if _ri < len(_dert_) and _dert_[_li] == _lx:   # if coordinate is valid
                # compare dert to _dert_[_ri] here

    return