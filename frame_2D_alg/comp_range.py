from collections import deque

def comp_range(P_, dert_buff___, dert___):  # compare derts at increasing range
    # P_: contains Ps at current line
    # dert_buff___: each contains 1 dert__, which, in turn contains dert_s, spans of contiguous derts

    rng = dert_buff___.maxlen

    dert__ = lateral_comp(P_, rng)                  # horizontal comparison

    vertical_comp(dert__, dert_buff___, dert___)    # vertical and diagonal comparison

def lateral_comp(P_, rng):  # horizontal comparison between pixels at distance == rng

    new_dert__ = []                 # initialize output

    dert_buff_ = deque(maxlen=rng)  # new dert's buffer

    max_index = rng - 1             # max_index of dert_buff_

    _x0 = 0                         # prior x0, or x0 of previous P

    for P in P_:

        new_dert_ = []

        x0 = P[1]

        for x in range(_x0, x0):    # through invalid coordinates

            dert_buff_.append(None)

        Pdert_ = P[-1]

        for x, (i, dy, dx, g) in enumerate(Pdert_, start=x0):
            ncomp = 4
            if len(dert_buff_) == rng and dert_buff_[max_index] != None:  # xd == rng and valid coordinate

                _i, _ncomp, _dy, _dx = dert_buff_[max_index]

                d = i - _i      # lateral comparison

                ncomp += 1      # bilateral accumulation
                dx += d         # bilateral accumulation
                _ncomp += 1     # bilateral accumulation
                _dx += d        # bilateral accumulation

                new_dert_.append((_i, _ncomp, _dy, _dx))

            dert_buff_.appendleft((i, ncomp, dy, dx))

        while dert_buff_:
            new_dert_.append(dert_buff_.pop())

        new_dert__.append((x0, new_dert_))      # each new_dert_ (span of contiguous derts) is appended into new_dert__

        _x0 = x0

    return new_dert__

def vertical_comp(dert__, dert_buff___, dert___):    # vertical and diagonal comparison
    # if dert_buff___ has reached it's maxlen, it's last element (containing first line derts)  is appended to dert___

    dert_ = [(x,) + dert for x, dert in enumerate(dert_, start=x0) for x0, dert_ in dert__]  # flatten current line derts

    for yd, _dert__ in enumerate(dert_buff___, start=1):    # yd: vertical distance between dert_ and _dert_

        _dert_ = [(_x,) + _dert for _x, _dert in enumerate(_dert_, start=_x0) for _x0, _dert_ in _dert__]  # flatten higher line derts

        if yd == rng:   # vertical comp

            i = 0  # index of dert
            _i = 0  # index of higher line dert

            while i < len(dert_) and _i < len(_dert_):  # while there's still comparison to be performed. Loop is per i
                dert = dert_[i]

                x = dert[0]

                while _i < len(_dert_) and _dert_[_i] < x:      # search for the right coordinate
                    _i += 1

                if _i < len(_dert_) and _dert_[_i] == x:    # if coordinate is valid

                    # perform comparison between dert and _dert_[_li] here

        else:           # diagonal comp

            xd = rng - yd

            i = 0       # index of dert
            _li = 0     # index of upper-left dert
            _ri = 0     # inder of upper-right dert

            while i < len(dert_) and (_li < len(_dert_) or _ri < len(_dert_)):  # while there's still comparison to be performed. Loop is per i
                dert = dert_[i]

                x = dert[0]

                _lx = x - xd    # upper-left x coordinate
                _lx = x + xd    # upper-right x coordinate

                # upper-left comp:

                while _li < len(_dert_) and _dert_[_li] < _lx:  # search for the right coordinate
                    _li += 1

                if _li < len(_dert_) and _dert_[_li] == _lx:    # if coordinate is valid

                    # perform comparison between dert and _dert_[_li] here

                # upper-right comp:

                while _ri < len(_dert_) and _dert_[_ri] < _rx:  # search for the right coordinate
                    _ri += 1

                if _ri < leng(_dert_) and _dert_[_li] == _lx:   # if coordinate is valid

                    # perform comparison between dert and _dert_[_ri] here

    return