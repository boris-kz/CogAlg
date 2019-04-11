import numpy as np
import numpy.ma as ma
from math import hypot
from collections import deque

def comp_range(P_, dert_buff___, dert___):  # compare derts at increasing range
    # P_: contains Ps at current line
    # dert_buff___: each contains 1 dert__, which, in turn contains dert_s, spans of contiguous derts

    rng = dert_buff___.maxlen
    for yd, _dert__ in enumerate(dert_buff___, start=1):    # yd: vertical distance between dert_ and _dert_

        _dert_ = [(_x,) + _dert for _x, _dert in enumerate(_dert_, start=_x0) for _x0, _dert_ in _dert__]  # flatten higher line derts
        dert_ = [(x,) + dert for x, dert in enumerate(P[-1], start=P[1]) for P in P_]  # flatten current line derts

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

                if _ri < len(_dert_) and _dert_[_li] == _lx:   # if coordinate is valid

                # perform comparison between dert and _dert_[_ri] here
