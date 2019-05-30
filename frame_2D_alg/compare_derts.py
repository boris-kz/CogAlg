from collections import deque
from math import hypot, pi
from cmath import rect, phase

two_pi = 2 * pi         # angle constraint
angle_coef = 256 / pi   # to scale angle into (-128, 128)

# flags:
Flag_angle          = b001
Flag_inc_rng        = b010
Flag_hypot_g        = b100

# ************ FUNCTIONS ************************************************************************************************
# -compare_derts()
# ***********************************************************************************************************************

def comp_dert(P_, buff___, Ave, rng, flags):    # comparison of input param between derts at range = rng
    # dert_buff___ in blob ( dert__ in P_ line ( dert_ in P

    fa = flags & Flag_angle

    if flags & Flag_hypot_g:
        _derts__ = hypot_g(P_)
    else:
        if fa and rng == 1:
            pass

    return

    # ---------- compare_derts() end ----------------------------------------------------------------------------------------