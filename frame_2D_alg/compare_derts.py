import numpy as np

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_comp_g         = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -comp_derts()
# -construct_input_array(P_, x0, mask)
# ***********************************************************************************************************************

def comp_derts(P_, buff__, bounds, mask, Ave, rng, flags):    # comparison of input param between derts at range = rng
    # dert_buff___ in blob ( dert__ in P_ line ( dert_ in P

    i_ = construct_input_array(P_, bounds, flags)   # construct input array with predetermined shape

    if buff__.shape[0] < rng:
        return [], np.concatenate((buff__, i_.reshape(1, -1)), axis=0)

    np.concatenate((buff__[1:], i_.reshape(1, -1)), axis=0)     # discard top line, append last line buff

    _derts__ = # convolve buff__ with kernels, compute g, fold into _derts__

    return _derts__ , buff__

    # ---------- comp_derts() end -------------------------------------------------------------------------------------------