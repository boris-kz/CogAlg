'''
    Kernel-based version of:
    Comparison of input param between derts at range=rng, summing derivatives from shorter + current range comps per pixel
    Input is pixel brightness p or gradient g in dert[0] or angle a in dert[1]: g_dert = g, (dy, dx); ga_dert = g, a, (dy, dx)
    if fa: compute and compare angle from dy, dx in dert[-1], only for g_dert in 2nd intra_comp of intra_blob
    else:  compare input param in dert[fia]: p|g in derts[cyc][0] or angle a in dert[1]
    flag ga: i_dert = derts[cyc][fga], both fga and fia are set for current intra_blob forks and potentially recycled
    flag ia: i = i_dert[fia]: selects dert[1] for incremental-range comp angle only
'''

import numpy as np

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_hypot_g        = 0b00000100

# ************ FUNCTIONS ************************************************************************************************
# -compare_i()
# ***********************************************************************************************************************

def compare_i(derts__, map, flags, rng):    # comparison of input param between derts at range = rng. Currently under revision
    # _dert___ in blob ( dert__ in P_ line ( dert_ in P
    if not (flags & f_hypot_g): # not hypot_g

        fia = flags & f_angle
        fga = fia and not (flags & f_inc_rng)
        cyc = -rng - 1 + fia

        # ...

    else:                       # hypot_g: (dert__ not derts__)
        dert__ = derts__
        dy__ = dert__[:, :, 2]    # the third one is to reserve 3-D shape of array
        dx__ = dert__[:, :, 3]
        g__ = np.empty(dy__.shape)

        g__[map] = np.hypot(dy__[map], dx__[map])

        new_dert__ = [np.stack((dy__, dx__, g__), axis=2)]

    return new_dert__

    # ---------- compare_i() end --------------------------------------------------------------------------------------------