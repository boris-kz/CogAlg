import numpy as np
from math import hypot
from math import atan2
from collections import deque, namedtuple

nt_blob = namedtuple('blob', 'typ sign Y X Ly L Derts seg_ sub_blob_ layers_f sub_Derts map box add_dert rng ncomp')

# ************ FUNCTIONS ************************************************************************************************
# -form_sub_blob()
# -unfold_blob()
# -form_P_()
# -scan_P_()
# -form_seg_()
# -form_blob()
# ***********************************************************************************************************************

''' this module is under revision '''


def form_sub_blob(dert_, root_blob):  # redefine blob as branch-specific master blob: local equivalent of frame

    seg_ = deque()

    while dert_:
        P_ = form_P_()                     # horizontal clustering
        P_ = scan_P_()       # vertical clustering
        seg_ = form_seg_()   # vertical clustering
    while seg_: form_blob()  # terminate last running segments

    # ---------- add_sub_blob() end -----------------------------------------------------------------------------------------

def unfold_blob(blob, comp):     # unfold and compare
    return
    # ---------- unfold_blob() end ------------------------------------------------------------------------------------------

def form_P_():  # cluster and sum horizontally consecutive pixels and their derivatives into Ps
    return
    # ---------- form_P_() end ------------------------------------------------------------------------------------------


def scan_P_():  # this function detects connections (forks) between Ps and _Ps, to form blob segments
    return
    # ---------- scan_P_() end ------------------------------------------------------------------------------------------

def form_seg_():  # Convert or merge every P into segment. Merge blobs
    return
    # ---------- form_seg_() end --------------------------------------------------------------------------------------------


def form_blob():  # terminated segment is merged into continued or initialized blob (all connected segments)
    return
    # ---------- form_blob() end -------------------------------------------------------------------------------------