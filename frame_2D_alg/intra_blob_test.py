from functools import reduce

from utils import over_draw
from compare_i import compare_i

'''
Note: This version is mostly for testing.
For more informations, see intra_blob.py
'''

# -----------------------------------------------------------------------------
ave = 20
ave_blob = 1000  # fixed cost of blob syntax
rave = 10  # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10  # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

# flags:
f_angle = 0b01
f_derive = 0b10

# -----------------------------------------------------------------------------

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else revised with each access?
'''


def intra_blob(blob_, derts__, Ave_blob):  # arguments are added bit by bit
    '''Evaluate root-blob, perform forking and computing sub-blobs.'''
    # Angle fork:
    ablob_, Ave_blob = eval_fork(blob_, derts__, Ave_blob, f_angle)


    Ave_blob *= rave  # estimated cost of redundant representations per blob
    Ave += ave  # estimated cost per dert

    # evaluation for angle comp:
    selected_ablob_ = map(lambda ablob: ablob.Derts[0].G > Ave_blob, ablob_)

    dert_map = reduce(lambda map, blob:
                      over_draw(map, blob.map, blob.box, tv=False),
                      sequence=selected_blob_,
                      initial=np.zeros(derts__[0].shape, dtype=bool))

    # compare:
    dert__ = compare_i(derts__, dert_map, flags | f_angle)


def eval_fork(blob_, derts__, Ave_blob, flags=0):
    """Return sub-blobs of fork."""
    # Blob evaluation and filtering:
    selected_blob_ = map(lambda blob: blob.Derts[0].G > Ave_blob, blob_)

    # apply overdraw to the sequence, return whole map:
    dert_mask = reduce(lambda map, blob:
                       over_draw(map, blob.map, blob.box, tv=False),
                       sequence=selected_blob_,
                       initial=np.zeros(derts__[0].shape, dtype=bool))
    # compare:
    dert__ = compare_i(derts__, dert_map, flags)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------