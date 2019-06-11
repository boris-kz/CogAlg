from intra_comp import intra_comp

''' Note: This version is mostly for testing

    For more informations, see intra_blob.py '''

ave = 20
ave_blob = 1000  # fixed cost of blob syntax
rave = 10  # fixed root_blob / blob cost ratio: add sub_blobs, Levels+=Level, derts+=dert
ave_n_sub_blobs = 10  # determines rave, adjusted per intra_comp
ave_intra_blob = 1000  # cost of default eval_sub_blob_ per intra_blob

# flags:
f_angle          = 0b00000001
f_inc_rng        = 0b00000010
f_hypot_g        = 0b00000100

''' These filters are accumulated for evaluated intra_comp:
    Ave += ave: cost per next-layer dert, fixed comp grain: pixel
    Ave_blob *= rave: cost per next root blob, variable len sub_blob_
    represented per fork if tree reorder, else revised with each access?
'''


def intra_blob(root_blob):  # arguments are added bit by bit

    # blob_: contain all (sub) blobs of a single layer of a single branch