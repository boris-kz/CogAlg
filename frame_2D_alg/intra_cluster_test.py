"""
Test first few instances of intra_cluster calls in intra_blob.
"""

import matplotlib.pyplot as plt

from utils import imread
from frame_blobs import (
    image_to_blobs,
    ave as Ave, # Initial Ave.
)
from intra_blob import (
    # Functions:
    intra_cluster, filter_blobs,
    # Constants:
    ave, rave,
)
from comp_i import comp_i, F_ANGLE, F_DERIV

if __name__ == "__main__":
    Ave_blob = 1000
    image = imread("./images/raccoon_eye.jpg")
    frame = image_to_blobs(image)
    filter_blobs(frame, Ave_blob)

    # Comparison:
    dert___ = comp_i(root_fork['dert___'],
                     root_fork['rng'],
                     flags)

    # Initialize new fork:
    sub_fork = dict(fork_type=flags,
                    rng=rng,
                    dert___=dert___,
                    mask=None,
                    G=0, M=0, Dy=0, Dx=0, L=0, Ly=0, blob_=[])

    for blob in blob_:
        Ave_blob = intra_cluster(blob, sub_fork, Ave, Ave_blob)
        Ave_blob *= rave  # estimated cost of redundant representations per blob
        Ave += ave  # estimated cost per dert

    filter_blobs(sub_fork, Ave_blob)  # Filter below-Ave_blob blobs.

    print('Done!')


# ------------------------------------------------------------------------
# -------------------------------------------------------------------------------