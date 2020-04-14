"""
For testing intra_comp operations and sub-blobs of different depths.
Requirements: numpy, frame_blobs, intra_comp, utils.
Note: Since these operations performed only on multivariate variables,
"__" in variable names will be skipped.
"""

import frame_blobs
from intra_comp import *
from utils import imread, imwrite
from intra_blob_draft import ave

# -----------------------------------------------------------------------------
# Constants

# Input:
IMAGE_PATH = "../images/raccoon.jpg"

# Outputs:
OUTPUT_PATH = "../visualization/images/"
OUTPUT_BIN = False
OUTPUT_NORMALIZE = True

# Aves:
ANGLE_AVE = 100

# Maximum recursion depth:
MAX_DEPTH = 3

# -----------------------------------------------------------------------------
# Functions

def recursive_comp(dert_, rng, Ave, fork_history, depth, fca, nI):  # replace nI with flags
    """Comparisons under a fork."""
    dert_ = comp_g(dert_)  # add other comp forks
    Ave += ave

    dscope = len(dert_)
    assert (dscope in (4, 5, 11, 12))
    has_m = dscope in (5, 12)
    # 0 1 2 3 4 5 6 7 8 9 0 1
    # i g d d a a g d d d d
    # i g m d d a a g d d d d
    if not fca:
        assert len(dert_) in (4, 5)
        recursive_comp(dert_, rng, Ave, fork_history, depth-1, fca=1,
                       nI=2+has_m)
    else:
        assert len(dert_) in (11, 12)
        # Ongoing forks will have i outputed while
        # terminated forks will have all i, g and ga outputed.
        imwrite_fork(dert_, 'i', Ave, fork_history)
        if depth == 0 or rng*2 + 1 > 3:
            imwrite_fork(dert_, 'g', Ave, fork_history+"g")
            imwrite_fork(dert_, 'ga', Ave, fork_history+"ga")
            return

        recursive_comp(dert_, rng+1, Ave, fork_history+"r", depth,
                       fca=0, nI=0)
        recursive_comp(dert_, rng*2+1, Ave, fork_history+"g", depth,
                       fca=0, nI=1)
        recursive_comp(dert_, rng*2+1, Ave, fork_history+"ga", depth,
                       fca=0, nI=7+has_m)
        recursive_comp(dert_, rng+1, Ave, fork_history + "ra", depth,
                       fca=1, nI=4+has_m)


def imwrite_fork(dert_, param, Ave, fork_history):
    """Output fork's gradient image."""
    # Select dert_ to draw:
    if param == 'i':
        o = dert_[0]
    elif param == 'g':
        o = dert_[1]
    elif param == 'm':
        assert len(dert_) in (5, 12)
        o = dert_[2]
    elif param == 'ga':
        o = dert_[7] if len(dert_) == 12 else dert_[6]
    else:
        o = None

    if OUTPUT_BIN:
        if fork_history[-1] == "a":
            Ave = ANGLE_AVE
        imwrite(OUTPUT_PATH + fork_history, (o > Ave) * 255)
    elif OUTPUT_NORMALIZE:
        imwrite(OUTPUT_PATH + fork_history,
             255 * (o - o.min()) / (o.max() - o.min()))
    else:
        imwrite(OUTPUT_PATH + fork_history, o)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(IMAGE_PATH)
    print('Done!')

    print('Doing first comp...')
    dert_ = frame_blobs.comp_pixel(image)
    print('Done!')

    print('Doing recursive comps...')
    # Recursive comps:
    recursive_comp(dert_=dert_,
                   rng=1,
                   fca=1,
                   nI=2,
                   Ave=ave,
                   fork_history="i",
                   depth=MAX_DEPTH)
    print('Done!')

    print('Terminating...')

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------