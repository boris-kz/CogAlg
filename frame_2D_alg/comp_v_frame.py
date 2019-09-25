"""
For testing comparison operations and sub-blobs of different depths.

Requirements: numpy, frame_blobs, comp_v, utils.

Note: Since these operations performed only on multivariate variables,
"__" in variable names will be skipped.
"""

import numpy as np
import numpy.ma as ma

import frame_blobs

from comp_v import comp_v
from utils import imread, imwrite
from intra_blob import ave

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

# How ave is increased:
increase_ave = lambda ave: ave * rave

# -----------------------------------------------------------------------------
# Functions

def recursive_comp(derts, rng, Ave, fork_history, depth, fa, nI):
    """Comparisons under a fork."""
    derts = comp_v(derts, nI, rng)
    Ave += ave

    dsize = len(derts)
    assert (dsize in (4, 5, 11, 12))
    has_m = dsize in (5, 12)
    # 0 1 2 3 4 5 6 7 8 9 0 1
    # i g d d a a g d d d d
    # i g m d d a a g d d d d
    if not fa:
        assert len(derts) in (4, 5)
        recursive_comp(derts, rng, Ave, fork_history, depth-1, fa=1,
                       nI=2+has_m)
    else:
        assert len(derts) in (11, 12)
        # Ongoing forks will have i outputed while
        # terminated forks will have all i, g and ga outputed.
        imwrite_fork(derts, 'i', Ave, fork_history)
        if depth == 0 or rng*2 + 1 > 3:
            imwrite_fork(derts, 'g', Ave, fork_history+"g")
            imwrite_fork(derts, 'ga', Ave, fork_history+"ga")
            return

        recursive_comp(derts, rng+1, Ave, fork_history+"r", depth,
                       fa=0, nI=0)
        recursive_comp(derts, rng*2+1, Ave, fork_history+"g", depth,
                       fa=0, nI=1)
        recursive_comp(derts, rng*2+1, Ave, fork_history+"ga", depth,
                       fa=0, nI=7+has_m)
        recursive_comp(derts, rng+1, Ave, fork_history + "ra", depth,
                       fa=1, nI=4+has_m)


def imwrite_fork(derts, param, Ave, fork_history):
    """Output fork's gradient image."""
    # Select derts to draw:
    if param == 'i':
        o = derts[0]
    elif param == 'g':
        o = derts[1]
    elif param == 'm':
        assert len(derts) in (5, 12)
        o = derts[2]
    elif param == 'ga':
        o = derts[7] if len(derts) == 12 else derts[6]
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
    derts = frame_blobs.comp_pixel(image)
    print('Done!')

    print('Doing recursive comps...')
    # Recursive comps:
    recursive_comp(derts=derts,
                   rng=1,
                   fa=1,
                   nI=2,
                   Ave=ave,
                   fork_history="i",
                   depth=MAX_DEPTH)
    print('Done!')

    print('Terminating...')
    
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------