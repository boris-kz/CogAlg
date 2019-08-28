"""
For testing comparison operations and sub-blobs of different depths.

Requirements: numpy, frame_blobs, comp_i, utils.

Note: Since these operations performed only on multivariate variables,
"__" in variable names will be skipped.
"""

import numpy as np
import numpy.ma as ma

import frame_blobs

from comp_i import comp_i
from utils import imread, draw
from intra_blob import ave

# -----------------------------------------------------------------------------
# Adjustable parameters

# Input:
image_path = "../images/raccoon.jpg"

# Outputs:
output_path = "../visualization/images/"
output_bin = False
output_normalize = True

# Aves:
angle_ave = 100

# How ave is increased:
increase_ave = lambda ave: ave * rave

# Maximum recursion depth:
max_depth = 3

# -----------------------------------------------------------------------------
# Functions

def recursive_comp(derts, rng, Ave, fork_history, depth, fa, iG=None):
    """Comparisons under a fork."""
    derts = comp_i(derts, rng, fa, iG)
    Ave += ave

    if not fa:
        recursive_comp(derts, rng, Ave, fork_history, depth, fa=1)
    else:
        draw_fork(derts[0], Ave, fork_history)
        if depth == 0 or rng*2 + 1 > 3:
            draw_fork(derts[1], Ave, fork_history+"g")
            draw_fork(derts[5], Ave, fork_history+"ga")
            return
        recursive_comp(derts, rng+1, Ave, fork_history+"r", depth,
                       fa=0, iG=0)
        recursive_comp(derts, rng*2+1, Ave, fork_history+"g", depth,
                       fa=0, iG=1)
        recursive_comp(derts, rng*2+1, Ave, fork_history+"ga", depth,
                       fa=0, iG=5)


def draw_fork(g, Ave, fork_history):
    """Output fork's gradient image."""
    if output_bin:
        if fork_history[-1] == "a":
            Ave = angle_ave
        draw(output_path + fork_history, (g > Ave) * 255)
    elif output_normalize:
        draw(output_path + fork_history,
             255 * (g - g.min()) / (g.max() - g.min()))
    else:
        draw(output_path + fork_history, g)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(image_path)
    print('Done!')

    print('Doing first comp...')
    derts = frame_blobs.comp_pixel(image)
    print('Done!')

    print('Doing recursive comps...')
    # Recursive comps:
    recursive_comp(derts=derts,
                   rng=1,
                   fa=1,
                   Ave=ave,
                   fork_history="i",
                   depth=max_depth)
    print('Done!')

    print('Terminating...')
    
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------