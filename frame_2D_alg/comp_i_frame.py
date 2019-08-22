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

# -----------------------------------------------------------------------------
# Constants

# Declare comparison flags:
F_ANGLE = 0b001
F_DERIV = 0b010
F_RANGE = 0b100

# Branche dict:
FORK_TYPES = {
    'a': F_ANGLE,
    'g': F_DERIV,
    'r': F_RANGE,
}

# -----------------------------------------------------------------------------
# Adjustable parameters

# Input:
image_path = "../images/raccoon.jpg"

# Outputs:
output_path = "../visualization/images/"
output_bin = False
output_normalize = False

# Aves:
init_ave = 20
angle_ave = 100

# How ave is increased?
increase_ave = lambda ave, rng: ave * ((rng * 2 + 1) ** 2 - 1) / 2
# Uncomment below definition of increase_ave for identity function:
# increase_ave = lambda ave, rng: ave

# Recursive comps' pipelines:
pipe_lines = [ # 3 forks per g, 2 per p | a: no rng+, replaced by g | ga:
    ("a", [  # actually ga
        ("g", [
            ("r", []),
            ("a", []),
            ("g", []),
        ]),
        ("a", [
            ("a", []),
            ("g", []),
        ]),
    ]),
    ("g", [
        ("g", [
            ("r", []),
            ("a", []),
            ("g", []),
        ]),
        ("a", [
            ("a", []),
            ("g", []),
        ]),
        ("r", [
            ("r", []),
            ("a", []),
            ("g", []),
        ]),
    ]),
]

# -----------------------------------------------------------------------------
# Functions

def recursive_comp(derts, rng, Ave, fork_history, pipes):
    """Comparisons under a fork."""
    for branch, subpipes in pipes: # Stop recursion if pipes = [].
        forking(derts, rng, Ave, fork_history,
                branch, subpipes)

def forking(derts, rng, Ave, fork_history, branch, subpipes):
    """Forking comps into further forks."""
    if branch == 'r':
        rng += 1
    else:
        if branch == "a":
            rng = 1

    Ave = increase_ave(Ave, rng)
    derts = comp_i(derts,
                   rng=rng,
                   flags=FORK_TYPES[branch])

    fork_history += branch  # Add new derivation.
    draw_fork(derts, Ave, fork_history)

    recursive_comp(derts, rng, Ave, fork_history, subpipes)

def draw_fork(derts, Ave, fork_history):
    """Output fork's gradient image."""
    out = derts[-1][0]
    if output_bin:
        if fork_history[-1] == "a":
            Ave = angle_ave
        draw(output_path + fork_history, (out > Ave) * 255)
    elif output_normalize:
        draw(output_path + fork_history,
             (out - out.min()) / (out.max() - out.min()))
    else:
        draw(output_path + fork_history, out)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(image_path)
    print('Done!')

    print('Doing first comp...')
    input, dert = frame_blobs.comp_pixel(image)
    print('Done!')

    print('Outputting first g array...')
    draw_fork([dert], init_ave, "g")
    print('Done!')

    print('Doing recursive comps...')
    # Recursive comps:
    recursive_comp(derts=[
                       ma.masked_array(image)[np.newaxis, ...],
                       ma.masked_array(dert),
                   ],
                   rng=1,
                   Ave=init_ave,
                   fork_history="g",
                   pipes=pipe_lines)
    print('Done!')

    print('Terminating...')

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------