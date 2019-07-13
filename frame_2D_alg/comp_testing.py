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
F_ANGLE = 0b01
F_DERIV = 0b10

# Branche dict:
branch_dict = {
    'r': 0,
    'a': F_ANGLE,
    'g': F_DERIV,
}

# -----------------------------------------------------------------------------
# Adjustable parameters

# Input:
image_path = "../images/raccoon.jpg"

# Outputs:
output_path = "../debug/"
binary_output = True

# Aves:
init_ave = 20
angle_ave = 20

# How ave is increased?
increase_ave = lambda ave, rng: ave * ((rng * 2 + 1) ** 2 - 1) / 2
# Uncomment below definition of increase_ave for identity function:
# increase_ave = lambda ave, rng: ave

# Recursive comps' pipelines:
pipe_lines = [ # 3 forks per g, 2 per p | a: no rng+, replaced by g | ga:
    ("g", [
        ("g", [
            ("g", []),
            ("a", []),
            ("r", []),
        ]),
        ("a", [
            ("g", []),
            ("a", []),
        ]),
        ("r", [
            ("g", []),
            ("a", []),
        ]),
    ]),
    ("a", [  # actually ga
        ("g", [
            ("g", []),
            ("a", []),
            ("r", []),
        ]),
        ("a", [
            ("g", []),
            ("a", []),
        ]),
    ])
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
        fork_history = fork_history[:-1] + str(rng)  # Replace rng only.
    else:
        fork_history += branch + str(rng)  # Add new derivation.
        if branch == "a":
            rng = 1

    Ave = increase_ave(Ave, rng)
    _, derts = comp_i(derts,
                      rng=rng,
                      flags=branch_dict[branch])

    draw_fork(derts, Ave, fork_history)

    recursive_comp(derts, rng, Ave, fork_history, subpipes)

def draw_fork(derts, Ave, fork_history):
    """Output fork's gradient image."""
    out = derts[-1][0]
    if binary_output:
        if fork_history[-2] == "a":
            Ave = angle_ave
        draw(output_path + fork_history, (out > Ave) * 255)
    else:
        draw(output_path + fork_history,
             # out)
             (out - out.min()) / (out.max() - out.min()) * 255)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    image = imread(image_path)
    input, dert = frame_blobs.comp_pixel(image)
    draw_fork([dert], init_ave, "g" + str(frame_blobs.rng))

    # Recursive comps:
    recursive_comp(derts=[
                       ma.masked_array(image)[np.newaxis, ...],
                       ma.masked_array(dert),
                   ],
                   rng=1,
                   Ave=init_ave,
                   fork_history="g" + str(frame_blobs.rng),
                   pipes=pipe_lines)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------