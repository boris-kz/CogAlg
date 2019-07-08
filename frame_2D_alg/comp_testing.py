"""
For testing comparison operations and sub-blobs of different depths.

Requirements: numpy, frame_blobs, comp_i, utils.

Note: Since these operations performed only on multivariate variables,
"__" in variable names will be skipped.
"""

import numpy as np
import numpy.ma as ma

from frame_blobs import comp_pixel
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

image_path = "../images/raccoon.jpg"

# Aves:
init_ave = 20
angle_ave = 40

# How ave is increased?
increase_ave = lambda ave, rng: ave * ((rng * 2 + 1) ** 2 - 1) / 2
# Uncomment below definition of increase_ave for identity function:
# increase_ave = lambda ave, rng: ave

# Recursive comps' pipelines:
pipe_lines = [
    ("a", [
        ("r3", []),
    ]),
    ("r3", [
        ("g3",[
            ("a", []),
        ]),
    ]),
    ("g3", []),
]

# -----------------------------------------------------------------------------
# Functions

def recursive_comp(derts, rng, Ave, fork_sequence, pipes):
    """Comparisons under a fork."""
    for branch, subpipes in pipes: # Stop recursion if pipes = [].
        forking(derts, rng, Ave, fork_sequence,
                branch, subpipes)

def forking(derts, rng, Ave, fork_sequence, branch, subpipes):
    """Forking comps into further forks."""
    # Identify branch and target rng:
    if len(branch) == 1:
        new_rng = 1
    else:
        new_rng = int(branch[1:])
        branch = branch[0]

    if branch != 'r':
        rng = 1 # Reset rng.
        _, derts = comp_i(derts,
                          rng=rng,
                          flags=branch_dict[branch])
        Ave = increase_ave(Ave, rng)
        fork_sequence += branch + str(rng) # Add new derivation.
        draw_fork(derts, Ave, fork_sequence)

    # Increased range comparison:
    for r in range(rng + 1, new_rng + 1):
        _, derts = comp_i(derts,
                          rng=r,
                          flags=branch_dict['r'])

        Ave = increase_ave(Ave, rng)
        fork_sequence = fork_sequence[:-1] + str(r) # Replace rng only.
        draw_fork(derts, Ave, fork_sequence)

    recursive_comp(derts, new_rng, Ave, fork_sequence, subpipes)

def draw_fork(derts, Ave, fork_sequence):
    """Output fork's gradient image."""
    if fork_sequence[-2] == "a":
        Ave = angle_ave
    draw("../debug/" + fork_sequence, (derts[-1][0] > Ave) * 255)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    image = imread(image_path)
    input, dert = comp_pixel(image)
    draw_fork([dert], init_ave, "g0")

    # Recursive comps:
    recursive_comp(derts=[
                       ma.masked_array(input)[np.newaxis, ...],
                       ma.masked_array(dert),
                   ],
                   rng=0,
                   Ave=init_ave,
                   fork_sequence="g0",
                   pipes=pipe_lines)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------