"""
For testing comparison operations and sub-blobs of different depths.
Requirements: numpy, frame_blobs, comp_i, utils.
"""

import numpy as np
import numpy.ma as ma
import frame_blobs
from comp_i import comp_i
from utils import imread, draw

# -----------------------------------------------------------------------------

# Declare comparison flags:
F_ANGLE = 0b01
F_DERIV = 0b10

fork_dict = {
    'r': 0,
    'a': F_ANGLE,
    'g': F_DERIV,
}

# -----------------------------------------------------------------------------
# Input:
image_path = "../images/raccoon.jpg"

# Outputs:
output_path = "../debug/"
binary_output = True

# filters:
init_ave = 20
angle_ave = 20

increase_ave = lambda ave, rng: ave * ((rng * 2 + 1) ** 2 - 1) / 2
# Uncomment below definition of increase_ave for identity function:
# increase_ave = lambda ave, rng: ave

# comp sequence:
fork_tree = [
    ("a", [
        ("r", [
            ("r", []),
        ]),
    ]),
    ("g", [
        ("a", []),
        ("r", [
            ("a", []),
        ]),
    ]),
]
# -----------------------------------------------------------------------------
# Functions

def comp_tree(derts, rng, Ave, past_forks, tree):  # comparisons down fork_tree

    for fork, subtree in tree: # Stop recursion if tree = [].
        next_layer(derts, rng, Ave, past_forks,
                fork, subtree)

def next_layer(derts, rng, Ave, past_forks, fork, subtree):  # calls deeper forks

    if fork == 'r':
        rng += 1
        past_forks = past_forks[:-1] + str(rng)  # Replace rng only.
    else:
        past_forks += fork + str(rng)  # Add new derivation.
        if fork == "a":
            rng = 1

    Ave = increase_ave(Ave, rng)
    _, derts = comp_i(derts,
                      rng=rng,
                      flags=fork_dict[fork])

    draw_fork(derts, Ave, past_forks)
    comp_tree(derts, rng, Ave, past_forks, subtree)

def draw_fork(derts, Ave, past_forks):  # Output fork' gradient image

    out = derts[-1][0]
    if binary_output:
        if past_forks[-2] == "a":
            Ave = angle_ave
        draw(output_path + past_forks, (out > Ave) * 255)
    else:
        draw(output_path + past_forks, (out - out.min()) / (out.max() - out.min()) * 255)

# ---------------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    image = imread(image_path)
    input, dert = frame_blobs.comp_pixel(image)
    draw_fork([dert], init_ave, "g" + str(frame_blobs.rng))

    # Recursive comps:
    comp_tree(derts=[
                       ma.masked_array(image)[np.newaxis, ...],
                       ma.masked_array(dert),
                   ],
                   rng=1,
                   Ave=init_ave,
                   past_forks ="g" + str(frame_blobs.rng),
                   tree=fork_tree)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
