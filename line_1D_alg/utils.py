"""
Provide some tools to manipulate CogAlg frames and for debugging.
"""

import pickle
from collections import defaultdict
from itertools import tee


import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# itertools recipes

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# -----------------------------------------------------------------------------
# line_patterns utilities

def save_pkl_file(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)


def load_pkl_file(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data


def save_frame_data(obj, file_name="frame.pkl"):
    save_pkl_file(obj, file_name)


def load_frame_data(file_name="frame.pkl"):
    return load_pkl_file(file_name)


def load_checkpoints(file_name="checkpoints.pkl"):
    return load_pkl_file(file_name)


def describe_recursion(frame):
    typ_cnt = [0, 0, 0, 0]
    for P_ in frame:
        for P in P_:
            if len(P[-2]) == 0 and len(P[-3]) == 0:
                continue
            typ = 2 * (~P[0]) + (len(P[-2]) > 0)
            typ_cnt[typ] += 1
    labels = ['rng', 'rng_seg', 'der', 'der_seg']
    plt.pie(typ_cnt, labels=labels)


def describe_L_distribution(frame):
    # Extract and flatten Ls
    L_ = []
    for P_ in frame:
        for P in P_:
            L_.append(P[1])

    # Percentiles
    L1, L2, L3, L4 = np.percentile(L_, [5, 16, 84, 95])
    print(f'90% of Ls are within range: {L1} - {L4}')
    print(f'68% are within range: {L2} - {L3}')

    # Save L distribution histogram
    plt.cla()
    plt.hist(L_, bins=50)
    plt.savefig('L_distribution.png')

def check_for_overflow(param, before, added_value, after,
                       checkpoints=None,
                       file_name=None,
                       raise_exception=True,
                       max_value=None):
    """Check for overflow in param accumulation."""
    overflow = (abs(after) > max_value) if max_value is not None else \
               ((added_value > 0) and not (after > before)) or \
               ((added_value < 0) and not (after < before))
    if overflow:
        if checkpoints is not None:
            plot_recursion_checkpoints(checkpoints) # plot filters
            if file_name is not None:
                with open(file_name + '.pkl', "wb") as file: # save to disk
                    pickle.dump(checkpoints, file)
        if raise_exception:
            raise OverflowError(f'Overflow in {param} accumulation')


def plot_recursion_checkpoints(checkpoints,
                               ave_M=255,
                               ave_D=127,
                               ave_Lm=20,
                               ave_Ld=10,
                               keys=('L', 'I', 'D', 'M', 'dert_', 'rng', 'rdn', 'typ')):
    """Plot checkpoint value and filter progression."""
    # Filter expressions
    filters = (
        lambda checkpoints: ave_Lm * checkpoints['rdn'],
        lambda checkpoints: ave_D * checkpoints['rdn'],
        lambda checkpoints: ave_M / 2 * checkpoints['rdn'],
        lambda checkpoints: ave_Ld * checkpoints['rdn'],
        lambda checkpoints: ave_D * checkpoints['rdn'],
    )
    criterions = (
        lambda checkpoints: checkpoints['L'],
        lambda checkpoints: -checkpoints['M'],
        lambda checkpoints: checkpoints['M'],
        lambda checkpoints: checkpoints['L'],
        lambda checkpoints: checkpoints['D'],
    )

    # Convert check points to dict
    cps_dict = [*map(dict, map(lambda cp: zip(keys, cp), checkpoints))]

    # Value and filter sequences
    val_ = [criterions[cp['typ']](cp) for cp in cps_dict]
    filt_ = [filters[cp['typ']](cp) for cp in cps_dict]

    # Plot with linear scale
    plt.subplot(211)
    plt.plot(val_, label='values')
    plt.plot(filt_, label='filters')
    plt.legend()
    plt.ylabel('Value')
    plt.title('Checkpoints')

    # Plot with linear scale
    plt.subplot(212)
    plt.semilogy(val_, label='values')
    plt.semilogy(filt_, label='filters')
    plt.legend()
    plt.ylabel('Value (log-scale)')
    plt.xlabel('Recursion depth')
    plt.show()


def draw_pattern(P, rng, esize=1, sgn_index=0, dert_index=-1):
    """
    Take a CogAlg pattern. Return numpy.ndarray as an image
    represent the pattern.
    :param P: list, the pattern object to visualize.
    :param rng: integer, gap between actual elements.
    :param esize: integer, number of pixels each pattern's element take
    = esize^2.
    :param sgn_index: integer, index of sign of pattern in the object P.
    :param dert_index: integer, index of element list in the object P.
    :return out_img: numpy.ndarray, image represent the pattern.
    """
    sgn = P[sgn_index]
    value = sgn * 255
    elements = P[dert_index]
    height = esize
    width = esize * ((len(elements) - 1)*rng + 1)
    out_img = np.full((height, width), value, 'uint8')

    return out_img

def place_pattern(img, pattern_img, pos):
    """
    Place drawn pattern image onto an existing image
    :param img: numpy.ndarray, the image to draw on.
    :param pattern_img: numpy.ndarray, image of the pattern.
    :param pos: tuple, contain position on horizontal and vertical axes.
    :return result_img: numpy.ndarray, the result image.
    """

    x, y = pos
    h, w = pattern_img.shape
    img[y : y+h, x : x+w] = pattern_img

    return img

def draw_all_patterns(P__, shape, rng,
                      esize=1, sgn_index=0, dert_index=-1):
    """
    Draw all patterns into an image
    :param P__: list, nested list of patterns' data.
    :param shape: tuple, contain height and width of the output image.
    :param rng: int, gaps
    :param esize:
    :param sgn_index:
    :param dert_index:
    :return img: numpy.ndarray, the result image.
    """
    img = np.full(shape, 127, 'uint8')

    for y, P_ in enumerate(P__):
        x = 0
        for P in P_:
            P_img = draw_pattern(P, rng, esize, sgn_index, dert_index)
            place_pattern(img, P_img, (x, y))
            x += (len(P[dert_index]) - 1)*rng + 1

    return img