"""
Provide some tools to manipulate CogAlg frames and for debugging.
"""

import pickle
from collections import defaultdict
from itertools import chain, tee
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# miscs

def try_extend(lol, i, l):
    """
    Try to extend lol[i] with l.
    If IndexError, append [l] to lol instead.
    """
    try:
        lol[i].extend(l)
    except IndexError:
        lol.append(l)

# -----------------------------------------------------------------------------
# itertools recipes

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

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


def draw_pattern(P, rng, esize=1,
                 sgn_index=0, dert_index=-2,
                 sgn_typ='binary'):
    """
    Take a CogAlg pattern. Return numpy.ndarray as an image
    represent the pattern.
    :param P: tuple
        The pattern object to visualize.
    :param rng: integer
        Gap between actual elements.
    Optional parameters:
    :param esize: integer
        Number of pixels each pattern's element take = esize^2.
    :param sgn_index: integer
        Index of sign of pattern in the object P.
    :param dert_index: integer
        Index of element list in the object P.
    :param sgn_typ: str
        'binary' or 'ternary'.
    :return out_img: numpy.ndarray
        Image represent the pattern.
    """
    sgn = P[sgn_index]
    if sgn_typ == 'binary':
        fill_value = 255 if sgn else 0
    elif sgn_typ == 'ternary':
        fill_value = [0, 0, 0]
        fill_value[sgn] = 255 # red, green or blue
    else:
        raise ValueError('unknown sign type')

    elements = P[dert_index]
    height = esize
    width = esize * ((len(elements) - 1)*rng + 1)

    out_img = np.empty((height, width, 3), 'uint8')
    out_img[:] = fill_value

    return out_img, width

def place_pattern(img, pattern_img, pos):
    """
    Place drawn pattern image onto an existing image
    :param img: numpy.ndarray
        The image to draw on.
    :param pattern_img: numpy.ndarray
        Image of the pattern.
    :param pos: tuple
        Contain position on horizontal and vertical axes.
    :return result_img: numpy.ndarray
        The result image.
    """

    x, y = pos
    h, w, _ = pattern_img.shape
    img[y : y+h, x : x+w] = pattern_img

    return img

def draw_all_patterns(P__, shape, rng=1, esize=1,
                      sgn_index=0, dert_index=-2):
    """
    Draw all patterns into an image
    :param P__: list
        Nested list of patterns' data.
    :param shape: tuple
        Contain height and width of the output image.
    Optional parameters:
    :param rng, esize, sgn_index, dert_index:
        See draw_pattern for more information
    :return img: numpy.ndarray
        The result image.
    """
    img = np.full(shape+(3,), 0, 'uint8')

    for row, P_ in enumerate(P__):
        y = row * esize
        x = 0
        for P in P_:
            P_img, step = draw_pattern(P, rng, esize, sgn_index, dert_index)
            place_pattern(img, P_img, (x, y))
            x += step

    return img


def extract_sub_patterns(P, layers,
                         sub_index=7,
                         lateral_sub_keys = (
                             'sign', 'lL', 'fseg',
                             'fid', 'sub_rdn', 'rng',
                             'lateral_sub_'),
                         **filters,
                         ):
    """
    Get sub patterns of specified layer(s).
    :param P: tuple
        Pattern object containing sub patterns to extract.
    :param layers: int or list
        Layer or layers to extract.
    Optional parameters:
    :param sub_index: int, optional
        Index of sub_ in the object P.
    :param lateral_sub_keys: list, optional
        Index of sub_ in the object P.
    :param filters: keyword arguments
        Value of params to be met for example rng = 1, fid = True...
    :return sub_P_: list
        A list of sub patterns.
    """
    sub_ = P[sub_index]
    sub_P_ = []

    lat_sub_param_dict = \
        dict([(k, v) for v, k in enumerate(lateral_sub_keys)])

    if isinstance(layers, int):
        layers = [layers]

    for layer in layers:
        try:
            lat_sub_ = sub_[layer]
        except IndexError:
            break
        sub_P_.extend([*flatten(
            [lat_sub[lat_sub_param_dict['lateral_sub_']]
             for lat_sub in lat_sub_
             if not sum([lat_sub[lat_sub_param_dict[key]] != value
                         for key, value in filters.items()]) > 0])])

    return sub_P_