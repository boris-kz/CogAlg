from collections import defaultdict

import pickle
import numpy as np
import matplotlib.pyplot as plt


def save_pkl_file(obj, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(obj, file)


def load_pkl_file(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data


def save_frame_data(obj, file_name="frame_of_patterns_.pkl"):
    save_pkl_file(obj, file_name)


def load_frame_data(file_name="frame_of_patterns_.pkl"):
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
                       raise_exception=True):
    """Check for overflow in param accumulation."""
    if ((added_value > 0) and not (after > before)) or \
       ((added_value < 0) and not (after < before)):
        if checkpoints is not None:
            plot_recursion_checkpoints(checkpoints) # plot filters
            if file_name is not None:
                with open(file_name, "wb") as file: # save to disk
                    pickle.dump(checkpoints, file)
        if raise_exception:
            raise OverflowError(f'Overflow in {param} accumulation')


def plot_recursion_checkpoints(checkpoints,
                               ave_M=255,
                               ave_D=127,
                               ave_Lm=20,
                               ave_Ld=10,
                               keys=('L', 'I', 'D', 'M', 'rng', 'rdn', 'typ')):
    """Plot checkpoint value and filter progression."""
    # Filter expressions
    filters = (
        lambda checkpoints: ave_M * checkpoints['rdn'],
        lambda checkpoints: ave_Lm * checkpoints['rdn'],
        lambda checkpoints: ave_D * checkpoints['rdn'],
        lambda checkpoints: ave_Ld * checkpoints['rdn'],
        lambda checkpoints: ave_M / 2 * checkpoints['rdn'],
        lambda checkpoints: ave_D * checkpoints['rdn'],
    )
    criterions = (
        lambda checkpoints: checkpoints['M'],
        lambda checkpoints: checkpoints['L'],
        lambda checkpoints: -checkpoints['M'],
        lambda checkpoints: checkpoints['L'],
        lambda checkpoints: checkpoints['M'],
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


def draw_P(x0, X, y, P, pattern_images, rng=1, fork_label='p'):
    image = pattern_images[fork_label]
    image[y, x0:X] = 255 * P[0]

    # Deeper P check
    sP_ = None
    if len(P[5]) != 0:
        frng, fid, sub_ = P[5]
        if P[0]:
            rng += 1
            fork_label += '0'
        else:
            rng = 1
            fork_label += '2'
        sP_ = sub_
    elif fork_label[-1] not in ('1', '3'): # P is not P_seg
        if len(P[6]) != 0:
            fmm, fid, seg_ = P[6]
            if P[0]:
                rng = rng
                fork_label += '1'
            else:
                rng = 1
                fork_label += '3'
            frng, fid, sP_ = seg_

    if sP_ is not None:
        # Compute lost resolution
        x_lost = (X - x0) - sum([(sP[1]-1)*rng + 1 for sP in sP_])
        step = x_lost / len(sP_) # as float
        x_float = float(x0)
        for sP in sP_:
            next_x_float = x_float + step

            x = int(x_float)
            next_x = int(next_x_float)
            draw_P(x, next_x, y, sP, pattern_images, rng, fork_label)

            x_float = next_x_float


def visualize_patterns(frame, shape, ini_x=0, ini_y=0):

    pattern_images = defaultdict(lambda: np.full(shape, 0x80, dtype='uint8'))
    for y, P_ in enumerate(frame, ini_y):
        x = ini_x
        for P in P_:
            next_x = x + P[1] # next P is displaced by L units
            draw_P(x, next_x, y, P, pattern_images)
            x = next_x

    return patterns