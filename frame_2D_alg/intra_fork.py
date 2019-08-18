"""
Fork-centric version of intra-blob.
"""

import operator as op
import numpy as np

from collections import deque, defaultdict
from functools import reduce
from itertools import starmap, takewhile
from comp_i import (
    comp_i, F_ANGLE, F_DERIV, F_RANGE,
)
from intra_blob import (
    form_P__, scan_P__, form_segment_,
    ave, rave, ave_blob, ave_intra_blob, ave_n_sub_blobs,
)

# -----------------------------------------------------------------------------
# Functions

def intra_fork(fork, flags, Ave_blob, Ave, rdn):
    """
    Recursive operation upon a fork.
    Form sub-fork of specific branch (compare input with incremented rng,
    compare maximal gradients, or compare angles of maximum gradients)
    """

    rdn += 1
    Ave_blob += ave_blob * rave * rdn
    Ave += ave * rdn

    # Form new sub-fork:
    sub_fork = form_sub_fork(fork, flags)

    # Clustering:
    for blob in fork['blob_']:
        Ave_blob = intra_clustering(blob, sub_fork, Ave, Ave_blob)
        Ave_blob *= rave  # estimated cost of redundant representations per blob
        Ave += ave  # estimated cost per dert

    # Remove smaller blobs:
    filter_blobs(sub_fork, Ave_blob)  # Filter below-Ave_blob sub-blobs.

    # Evaluations for deeper forks:
    if sub_fork['G'] > ave_intra_blob * rdn:
        intra_fork(sub_fork, F_ANGLE, Ave_blob, Ave, rdn+1)
        intra_fork(sub_fork, F_DERIV, Ave_blob, Ave, rdn+1)

    if (not flags & F_ANGLE # Angle fork don't have F_RANGE sub-fork
            and fork['G'] + sub_fork['M'] > ave_intra_blob * rdn):
        intra_fork(sub_fork, F_RANGE, Ave_blob, Ave, rdn+1)


def form_sub_fork(fork, flags):
    # Determine rng:
    if flags & F_RANGE:
        rng = fork['rng'] + 1
    else:
        rng = fork['rng']*2 + 1

    # Generate dert___:
    dert___ = comp_i(fork['dert___'], rng, flags)

    sub_fork = dict(fork_type=flags,
                    rng=rng,
                    dert___=dert___,
                    mask=None,
                    G=0, M=0, Dy=0, Dx=0, L=0, Ly=0, blob_=[])

    return sub_fork


def filter_blobs(fork, Ave_blob):
    """Filter and sort blobs of a fork."""
    # Filter below Ave_blob blobs:
    fork['blob_'] = [*takewhile(lambda blob: blob['Dert']['G'] > Ave_blob,
                                sorted(fork['blob_'],
                                       key=lambda blob: blob['Dert']['G'],
                                       reverse=True))]
    # noisy or directional G | Ga: > intra_clustering cost: rel root blob + sub_blob_

    # Mask irrelevant parts:
    fork['mask'] = reduce(merge_mask,
                          fork['blob_'],
                          np.ones(shape=fork['dert___'][-1].shape[1:],
                                  dtype=bool))


def intra_clustering(root_blob, fork, Ave, Ave_blob):
    """Cluster derts into blobs."""

    # Take dert__ inside root_blob's bounding box:
    dert__ = fork['dert___'][-1][root_blob['slices']] # Newly derived derts.
    # global_dert__ = fork['dert___'][-1]
    # local_slices = root_blob['slices']
    # dert__ = global_dert__[local_slices]

    y0, yn, x0, xn = root_blob['box']

    P__ = form_P__(dert__, Ave,
                   fa=flags&F_ANGLE,
                   ncomp=((2*fork['rng'] + 1)**2-1),
                   x0, y0) # Horizontal clustering
    P_ = scan_P__(P__)
    seg_ = form_segment_(P_)
    blob_ = form_blob_(seg_, root_blob, fork, flags)

    return Ave_blob * len(blob_) / ave_n_sub_blobs


def form_blob_(seg_, root_blob, fork, flags):
    encountered = []
    blob_ = []
    for seg in seg_:
        if seg in encountered:
            continue

        q = deque([seg])
        encountered.append(seg)

        s = seg['Py_'][0]['sign']
        G, M, Dy, Dx, L, Ly, blob_seg_ = 0, 0, 0, 0, 0, 0, []
        x0, xn = 9999999, 0
        while q:
            blob_seg = q.popleft()
            for ext_seg in blob_seg['fork_'] + blob_seg['root_']:
                if ext_seg not in encountered:
                    encountered.append(ext_seg)
            G += blob_seg['G']
            M += blob_seg['M']
            Dy += blob_seg['Dy']
            Dx += blob_seg['Dx']
            L += blob_seg['L']
            Ly += blob_seg['Ly']
            blob_seg_.append(blob_seg)

            x0 = min(x0, min(map(op.itemgetter('x0'), blob_seg['Py_'])))
            xn = max(xn, max(map(lambda P: P['x0']+P['L'], blob_seg['Py_'])))

        y0 = min(map(op.itemgetter('y0'), blob_seg_))
        yn = max(map(lambda segment: segment['y0']+segment['Ly'], blob_seg_))

        mask = np.ones((yn - y0, xn - x0), dtype=bool)
        for blob_seg in blob_seg_:
            for y, P in enumerate(blob_seg['Py_'], start=blob_seg['y0']):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y - y0, x_start:x_stop] = False

        # Form blob:
        blob = dict(
            G=G, M=M, Dy=Dy, Dx=Dx, L=L, Ly=Ly,
            sign=s,
            box=(y0, yn, x0, xn),  # boundary box
            seg_=blob_seg_,
            slices=(Ellipsis, slice(y0, yn), slice(x0, xn)), # For quick slicing from global dert__.
            mask=mask, # mask of this blob.
            fork=fork, # Contain all blobs (possibly from different root_blob) that belong to the same fork.
            root_blob=root_blob,
            child_forks=defaultdict(list), # Contain sub-blobs that belong to this blob.
            fork_types=flags, # fork_type of a blob is relative to it's root_blob.
        )

        feedback(blob)

        blob_.append(blob)

    return blob_


def feedback(blob, sub_fork_type=None): # Add each Dert param to corresponding param of recursively higher root_blob. Currently under revision.
    root_blob = blob['root_blob']
    if root_blob is None: # Stop recursion.
        return
    fork_type = blob['fork_type']

    # Last blob Layer is deeper than last root_blob Layer:
    len_sub_layers = max(0, 0, *map(len, blob['child_forks'].values()))
    if len(root_blob['child_forks'][fork_type]) == len_sub_layers:
        root_blob['child_forks'][fork_type].append((0, 0, 0, 0, 0, 0, []))

    # Global fork accumulation:
    G, M, Dy, Dx, L, Ly = blob['Dert'].values()
    blob['fork'].update(
        G=blob['fork']['G']+G,
        M=blob['fork']['M']+M,
        Dy=blob['fork']['Dy']+Dy,
        Dx=blob['fork']['Dx']+Dx,
        L=blob['fork']['L']+L,
        Ly=blob['fork']['Ly']+ly,
        blob_ + [blob],
    )

    # First layer accumulations:
    Gr, Mr, Dyr, Dxr, Lr, Lyr, sub_blob_ = root_blob['child_forks'][fork_type][0]
    root_blob['child_forks'][fork_type][0] = (
        Gr + G, Mr + M, Dyr + Dy, Dxr + Dx, Lr + L, Lyr + Ly,
        sub_blob_ + [blob],
    )

    # Accumulate deeper layers:
    root_blob['child_forks'][fork_type][1:] = \
        [*starmap( # Like map() except for taking multiple arguments.
            # Function (with multiple arguments):
            lambda Dert, sDert:
                (*starmap(op.add, zip(Dert, sDert)),), # Dert and sub_blob_ accum
            # Mapped iterables:
            zip(
                root_blob['child_forks'][fork_type][1:],
                blob['child_forks'][sub_fork_type][:],
            ),
        )]
    # Dert-only numpy.ndarray equivalent: (no sub_blob_ accumulation)
    # root_blob['forks'][fork_type][1:] += blob['forks'][fork_type]

    feedback(root_blob, fork_type)

# -----------------------------------------------------------------------------
# Utility functions

def merge_mask(mask, blob):
    mask[blob['slices']] &= blob['mask']
    return mask