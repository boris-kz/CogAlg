"""
Cross-comparison of pixels, angles, or gradients, in 2x2 or 3x3 kernels
"""
import numpy as np
import numpy.ma as ma
# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Functions

def comp_g(dert__, odd):
    """
    cross-comp of g or ga, in 2x2 kernels unless root fork is comp_r: odd=TRUE

    >>> dert = i, g, dy, dx
    >>> adert = ga, day, dax
    >>> odd = bool  # initially FALSE, set to TRUE for comp_a and comp_g called from comp_r fork
    # input is not passed, always = dert[1]
    <<< gdert = g, gg=0, gdy=0, gdx=0, gm=0, ga, day, dax
    """
    pass


def comp_r(dert__, fig):
    """
    Cross-comp of input param (dert[0]) over rng set in intra_blob.
    This comparison is selective for blobs with below-average gradient,
    where input intensity doesn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping
    alternating derts as a kernel-central dert at current comparison range,
    which forms increasingly sparse input dert__ for greater range cross-comp,
    while maintaining one-to-one overlap between kernels of compared derts.
    With increasingly sparse input, unilateral rng between central derts c
    an only increase as 2^(n + 1), where n starts at 0:

    rng = 1 : 3x3 kernel, skip orthogonally alternating derts as centrals,
    rng = 2 : 5x5 kernel, skip diagonally alternating derts as centrals,
    rng = 3 : 9x9 kernel, skip orthogonally alternating derts as centrals,
    ...
    That means configuration of preserved (not skipped) derts will always be 3x3.
    Parameters
    ----------
    dert__ : array-like
        Array containing inputs.
    fig : bool
        Set to True if input is g or derived from g
    Returns
    -------
    out : masked_array
        The results.
    Example
    -------
    >>> dert = i, g, dy, dx, m
    <<< rdert = i, g, dy, dx, m
    # results are accumulated in the input dert
    # comparand is always i
    """


def comp_a(dert__, odd, aga):
    """
    cross-comp of a or aga, in 2x2 kernels unless root fork is comp_r: odd=TRUE
    if aga:
        >>> dert = i, g, dy, dx, m
    else:
        iga, iday, idax
    <<< adert = ga, day, dax
    """
    pass


def calc_a(dert__, inp):
    """Compute angles of gradient."""
    return dert__[inp[1:]] / dert__[inp[0]]
    # please add comments


def calc_aga(dert__, inp):
    """Compute angles of angles of gradient."""
    g__ = dert__[inp[1]]
    day__ = np.arctan2(*dert__[inp[1:3]])
    dax__ = np.arctan2(*dert__[inp[3:]])
    return np.stack((day__, dax__)) / g__

# -----------------------------------------------------------------------------
# Utility functions

def angle_diff(a2, a1):
    """
    Return the sine(s) and cosine(s) of the angle between a2 and a1.
    Parameters
    ----------
    a1 , a2 : array-like
        Each contains sine and cosine of corresponding angle,
        in that order. For vectorized operations, sine/cosine
        dimension must be the first dimension.
    Return
    ------
    out : MaskedArray
        The first dimension is sine/cosine of the angle(s) between
        a2 and a1.
    Note
    ----
    This only works for angles in 2D space.
    """
    return ma.array([a1[1] * a2[0] - a1[0] * a2[1],
                     a1[0] * a2[0] + a1[1] * a2[1]])

    # OLD VERSION OF angle_diff
    # # Extend a1 vector(s) into basis/bases:
    # y, x = a1
    # bases = [(x, -y), (y, x)]
    # transform_mat = ma.array(bases)
    #
    # # Apply transformation:
    # da = ma.multiply(transform_mat, a2).sum(axis=1)

    # return da

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------