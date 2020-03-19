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
    Cross-comp of g or ga, in 2x2 kernels unless root fork is comp_r:
    odd=True or odd: sparse 3x3, is also effectively 2x2 input,
    recombined from one-line-distant lines?
    Parameters
    ----------
    dert__ : array-like
        The structure is (i, g, dy, dx) for dert or (ga, day, dax) for adert.
    odd : bool
        Initially False, set to True for comp_a and comp_g called from
        comp_r fork.
    Returns
    -------
    gdert__ : masked_array
        Output's structure is (g, gg, gdy, gdx, gm, ga, day, dax).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> odd = 'specific value'
    >>> comp_g(dert__, odd)
    'specific output'
    Notes
    -----
    Comparand is dert[1]
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
    With increasingly sparse input, unilateral rng (distance between
    central derts) can only increase as 2^(n + 1), where n starts at 0:
    rng = 1 : 3x3 kernel, skip orthogonally alternating derts as centrals,
    rng = 2 : 5x5 kernel, skip diagonally alternating derts as centrals,
    rng = 3 : 9x9 kernel, skip orthogonally alternating derts as centrals,
    ...
    That means configuration of preserved (not skipped) derts will always
    be 3x3.
    Parameters
    ----------
    dert__ : array-like
        dert's structure is (i, g, dy, dx, m).
    fig : bool
        True if input is g.
    Returns
    -------
    rdert__ : masked_array
        Output's structure is (i, g, dy, dx, m).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> fig = 'specific value'
    >>> comp_r(dert__, fig)
    'specific output'
    Notes
    -----
    - Results are accumulated in the input dert.
    - Comparand is dert[0].
    """
    pass

def comp_a(dert__, odd, aga):
    """
    cross-comp of a or aga, in 2x2 kernels unless root fork is
    comp_r: odd=True.
    Parameters
    ----------
    dert__ : array-like
        dert's structure is dependent to aga
    odd : bool
        Is True if root fork is comp_r.
    aga : bool
        If aga is True, dert's structure is interpreted as:
        (g, gg, gdy, gdx, gm, iga, iday, idax)
        Otherwise it is interpreted as:
        (i, g, dy, dx, m)
    Returns
    -------
    adert : masked_array
        adert's structure is (ga, day, dax).
    Examples
    --------
    >>> # actual python console code
    >>> dert__ = 'specific value'
    >>> odd = 'specific value'
    >>> aga = 'specific value'
    >>> comp_a(dert__, odd, aga)
    'specific output'
    """
    pass


def calc_a(dert__):
    """
    Compute vector representation of gradient angle.
    It is done by normalizing the vector (dy, dx).
    Numpy broad-casting is a viable option when the
    first dimension of input array (dert__) separate
    different CogAlg parameter (like g, dy, dx).
    Example
    -------
    >>> dert1 = np.array([0, 5, 3, 4])
    >>> a1 = calc_a(dert1)
    >>> print(a1)
    array([0.6, 0.8])
    >>> # 45 degrees angle
    >>> dert2 = np.array([0, 450**0.5, 15, 15])
    >>> a2 = calc_a(dert2)
    >>> print(a2)
    array([0.70710678, 0.70710678])
    >>> print(np.degrees(np.arctan2(*a2)))
    45.0
    >>> # -30 (or 330) degrees angle
    >>> dert3 = np.array([0, 10, -5, 75**0.5])
    >>> a3 = calc_a(dert3)
    >>> print(a3)
    array([-0.5      ,  0.8660254])
    >>> print(np.rad2deg(np.arctan2(*a3)))
    -29.999999999999996
    """
    return dert__[[2, 3]] / dert__[1]  # np.array([dy, dx]) / g



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