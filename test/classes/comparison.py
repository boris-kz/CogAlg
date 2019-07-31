"""
Cderts
-------
Compute gradient of 2D input array.

See frame_2D_alg/frame_blobs.py for more information.
"""

import operator as op

import numpy as np
import numpy.ma as ma

from comp_i import (
    Y_COEFFS, X_COEFFS,
    central_slice,
    rim_mask,
    translated_operation,
    angle_diff,
)

# -----------------------------------------------------------------------------
# Module constants

# -----------------------------------------------------------------------------
# Cdert class

class Cderts(object):

    def __init__(self, i, rng):
        """Create an instance of Cderts from a numeric 2D array."""
        self._i = i
        self._rng = rng
        self._derts = object()
        return

    # -------------------------------------------------------------------------
    # Properties
    @property
    def i(self):
        return self._i

    @property
    def rng(self):
        return self._rng

    @property
    def derts(self):
        return self._derts

    @property
    def g(self):
        return self._derts[0]

    @property
    def dy(self):
        return self._derts[-2]

    @property
    def dx(self):
        return self._derts[-1]

    # -------------------------------------------------------------------------
    # Methods

    def _compare(self):
        """Generate derivatives from inputs."""
        dy, dx = compare_slices(self._i, self._rng)

        self._derts[-2:][central_slice(self._rng)] += (dy, dx)
        self._derts[0] = ma.hypot(*self._derts[-2:])

        self._derts[rim_mask(self._i.shape, rng)] = ma.masked
        return self

    def _angles(self):
        return self._derts[-2:] / self._derts[0]

    def recursive_compare(self, fork_type, mask=None):
        if mask is not None:
            self._derts[..., mask] = ma.masked

        if fork_type == 'r':
            self._rng += 1
            self._compare()
            return self
        elif fork_type == 'g':
            return Cgderts(self.g, self._rng)
        elif fork_type == 'a':
            return Caderts(self._angles(), self._rng)
        else:
            raise KeyError

# -----------------------------------------------------------------------------
# initderts Class

class initderts(Cderts):
    """Derivatives from input data."""

    def __init__(self, i : np.ndarray, rng : int = 1) -> None:
        """Create an instance of Cderts from a numeric 2D array."""
        Cderts.__init__(self, i, rng)
        self._derts = ma.zeros((3,) + self._i.shape)

        self._compare()
        return

# -----------------------------------------------------------------------------
# Cgdert Class

class Cgderts(Cderts):
    def __init__(self, g : ma.masked_array, rng : int = 1) -> None:
        Cderts.__init__(self, g, rng)

        self._derts = ma.zeros((4,) + self._i.shape)

        self._compare()
        return

    # -------------------------------------------------------------------------
    # Additional properties
    @property
    def m(self):
        return self._derts[1]

    # -------------------------------------------------------------------------
    # Methods

    def _compare(self):
        Cderts._compare(self)
        self._derts[1] += translated_operation(self._i, rng, ma.minimum)
        return self

class Caderts(Cderts):
    def __init__(self, a : ma.masked_array, rng : int = 1) -> None:
        Cderts.__init__(self, g, rng)

        self._derts = ma.zeros((5,) + self._i.shape)

        self._compare()
        return

    # -------------------------------------------------------------------------
    # Overriden properties
    @property
    def dy(self):
        return self._derts[1:3]

    @property
    def dx(self):
        return self._derts[3:5]

    # -------------------------------------------------------------------------
    # Methods

    def _compare(self):
        dy, dx = compare_slices(self._i, self._rng, operator=angle_diff)

        self._derts[1:][central_slice(self._rng)] += np.concatenate((dy, dx))
        self._derts[0] = ma.hypot(
            ma.arctan2(*self.dy),
            ma.arctan2(*self.dx),
        )

        self._derts[rim_mask(self._i.shape, rng)] = ma.masked
        return self

# -----------------------------------------------------------------------------
# Functions

def compare_slices(i, rng, operator=op.sub):
    d = translated_operation(i, rng, operator)
    return (d * Y_COEFFS).sum(axis=-1), (d * X_COEFFS).sum(axis=-1)

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------