"""
Script for testing 2D alg.
Change quickly in parallel with development.
Currently testing: intra_blob.form_P__
"""

import frame_blobs_ternary

from utils import imread, draw
from intra_comp_ts import comp_i
from intra_blob_draft import form_P__, scan_P__

# -----------------------------------------------------------------------------
# Adjustable parameters

image_path = "../images/raccoon_eye.jpg"
output_path = "../visualization/images/2D_alg_test_out"

# -----------------------------------------------------------------------------
# Adjustable parameters

def normalize(a):
    return (a - a.min()) / (a.max() - a.min())

if __name__ == "__main__":
    print('Reading image...')
    image = imread(image_path)
    print('Done!')

    print('Doing first comp...')
    frame = frame_blobs_ternary.image_to_blobs(image)
    print('Done!')

    print('Extracting best blob...')
    best_blob = sorted(frame['blob_'],
                       key=lambda blob: blob['Dert']['G'])[0]
    print('Done!')

    print('Doing angle comp on best blob...')
    derts = comp_i(best_blob['dert__'], 1, fa=1)
    print('Done!')

    print('Outputting derts...')
    draw(output_path, 255 * normalize(derts[4]))
    print('Done!')

    print('Running form_P__...')
    y0, yn, x0, xn = best_blob['box']
    P__ = form_P__(x0, y0, derts, 50, fa=1)
    print('Done!')

    print('Running scan_P__...')
    P_ = scan_P__(P__)
    print('Done!')

    """
    Cderts
    -------
    Compute gradient of 2D input array.
    See frame_2D_alg/frame_blobs_ternary.py for more information.
    """

    import operator as op

    import numpy as np
    import numpy.ma as ma

    from intra_comp_ts import (
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

        def __init__(self, i: np.ndarray, rng: int = 1) -> None:
            """Create an instance of Cderts from a numeric 2D array."""
            Cderts.__init__(self, i, rng)
            self._derts = ma.zeros((3,) + self._i.shape)

            self._compare()
            return


    # -----------------------------------------------------------------------------
    # Cgdert Class

    class Cgderts(Cderts):
        def __init__(self, g: ma.masked_array, rng: int = 1) -> None:
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
        def __init__(self, a: ma.masked_array, rng: int = 1) -> None:
            Cderts.__init__(self, g, rng)

            self._derts = ma.zeros((5,) + self._i.shape)

            self._compare()
            return

        # -------------------------------------------------------------------------
        # Overridden properties
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