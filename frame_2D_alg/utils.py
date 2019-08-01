"""
Provide utilities for all kinds of operations.
Categories:
- General purpose functions
- Blob slicing
- Blob drawing
- Comparison related
"""

from itertools import (
    repeat, chain, product, starmap, tee
)

from collections import deque

import numpy as np

from imageio import imsave
from PIL import Image

# -----------------------------------------------------------------------------
# Constants

transparent_val = 127 # Pixel at this value are considered transparent

rim_slices = {
    0:[ # For flattening outer rim of last two dimensions an ndarray:
        (..., 0, slice(None, -1)),
        (..., slice(None, -1), -1),
        (..., -1, slice(-1, 0, -1)),
        (..., slice(-1, 0, -1), 0),
    ],

    1:[# For flattening outer rim of last two dimensions an ndarray:
        (0, slice(None, -1), ...),
        (slice(None, -1), -1, ...),
        (-1, slice(-1, 0, -1), ...),
        (slice(-1, 0, -1), 0, ...),
    ]
}

# -----------------------------------------------------------------------------
# General purpose functions

def bipolar(iterable):
    "[0, 1, 2, 3] -> [(0, 3), (1, 2), (2, 1), (3, 0)]"
    it1, it2 = tee(iterable)
    return zip(it1,
               map(lambda x: None if x is None else -x,
                   reversed(list(it2))))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def imread(path):
    '''
    Read an image from file as gray-scale.

    Parameters
    ----------
    path : str
        Path of image for reading.

    Return
    ------
    out : ndarray
        A  2D array of gray-scaled pixels.
    '''

    pil_image = Image.open(path).convert('L')
    image = np.array(pil_image.getdata()).reshape(*reversed(pil_image.size))
    return image

def draw(path, image, extension='.bmp'):
    '''
    Output into an image file.

    Parameters
    ----------
    path : str
        String contain path for saving image file.
    image : ndarray
        Array of image's pixels.
    extension : str
        Determine file-type of ouputed image.
    '''

    imsave(path + extension, image.astype('uint8'))

def flattened_rim(a, arranged_d=0):

    return np.concatenate(tuple(map(lambda slices: a[slices],
                                    rim_slices[arranged_d])),
                          axis=arranged_d-1)
# -----------------------------------------------------------------------------
# Blob slicing

def slice_to_box(slice):
    """
    Convert slice object to tuple of bounding box.

    Parameters
    ----------
    slice : tuple
        A tuple containing slice(start, stop, step) objects.

    Return
    ------
    box : tuple
        A tuple containing 4 integers representing a bounding box.
    """

    box = (slice[0].start, slice[0].stop,
           slice[1].start, slice[1].stop)

    return box

def localize(box, global_box):
    '''
    Compute local coordinates for given bounding box.
    Used for overwriting map of parent structure with
    maps of sub-structure, or other similar purposes.

    Parameters
    ----------
    box : tuple
        Bounding box need to be localized.
    global_box : tuple
        Reference to which box is localized.

    Return
    ------
    out : tuple
        Box localized with localized coordinates.
    '''
    y0s, yns, x0s, xns = box
    y0, yn, x0, xn = global_box

    return y0s - y0, yns - y0, x0s - x0, xns - x0

def shrink(shape, x, axes=(0, 1)):
    '''Return shape tuple that is shrunken by x units.'''
    return tuple(X - x if axis in axes else X for axis, X in enumerate(shape))

# -----------------------------------------------------------------------------
# Blob drawing

def map_sub_blobs(blob, traverse_path=[]):  # currently a draft

    '''
    Given a blob and a traversing path, map image of all sub-blobs
    of a specific branch belonging to that blob into a numpy array.

    Parameters
    ----------
    blob : Blob
        Contain all mapped sub-blobs.
    traverse_path : list
        Determine the derivation sequence of target sub-blobs.

    Return
    ------
    out : ndarray
        2D array of image's pixel.
    '''

    image = empty_map(blob.box)

    return image    # return filled image

def map_frame(frame):
    '''
    Map partitioned blobs into a 2D array.

    Parameters
    ----------
    frame : Frame
        Contain blobs that needs to be mapped.

    Return
    ------
    out : ndarray
        2D array of image's pixel.
    '''

    I, G, Dy, Dx, blob_, i__, dert__ = frame.values()
    height, width = dert__.shape[1:]
    box = (0, height, 0, width)
    image = empty_map(box)

    for i, blob in enumerate(blob_):
        blob_map = map_blob(blob)

        over_draw(image, blob_map, blob['box'], box)

    return image

def map_blob(blob, original=False):
    '''Map a single blob into an image.'''

    blob_img = empty_map(blob['box'])

    for seg in blob['seg_']:

        sub_box = segment_box(seg)

        seg_map = map_segment(seg, sub_box, original)

        over_draw(blob_img, seg_map, sub_box, blob['box'])

    return blob_img

def map_segment(seg, box, original=False):
    '''Map a single segment of a blob into an image.'''

    seg_img = empty_map(box)

    y0, yn, x0, xn = box

    for y, P in enumerate(seg['Py_'], start= seg['y0'] - y0):
        derts_ = P['dert_']
        for x, derts in enumerate(derts_, start=P['x0']-x0):
            if original:
                seg_img[y, x] = derts[0][0]
            else:
                seg_img[y, x] = 255 if P['sign'] else 0

    return seg_img

def segment_box(seg):
    y0s = seg['y0']            # y0
    yns = y0s + seg['Ly']     # Ly
    x0s = min([P['x0'] for P in seg['Py_']])
    xns = max([P['x0'] + P['L'] for P in seg['Py_']])
    return y0s, yns, x0s, xns

def over_draw(map, sub_map, sub_box, box=None, tv=transparent_val):
    '''Over-write map of sub-structure onto map of parent-structure.'''

    if  box is None:
        y0, yn, x0, xn = sub_box
    else:
        y0, yn, x0, xn = localize(sub_box, box)
    map[y0:yn, x0:xn][sub_map != tv] = sub_map[sub_map != tv]
    return map

def empty_map(shape):
    '''Create an empty numpy array of desired shape.'''

    if len(shape) == 2:
        height, width = shape
    else:
        y0, yn, x0, xn = shape
        height = yn - y0
        width = xn - x0

    return np.array([[transparent_val] * width] * height)

# -----------------------------------------------------------------------------
# Comparison related

def generate_kernels(max_rng, k2x2=0):
    '''
    Generate a deque of kernels corresponding to max range.
    Deprecated. Use GenCoeff instead.

    Parameters
    ----------
    max_rng : int
        Maximum range of comparisons.
    k2x2 : int
        If True, generate an additional 2x2 kernel.

    Return
    ------
    out : deque
        Sequence of kernels for corresponding rng comparison.
    '''
    indices = np.indices((max_rng, max_rng)) # Initialize 2D indices array.
    quart_kernel = indices / np.hypot(*indices[:]) # Compute coeffs.
    quart_kernel[:, 0, 0] = 0 # Fill na value with zero

    # Fill full dy kernel with the computed quadrant:
    # Fill bottom-left quadrant:
    half_kernel_y = np.concatenate(
                        (
                            np.flip(
                                quart_kernel[0, :, 1:],
                                axis=1),
                            quart_kernel[0],
                        ),
                        axis=1,
                    )

    # Fill upper half:
    kernel_y = np.concatenate(
                   (
                       -np.flip(
                           half_kernel_y[1:],
                           axis=0),
                       half_kernel_y,
                   ),
                   axis=0,
                   )

    kernel = np.stack((kernel_y, kernel_y.T), axis=0)

    # Divide full kernel into deque of rng-kernels:
    k_ = deque() # Initialize deque of different size kernels.
    k = kernel # Initialize reference kernel.
    for rng in range(max_rng, 1, -1):
        rng_kernel = np.array(k) # Make a copy of k.
        rng_kernel[:, 1:-1, 1:-1] = 0 # Set central variables to 0.
        rng_kernel /= rng # Divide by comparison distance.
        k_.appendleft(rng_kernel)
        k = k[:, 1:-1, 1:-1] # Make k recursively shrunken.

    # Compute 2x2 kernel:
    if k2x2:
        coeff = kernel[0, -1, -1] # Get the value of square root of 0.5
        kernel_2x2 = np.array([[[-coeff, -coeff],
                                [coeff, coeff]],
                               [[-coeff, coeff],
                                [-coeff, coeff]]])

        k_.appendleft(kernel_2x2)

    return k_


def kernel(rng):
    '''
    Return coefficients for decomposition of d
    (compared over rng) into dy and dx.
    Here, we assume that kernel width is odd.
    '''
    # Start with array of indices:
    indices = np.indices((rng+1, rng+1))

    # Apply computations:
    quart_kernel = indices / np.hypot(*indices)
    quart_kernel[:, 0, 0] = 0

    # Copy quarter of kernel into full kernel:
    half_ky = np.concatenate(
        (
            np.flip(
                quart_kernel[0, :, 1:],
                axis=1),
            quart_kernel[0],
        ),
        axis=1,
    )

    ky = np.concatenate(
        (
            -np.flip(
                half_ky[1:],
                axis=0),
            half_ky,
        ),
        axis=0,
    )

    kx = ky.T  # Compute kernel for dx (transpose of ky).

    return np.stack((ky, kx), axis=0)

# -----------------------------------------------------------------------------
# GenCoeffs class

class GenCoeffs(object):
    """
    Generate coefficients used by comparisons
    of rng in {1, ..., max_rng}.
    """
    def __init__(self, max_rng=3):
        """
        Instanciate a GenCoeffs object.
        """
        self._generate_coeffs(max_rng)

    def _generate_coeffs(self, max_rng):
        """
        Workhorse of GenCoeffs class, compute kernel
        and separate into rng specific coefficients.
        """
        # Calculate combined kernel of rng from 1 to max_rng:
        kers = kernel(max_rng)

        # Separate into kernels of each rng and flatten them:
        self._coeffs = reversed(list(
            map(flattened_rim,
                map(lambda slices: kers[slices],
                    zip(repeat(...),
                        *tee(chain((slice(None, None),),
                                   map(lambda i: slice(i, -i), range(1, 3)),
                                   ))
                        ),
                    ),
                )
        ))

    def to_file(self, path="coeffs.py"):
        """Write coeffs to text file."""
        ycoeffs, xcoeffs = zip(*map(lambda coeff: (coeff[0], coeff[1]),
                                    self._coeffs,
                                    ))
        with open(path, "w") as file:
            file.write('import numpy as np\n')
            file.write('Y_COEFFS = {\n')
            for i, ycoeff in enumerate(ycoeffs, start=1):
                file.write(str(i) + ":np.\\\n" + repr(ycoeff) + ",\n")
            file.write('}\n')
            file.write('X_COEFFS = {\n')
            for i, xcoeff in enumerate(xcoeffs, start=1):
                file.write(str(i) + ":np.\\\n" + repr(xcoeff) + ",\n")
            file.write('}\n')
    @property
    def coeff(self):
        return self._coeffs

# -----------------------------------------------------------------------------
# GenTransSlice class

class GenTransSlice(object):
    """
    Generate slicing for vectorized comparisons.
    """
    def __init__(self, max_rng=3):
        """
        Instanciate a GenTransSlice object.
        """
        self._generate_slices(max_rng)

    def _generate_slices(self, max_rng):
        """Generate target slices for comparison function."""
        self._slices = []
        slice_inds = [*chain((None,), range(1, max_rng * 2 + 1))]

        for r in range(3, max_rng * 2 + 2, 2):
            slices = [*starmap(slice, bipolar(slice_inds[:r]))]
            slices = [*chain(slices,
                             repeat(slices[-1],
                                    r - 2),
                             reversed(slices),
                             repeat(slices[0],
                                    r - 2))]
            slices = [*zip(repeat(...), slices[-r+1:] + slices[:-r+1], slices)]
            self._slices.append(slices)

    def to_file(self, path="slices.py"):
        """Write coeffs to text file."""
        with open(path, "w") as file:
            file.write('TRANSLATING_SLICES = {\n')
            for i, slices in enumerate(self._slices, start=0):
                file.write(str(i) + ":[\n")
                for sl in slices:
                    file.write(str(sl) + ",\n")
                file.write("],\n")
            file.write('}')

    @property
    def slices(self):
        return self._slices

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------