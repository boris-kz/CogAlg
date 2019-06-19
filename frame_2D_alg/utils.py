from collections import deque

import numpy as np

from imageio import imsave
from PIL import Image

# ************ MAIN FUNCTIONS **************************************************
# -imread(): read image from file as gray-scale
# -draw(): output numpy array of pixel as image file.
# -map_sub_blobs(): given a blob and a traversing path, map all sub blobs of a
# specific branch belongs to that blob into a numpy array.
# -map_blobs(): map all blobs in blob_ into a numpy array.
# -map_blob(): map all segments in blob.seg_ into a numpy array.
# -map_segment(): map all Ps of a segment into a numpy array.
# -over_draw(): used to draw sub-structure's map onto to current level
# structure.
# -empty_map(): create a numpy array representing blobs' map.
# -segment_box(): find bounding box of given segment(sub-composite structure
# that is building block of blob).
# -localize(): translate bounding box against a reference.
# -shrunk(): return shape tuple that is shrunken by x units.
# -kernel(): compute single kernel.
# -generate_kernels(): return n-range kernels.
# ******************************************************************************

transparent_val = 127       # a pixel at this value is considered transparent

def imread(path):
    '''
    Read an image from file as gray-scale.
    Argument:
        - path: path of image for reading.
    Return: numpy array of gray-scaled image
    '''

    pil_image = Image.open(path).convert('L')
    image = np.array(pil_image.getdata()).reshape(*reversed(pil_image.size))
    return image


def draw(path, image, extension='.bmp'):
    '''
    Output into an image file.
    Arguments:
        - path: path for saving image file.
        - image: input as numpy array.
        - extension: determine file-type of ouput image.
    Return: None
    '''

    imsave(path + extension, image.astype('uint8'))
    return
    # ---------- draw() end ----------------------------------------------------


def map_sub_blobs(blob, traverse_path=[]):  # currently a draft

    '''
    Given a blob and a traversing path, map image of all sub-blobs of a specific
    branch belonging to that blob into a numpy array.
    Arguments:
        - blob: contain all mapped sub-blobs.
        - traverse_path: list of values determine the derivation sequence of
        target sub-blobs.
            + 0 for hypot_g/comp_gradient
            + 1 for comp_angle
            + 2 for comp_range
    Return: numpy array of image's pixel
    '''

    image = empty_map(blob.box)

    return image    # return filled image
    # ---------- map_sub_blobs() end -------------------------------------------


def map_frame(frame):
    '''
    Map the whole frame of original image as computed blobs.
    Argument:
        - frame: frame object input (as a list).
    Return: numpy array of image's pixel
    '''

    (I, G, Dy, Dx, blob_), dert__ = frame
    height, width = dert__.shape[1:]
    box = (0, height, 0, width)
    image = empty_map(box)

    for i, blob in enumerate(blob_):
        blob_map = map_blob(blob)

        over_draw(image, blob_map, blob.box, box)

    return image
    # ---------- map_frame() end -----------------------------------------------


def map_blob(blob, original=False):
    '''
    Map a single blob into an image.
    Argument:
        - blob: the input blob.
        - original: each pixel is the original image's pixel instead of just
        black or white to separate blobs.
    Return: numpy array of image's pixel
    '''

    blob_img = empty_map(blob.box)

    for seg in blob.seg_:

        sub_box = segment_box(seg)

        seg_map = map_segment(seg, sub_box, original)

        over_draw(blob_img, seg_map, sub_box, blob.box)

    return blob_img
    # ---------- map_blob() end ------------------------------------------------


def map_segment(seg, box, original=False):
    '''
    Map a single segment of a blob into an image.
    Argument:
        - seg: the input segment.
        - box: the input segment's bounding box.
        - original: each pixel is the original image's pixel instead of just
        black or white to separate blobs.
    Return: numpy array of image's pixel
    '''

    seg_img = empty_map(box)

    y0, yn, x0, xn = box

    for y, P in enumerate(seg[-1], start= seg[0] - y0):
        x0P= P[1]
        x0P -= x0
        derts_ = P[-1]
        for x, derts in enumerate(derts_, start=x0P):
            if original:
                seg_img[y, x] = derts[0][0]
            else:
                seg_img[y, x] = 255 if P[0] else 0

    return seg_img

    # ---------- map_segment() end ---------------------------------------------


def over_draw(map, sub_map, sub_box, box=None, tv=transparent_val):
    '''
    Over-write map of sub-structure onto map of parent-structure.
    Argument:
        - map: map of parent-structure.
        - sub_map: map of sub-structure.
        - sub_box: bounding box of sub-structure.
        - box: bounding box of parent-structure, for computing local coordinate
        of sub-structure.
    Return: over-written map of parent-structure
    '''

    if  box is None:
        y0, yn, x0, xn = sub_box
    else:
        y0, yn, x0, xn = localize(sub_box, box)
    map[y0:yn, x0:xn][sub_map != tv] = sub_map[sub_map != tv]
    return map
    # ---------- over_draw() end -----------------------------------------------


def empty_map(shape):
    '''
    Create an empty numpy array of desired shape.
    Argument:
        - shape: desired shape of the output.
    Return: over-written map of parent-structure
    '''

    if len(shape) == 2:
        height, width = shape
    else:
        y0, yn, x0, xn = shape
        height = yn - y0
        width = xn - x0

    return np.array([[transparent_val] * width] * height)


def segment_box(seg):
    y0s = seg[0]            # y0
    yns = y0s + seg[-2]     # Ly
    x0s = min([P[1] for P in seg[-1]])
    xns = max([P[1] + P[-2] for P in seg[-1]])
    return y0s, yns, x0s, xns


def localize(box, global_box):
    '''
    Compute local coordinates for given bounding box.
    Used for overwriting map of parent structure with
    maps of sub-structure, or other similar purposes.
    Arguments:
        - box: bounding box need to be localized.
        - global_box: reference to which box is localized.
    Return: box localized with localized coordinates
    '''
    y0s, yns, x0s, xns = box
    y0, yn, x0, xn = global_box

    return y0s - y0, yns - y0, x0s - x0, xns - x0

def shrunk(shape, x, axes=(0, 1)):
    '''Return shape tuple that is shrunken by x units.'''
    return tuple(X - x if axis in axes else X for axis, X in enumerate(shape))

def kernel(n):
    '''
    Return kernel for comparison.
    Note: dx kernel is transpose of dy kernel.
    '''

    # Compute symmetrical coefficients of kernel:
    sides = np.array([*range(2, n + 1, 2)] + [n - 1] * (n // 2 - 1))
    coefs = sides / np.hypot(sides, np.flip(sides))

    # Calculate pivot point (positioned at the corner of the kernel):
    odd = n % 2
    ipivot = (n - 1 - odd) // 2

    # Construct margins of kernel:
    vert_coefs = coefs[:ipivot]
    hor_coefs = coefs[ipivot:]
    if odd:
        vert_coefs = np.concatenate((-np.flip(vert_coefs), [0], vert_coefs))
        hor_coefs = np.concatenate((np.flip(hor_coefs), [1], hor_coefs))
    else:
        vert_coefs = np.concatenate((-np.flip(vert_coefs), vert_coefs))
        hor_coefs = np.concatenate((np.flip(hor_coefs), hor_coefs))

    # Assign coefficients to kernel:
    ky = np.zeros((n, n), dtype=float) # Initialize kernel for dy.
    ky[0, :] = -hor_coefs # Assign upper coefficients.
    ky[-1, :] = hor_coefs # Assign lower coefficients.
    ky[1:-1, 0] = vert_coefs # Assign left-side coefficients.
    ky[1:-1, -1] = vert_coefs # Assign right-side coefficients.

    ky /= n - 1 # Divide by comparison distance.

    kx = ky.T # Compute kernel for dx (transpose of ky).

    return np.stack((ky, kx), axis=0)


def generate_kernels(max_rng, k2x2=0):
    '''
    Generate a deque of kernels corresponding to max range.
    Arguments:
        - max_rng: maximum range of comparisons.
        - k2x2: if True, generate an additional 2x2 kernel.
    Return: box localized with localized coordinates'''
    indices = np.indices((max_rng, max_rng)) # Initialize 2D indices array.
    quart_full_kernel = indices / np.hypot(*indices[:]) # Compute coeffs.
    quart_full_kernel[:, 0, 0] = 0 # Fill na value with zero

    # Fill full dy kernel with the computed quadrant:
    # Fill bottom-left quadrant:
    half_full_kernel_y = np.concatenate(
                             (
                                 np.flip(
                                     quart_full_kernel[0, :, 1:],
                                     axis=1),
                                 quart_full_kernel[0],
                                 ),
                             axis=1,
                         )

    # Fill upper half:
    full_kernel_y = np.concatenate(
                        (
                            -np.flip(
                                half_full_kernel_y[1:],
                                axis=0),
                            half_full_kernel_y,
                            ),
                    axis=0,
                    )

    full_kernel = np.stack((full_kernel_y, full_kernel_y.T), axis=0)

    # Divide full kernel into deque of rng-kernels:
    k_ = deque() # Initialize deque of different size kernels.
    k = full_kernel # Initialize reference kernel.
    for rng in range(max_rng, 1, -1):
        rng_kernel = np.array(k) # Make a copy of k.
        rng_kernel[:, 1:-1, 1:-1] = 0 # Set central variables to 0.
        rng_kernel /= rng # Divide by comparison distance.
        k_.appendleft(rng_kernel)
        k = k[:, 1: -1, 1:-1] # Make k recursively shrunken.

    # Compute 2x2 kernel:
    if k2x2:
        coeff = full_kernel[0, -1, -1] # Get the value of square root of 0.5
        kernel_2x2 = np.array([[[-coeff, -coeff],
                                [coeff, coeff]],
                               [[-coeff, coeff],
                                [-coeff, coeff]]])

        k_.appendleft(kernel_2x2)

    return k_

# ------------------------------------------------------------------------------