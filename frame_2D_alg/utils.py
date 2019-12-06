from itertools import (repeat, accumulate, chain, starmap, tee)
import numpy as np
from imageio import imsave
import cv2

import argparse
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon.jpg')
arguments = vars(argument_parser.parse_args())

# -----------------------------------------------------------------------------
# Constants
transparent_val = 128 # Pixel at this value are considered transparent

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

    image = blank_image(blob.box)

    return image    # return filled image

def map_frame(frame, raw=False):
    '''
    Map partitioned blobs into a 2D array.
    Parameters
    ----------
    frame : dict
        Contains blobs that need to be mapped.
    raw : bool
        Draw raw values instead of boolean.
    Return
    ------
    out : ndarray
        2D array of image's pixel.
    '''

    height, width = frame['dert__'].shape[1:]
    box = (0, height, 0, width)
    image = blank_image(box)

    for i, blob in enumerate(frame['blob_']):
        blob_map = draw_blob(blob, raw)

        over_draw(image, blob_map, blob['box'], box)

    return image

def draw_blob(blob, raw=False):
    '''Map a single blob into an image.'''

    blob_img = blank_image(blob['box'])

    for seg in blob['seg_']:
        sub_box = segment_box(seg)
        seg_map = draw_segment(seg, sub_box, blob['sign'], raw)
        over_draw(blob_img, seg_map, sub_box, blob['box'])

    return blob_img

def draw_segment(seg, box, s, raw=False):
    '''Map a single segment of a blob into an image.'''

    seg_img = blank_image(box)
    y0, yn, x0, xn = box

    for y, P in enumerate(seg['Py_'], start= seg['y0'] - y0):
        for x, dert in enumerate(P['dert_'], start=P['x0']-x0):
            if raw:
                seg_img[y, x] = dert[0]
            else:
                seg_img[y, x] = 255 if s else 0

    return seg_img

def segment_box(seg):
    y0s = seg['y0']            # y0
    yns = y0s + seg['Ly']     # Ly
    x0s = min([P['x0'] for P in seg['Py_']])
    xns = max([P['x0'] + P['L'] for P in seg['Py_']])
    return y0s, yns, x0s, xns

def debug_segment(background_shape, *segments):
    image = blank_image(background_shape)
    for seg in segments:
        seg_box = segment_box(seg)
        over_draw(image,
                  draw_segment(seg, seg_box, seg['sign']),
                  seg_box)
    return image

def debug_blob(background_shape, *blobs):
    image = blank_image(background_shape)
    for blob in blobs:
        over_draw(image,
                  draw_blob(blob),
                  blob['box'])
    return image

def over_draw(map, sub_map, sub_box, box=None, tv=transparent_val):
    '''Over-write map of sub-structure onto map of parent-structure.'''

    if  box is None:
        y0, yn, x0, xn = sub_box
    else:
        y0, yn, x0, xn = localize(sub_box, box)
    map[y0:yn, x0:xn][sub_map != tv] = sub_map[sub_map != tv]
    return map

def blank_image(shape):
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

def kernel(rng):
    """
    Return coefficients for decomposition of d
    (compared over rng) into dy and dx.
    Here, we assume that kernel width is odd.
    """
    # Start with array of indices:
    indices = np.indices((rng+1, rng+1))

    # Apply computations:
    quart_kernel = indices / (indices**2).sum(axis=0)
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

# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
'''
def imread(path):
    # Read an image from file as gray-scale.
    # path : str
    # Path of image for reading.
    # Return
    # out : ndarray  # 2D array of gray-scaled pixels.

    pil_image = Image.open(path).convert('L')
    image = np.array(pil_image.getdata()).reshape(*reversed(pil_image.size))
    return image

def imwrite(path, image, extension='.bmp'):
    # Output into an image file.
    # path : str
    # String contain path for saving image file.
    # image : ndarray  # Array of image's pixels.
    # extension : str  # Determine file-type of ouputed image.

    imsave(path + extension, image.astype('uint8'))
    
from PIL import Image
from utils import imread

image_path = "./../images/raccoon.jpg"
image = cv2.imread(image_path).astype(int)
'''
