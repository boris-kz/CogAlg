from itertools import (repeat, accumulate, chain, starmap, tee)
import numbers
import numpy as np
import numpy.ma as ma

from imageio import imsave
import cv2

# ----------------------------------------------------------------------------
# Constants

# colors
WHITE = 255
GREY = 128
BLACK = 0

transparent_val = 128 # Pixel at this value are considered transparent

SIGN_MAPS = {
    'binary': {
        False: BLACK,
        True: WHITE,
    },
    'ternary': {
        0: WHITE,
        1: BLACK,
        2: GREY
    },
}

# ----------------------------------------------------------------------------
# General purpose functions

def is_close(x1, x2):
    '''Recursively check equality of two objects containing floats.'''
    # Numeric
    if isinstance(x1, numbers.Number) and isinstance(x2, numbers.Number):
        return np.isclose(x1, x2)
    elif isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        try:
            return np.allclose(x1, x2)
        except ValueError as error_message:
            print(f'\nWarning: Error encountered for:\n{x1}\nand\n{x2}')
            print(f'Error: {error_message}')
            return False
    elif isinstance(x1, str) and isinstance(x2, str):
        return x1 == x2
    else:
        # Iterables
        try:
            if len(x1) != len(x2): # will raise an error if not iterable
                return False
            for e1, e2 in zip(x1, x2):
                if not is_close(e1, e2):
                    return False
            return True
        # Other types
        except TypeError:
            return x1 == x2


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


def array2image(a):
    "Rescale array values' range to 0-255."
    amin = a.min()
    return (255.99 * (a - amin) / (a.max() - amin)).astype('uint8')


def imread(filename, raise_if_not_read=True):
    "Read an image in grayscale, return array."
    try:
        return cv2.imread(filename, 0).astype(int)
    except AttributeError:
        if raise_if_not_read:
            raise SystemError('image is not read')
        else:
            print('Warning: image is not read')
            return None


def imwrite(filename, img):
    "Write image with cv2.imwrite."
    cv2.imwrite(filename, img)

# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Blob drawing

def map_sub_blobs(blob, traverse_path=[]):  # currently a draft

    '''
    Given a blob and a traversing path, map image of all sub-blobs
    of a specific branch belonging to that blob into a numpy array.
    Currently under development.
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


def map_frame_binary(frame, *args, **kwargs):
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

    for i, blob in enumerate(frame['blob__']):
        blob_map = draw_blob(blob, *args, **kwargs)

        over_draw(image, blob_map, blob['box'], box)

    return image


def map_frame(frame, *args, **kwargs):
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

    height, width = frame['gdert__'].shape[1:]
    box = (0, height, 0, width)
    image = blank_image(box)

    for i, blob in enumerate(frame['blob__']):
        blob_map = draw_blob(blob, *args, **kwargs)

        over_draw(image, blob_map, blob['box'], box)

    return image


def draw_blob(blob, *args, **kwargs):
    '''Map a single blob into an image.'''

    blob_img = blank_image(blob['box'])

    for stack in blob['stack_']:
        sub_box = stack_box(stack)
        stack_map = draw_stack(stack, sub_box, blob['sign'],
                               *args, **kwargs)
        over_draw(blob_img, stack_map, sub_box, blob['box'])

    return blob_img


def draw_stack(stack, box, sign,
               sign_map='binary'):
    '''Map a single stack of a blob into an image.'''

    if isinstance(sign_map, str) and sign_map in SIGN_MAPS:
        sign_map = SIGN_MAPS[sign_map]

    stack_img = blank_image(box)
    y0, yn, x0, xn = box

    for y, P in enumerate(stack['Py_'], start= stack['y0'] - y0):
        for x, dert in enumerate(P['dert__'], start=P['x0']-x0):
            if sign_map is None:
                stack_img[y, x] = dert[0]
            else:
                stack_img[y, x] = sign_map[sign]

    return stack_img


def stack_box(stack):
    y0s = stack['y0']           # y0
    yns = y0s + stack['Ly']     # Ly
    x0s = min([P['x0'] for P in stack['Py_']])
    xns = max([P['x0'] + P['L'] for P in stack['Py_']])
    return y0s, yns, x0s, xns


def debug_stack(background_shape, *stacks):
    image = blank_image(background_shape)
    for stack in stacks:
        sb = stack_box(stack)
        over_draw(image,
                  draw_stack(stack, sb, stack['sign']),
                  sb)
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

    return np.full((height, width), transparent_val)

# ---------------------------------------------------------------------
# ----------------------------------------------------------------------------