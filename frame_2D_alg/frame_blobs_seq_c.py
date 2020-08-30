from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from frame_blobs_yx import comp_pixel, ave
from class_cluster import ClusterStructure, NoneType
from utils import imread

class SDertRef(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_int),
    ]

class SBlob(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('S', c_ulonglong),
        ('sign', c_byte),
        ('fopen', c_byte),
        ('dert_ref', POINTER(SDertRef)),
    ]

class SFrameOfBlobs(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('nblobs', c_ulong),
        ('blobs', POINTER(SBlob)),
    ]

# Load derts2blobs function from C library
derts2blobs = CDLL("frame_blobs.so").derts2blobs
derts2blobs.restype = SFrameOfBlobs

class CBlob(ClusterStructure):
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool

def transfer_data(frame):
    """Transfer data from C structures to custom objects."""
    pass

def visualize_results(blobs, idmap):
    pass

if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpg')
    argument_parser.add_argument('-v', '--verbose', help='print details, useful for debugging', type=int, default=1)
    argument_parser.add_argument('-n', '--intra', help='run intra_blobs after frame_blobs', type=int, default=0)
    argument_parser.add_argument('-r', '--render', help='render the process', type=int, default=1)
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])
    verbose = arguments['verbose']
    intra = arguments['intra']
    render = arguments['render']

    start_time = time()
    dert__ = [*map(lambda a: a.astype('float64'),
                   comp_pixel(image))]
    height, width = dert__[0].shape
    idmap = np.empty((height, width), 'uint32')
    frame = derts2blobs(*map(lambda d: d.ctypes.data, dert__),
                        height, width, ave, idmap.ctypes.data)

    print(f"{frame.nblobs} blobs formed in {time() - start_time} seconds")
    plt.imshow(idmap, 'gray')
    plt.show()