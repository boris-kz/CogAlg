from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from frame_blobs_yx import comp_pixel, ave
from utils import imread

class DertRef(Structure):
    _fields_ = [
        ('x', c_int),
        ('y', c_int),
    ]

class Blob(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('S', c_ulonglong),
        ('sign', c_byte),
        ('fopen', c_byte),
        ('dert_ref', POINTER(DertRef)),
    ]

class FrameOfBlobs(Structure):
    _fields_ = [('blobs', POINTER(Blob)),
                ('nblobs', c_ulong)]

derts2blobs = CDLL("frame_blobs.so").derts2blobs
derts2blobs.restype = FrameOfBlobs

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