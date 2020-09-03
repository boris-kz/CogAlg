"""
's' or 'S' prefix for struct.
'c' or 'C' prefix for class.
'nt' prefix for namedtuple.
"""

from ctypes import *
import numpy as np
from frame_blobs_defs import CBlob, FrameOfBlobs
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
        ('box', c_uint * 4),
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

def transfer_data(sframe, dert__):
    """Transfer data from C structures to custom objects."""
    ntframe = FrameOfBlobs(
        I=sframe.I,
        G=sframe.G,
        Dy=sframe.Dy,
        Dx=sframe.Dx,
        blob_=[],
        dert__=dert__,
    )
    for i in range(sframe.nblobs):
        sblob = sframe.blobs[i]
        cblob = CBlob(
            I=sblob.I,
            G=sblob.G,
            Dy=sblob.Dy,
            Dx=sblob.Dx,
            S=sblob.S,
            box=list(sblob.box[:4]),
            sign=bool(sblob.sign),
            root_dert__=ntframe.dert__,
            adj_blobs=[[], 0, 0],
            fopen=bool(sblob.fopen),
        )
        cblob.dert_coord_.update(((sblob.dert_ref[i].y, sblob.dert_ref[i].x)
                                  for i in range(sblob.S)))

        ntframe.blob_.append(cblob)

    return ntframe

def cwrapped_derts2blobs(dert__):
    dert__ = [*map(lambda a: a.astype('float64'),
                   dert__)]
    height, width = dert__[0].shape
    idmap = np.empty((height, width), 'uint32')
    sframe = derts2blobs(*map(lambda d: d.ctypes.data, dert__),
                         height, width, idmap.ctypes.data)

    ntframe = transfer_data(sframe, dert__)

    return ntframe, idmap, set()

