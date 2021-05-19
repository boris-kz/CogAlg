"""
's' or 'S' prefix for struct.
'c' or 'C' prefix for class.
'nt' prefix for namedtuple.
"""

from ctypes import *
import numpy as np

from frame_blobs_defs import CBlob, FrameOfBlobs
from utils import imread

class SLinkedListElement(Structure):
    pass

SLinkedListElement._fields_ = [
    ('val', c_longlong),
    ('next', POINTER(SLinkedListElement)),
]

class SLinkedList(Structure):
    _fields_ = [
        ('first', POINTER(SLinkedListElement)),
    ]

    def __iter__(self):
        current = self.first
        while current:
            yield current[0].val
            current = current[0].next


class SBlob(Structure):
    _fields_ = [
        ('I', c_double),
        ('iDy', c_double),
        ('iDx', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('M', c_double),
        ('S', c_ulonglong),
        ('sign', c_byte),
        ('box', c_uint * 4),
        ('fopen', c_byte),
    ]

class SFrameOfBlobs(Structure):
    _fields_ = [
        ('I', c_double),
        ('G', c_double),
        ('Dy', c_double),
        ('Dx', c_double),
        ('nblobs', c_ulong),
        ('blobs', POINTER(SBlob)),
        ('adj_pairs', SLinkedList),
    ]

# Load derts2blobs function from C library
loaded_library = CDLL("frame_blobs.so")
flood_fill = loaded_library.flood_fill
flood_fill.restype = POINTER(SFrameOfBlobs)
clean_up = loaded_library.clean_up
clean_up.argtypes = [POINTER(SFrameOfBlobs)]

def transfer_data(sframe, dert__, idmap, blob_cls):
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
        y0, yn, x0, xn = sblob.box[:4]
        cblob = blob_cls(
            I=sblob.I,
            G=sblob.G,
            Dy=sblob.Dy,
            Dx=sblob.Dx,
            S=sblob.S,
            sign=bool(sblob.sign),
            box=(y0, yn, x0, xn),
            root_dert__=ntframe.dert__,
            adj_blobs=[[], 0, 0],
            fopen=bool(sblob.fopen),
        )
        try:
            cblob.iDy = sblob.iDy
            cblob.iDx = sblob.iDx
            cblob.M = sblob.M
        except AttributeError:
            pass
        cblob.mask = (idmap[y0:yn, x0:xn] != i)  # or blob.id


        ntframe.blob_.append(cblob)

    adj_pairs = set((packed_pair >> 32, packed_pair & 0xFFFF)
                    for packed_pair in
                    sframe.adj_pairs)

    return ntframe, adj_pairs

def wrapped_flood_fill(dert__, blob_cls=CBlob):
    dert__ = [*map(lambda a: a.astype('float64'),
                   dert__)]
    height, width = dert__[0].shape
    idmap = np.empty((height, width), 'int64')

    sframe_ptr = flood_fill(*map(lambda d: d.ctypes.data, dert__),
                            height, width, idmap.ctypes.data,
                            blob_cls.instance_cnt,
                            int(blob_cls != CBlob))

    ntframe, adj_pairs = transfer_data(sframe_ptr[0], dert__, idmap, blob_cls)

    clean_up(sframe_ptr)

    return ntframe, idmap, adj_pairs