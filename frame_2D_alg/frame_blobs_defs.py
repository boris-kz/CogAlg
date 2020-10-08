from collections import namedtuple
from class_cluster import ClusterStructure, NoneType

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, Dy, Dx, G, M, blob_, dert__')

class CBlob(ClusterStructure):
    # Dert params
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    # blob params
    S = int
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool


class CDeepBlob(ClusterStructure):
    # Dert params
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    # blob params
    S = int
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    fia = bool  # flag: input is from comp angle
    fca = bool  # flag: current fork is comp angle
    rdn = float
    rng = int
    Ls = int  # for visibility and next-fork rdn
    sub_layers = list