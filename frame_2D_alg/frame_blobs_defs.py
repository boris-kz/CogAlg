from collections import namedtuple
from class_cluster import ClusterStructure, NoneType

FrameOfBlobs = namedtuple('FrameOfBlobs', 'I, G, Dy, Dx, blob_, dert__')

class CBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool


class CDeepBlob(ClusterStructure):
    # Derts
    I = int
    iDy = int
    iDx = int
    G = int
    Dy = int
    Dx = int
    M = int
    S = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    # other data
    sign = NoneType
    box = list
    mask = object
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    fcr = bool
    fig = bool
    fca = bool
    figa = bool
    rdn = float
    rng = int
    Ls = int  # for visibility and next-fork rdn
    sub_layers = list