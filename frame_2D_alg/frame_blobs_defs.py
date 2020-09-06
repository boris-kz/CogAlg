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
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool

class CDeepBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    M = int
    iDy = int
    iDx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    fcr = bool
    fig = bool
    rdn = float
    rng = int
    Ls = int  # for visibility and next-fork rdn
    sub_layers = list