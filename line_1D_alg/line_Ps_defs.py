from class_cluster import ClusterStructure, NoneType


# line_patterns - class initialization
class Cdert(ClusterStructure):
    p = int
    d = int
    m = int


class CP(ClusterStructure):
    sign = NoneType
    L = int
    I = int
    D = int
    M = int
    dert_ = list
    sub_layers = list
    smP = NoneType
    fdert = NoneType


# line_PPs - class initialization
class Cdert_P(ClusterStructure):
    smP = NoneType
    MP = int
    Neg_M = int
    Neg_L = int
    P = list
    ML = int
    DL = int
    MI = int
    DI = int
    MD = int
    DD = int
    MM = int
    DM = int


class CPP(ClusterStructure):
    smP = NoneType
    MP = int
    Neg_M = int
    Neg_L = int
    P_ = list
    ML = int
    DL = int
    MI = int
    DI = int
    MD = int
    DD = int
    MM = int
    DM = int