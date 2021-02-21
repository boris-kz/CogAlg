'''
    This is a preprocessing for a terminal fork of intra_blob: comp_slice.
    Slice_blob converts selected smooth-edge blobs (high G, low Ga) into sliced blobs,
    adding horizontal blob slices: Ps or 1D patterns
'''
import sys
from collections import deque
from class_cluster import ClusterStructure, NoneType
from slice_utils import *

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback, not needed here
aveG = 50  # filter for comp_g, assumed constant direction
flip_ave = 2000

class CP(ClusterStructure):
    I = int
    Dy = int
    Dx = int
    G = int
    M = int
    Dyy = int
    Dyx = int
    Dxy = int
    Dxx = int
    Ga = int
    Ma = int
    L = int
    x0 = int
    sign = NoneType  # sign of gradient deviation
    dert_ = list  # array of pixel-level derts: (p, dy, dx, g, m) tuples
    gdert_ = list
    Dg = int
    Mg = int
    downconnect_ = list
    upconnect_ = list

# Functions:

def slice_blob(blob, verbose=False):

    flip_eval(blob)
    dert__ = blob.dert__
    mask__ = blob.mask__

    height, width = dert__[0].shape
    if verbose: print("Converting to image...")
    P__ = []  # frame of Ps
    _P_ = []  # upper row Ps

    for y, dert_ in enumerate(zip(*dert__)):  # first and last row are discarded?
        if verbose: print(f"\rProcessing line {y + 1}/{height}, ", end=""); sys.stdout.flush()

        P_ = form_P_(list(zip(*dert_)), mask__[y])  # horizontal clustering
        if _P_:
            scan_P_(_P_, P_) # check connectivity between Ps
        _P_ = P_ # set current row P as next row's upper row P
        P__.append(P_)

    blob.P__ = P__ # update blob's P__ reference

'''
Parameterized connectivity clustering:
- form_P sums dert params within P and increments its L: horizontal length.
- scan_P_ searches for horizontal (x) overlap between Ps of consecutive (in y) rows.
dert: tuple of derivatives per pixel, initially (p, dy, dx, g), extended in intra_blob
Dert: params of cluster structures (P, stack, blob): summed dert params + dimensions: vertical Ly and area A
'''

def form_P_(idert_, mask_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    dert_ = [list(idert_[0])]  # get first dert from idert_ (generator/iterator)
    _mask = mask_[0]  # mask bit per dert
    if ~_mask:
        I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert_[0]; L = 1; x0=0  # initialize P params with first dert

    for x, dert in enumerate(idert_[1:], start=1):  # left to right in each row of derts
        mask = mask_[x]  # masks = 1,_0: P termination, 0,_1: P initialization, 0,_0: P accumulation:
        if mask:
            if ~_mask:  # _dert is not masked, dert is masked, terminate P:
                P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_)
                P_.append(P)
        else:  # dert is not masked
            if _mask:  # _dert is masked, initialize P params:
                I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma = dert; L = 1; x0=x; dert_=[dert]
            else:
                I += dert[0]  # _dert is not masked, accumulate P params with (p, dy, dx, g, m, dyy, dyx, dxy, dxx, ga, ma) = dert
                Dy += dert[1]
                Dx += dert[2]
                G += dert[3]
                M += dert[4]
                Dyy += dert[5]
                Dyx += dert[6]
                Dxy += dert[7]
                Dxx += dert[8]
                Ga += dert[9]
                Ma += dert[10]
                L += 1
                dert_.append(dert)
        _mask = mask

    if ~_mask:  # terminate last P in a row
        P = CP(I=I, Dy=Dy, Dx=Dx, G=G, M=M, Dyy=Dyy, Dyx=Dyx, Dxy=Dxy, Dxx=Dxx, Ga=Ga, Ma=Ma, L=L, x0=x0, dert_=dert_)
        P_.append(P)

    return P_


def scan_P_(_P_, P_):

    for _P in _P_:  # upper row Ps
        for P in P_:  # lower row Ps

            if _P.x0 - 1 < (P.x0 + P.L) and (_P.x0 + _P.L) + 1 > P.x0:   # x overlap between P and _P in 8 directions:
                _P.downconnect_.append(P)
                P.upconnect_.append(_P)

            elif (_P.x0 + _P.L) < P.x0:  # stop scanning the rest of lower row Ps if there is no overlap
                break


def flip_eval(blob):

    horizontal_bias = ( blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0]) \
                      * (abs(blob.Dy) / abs(blob.Dx))
        # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: , preferential comp over low G

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):

        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})