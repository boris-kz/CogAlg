'''
  line_Ps is a principal version of 1st-level 1D algorithm
  Operations:
  -
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|,
  rather than a minimum for directly defined match: albedo of an object doesn't correlate with its predictive value.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they do overlap in line_patterns_olp.
'''

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath

from numpy import int16, int32
sys.path.insert(0, abspath(join(dirname("CogAlg"), '../..')))
import cv2
# import argparse
import pickle
from time import time
from matplotlib import pyplot as plt
from itertools import zip_longest
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param

class Cdert(ClusterStructure):
    i = int  # input for range_comp only
    p = int  # accumulated in rng
    d = int  # accumulated in rng
    m = int  # distinct in deriv_comp only
    mrdn = int  # 1 if abs d is stronger than m, same as ~m_sign here

class CP(ClusterStructure):
    L = int
    I = int
    D = int
    M = int  # summed ave - abs(d), different from D
    Rdn = lambda: 1.0  # 1 + binary dert.mrdn cnt / len(dert_)
    x0 = int
    dert_ = list  # contains (i, p, d, m, mrdn)
    subset = list  # 1st sublayer' rdn, rng, xsub_pmdertt_, _xsub_pddertt_, sub_Ppm_, sub_Ppd_
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ dert_], 3rd: sublayers[ sub_P_[ dert_]]:
    sublayers = list  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
    subDertt_ = list  # m,d' [L,I,D,M] per sublayer, conditionally summed in line_PPs
    derDertt_ = list  # for subDertt_s compared in line_PPs

verbose = False
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_splice = 50  # to merge a kernel of 3 adjacent Ps
init_y = 500  # starting row, set 0 for the whole frame, mostly not needed
halt_y = 501  # ending row, set 999999999 for arbitrary image
'''
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
'''

def line_Ps_root(pixel_):  # Ps: patterns, converts frame_of_pixels to frame_of_patterns, each pattern may be nested

    dert_ = []  # line-wide i_, p_, d_, m_, mrdn_
    _i = pixel_[0]
    # cross_comparison:
    for i in pixel_[1:]:  # pixel i is compared to prior pixel _i in a row:
        d = i - _i  # accum in rng
        p = i + _i  # accum in rng
        m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
        mrdn = m < 0  # 1 if abs(d) is stronger than m, redundant here
        dert_.append( Cdert( i=i, p=p, d=d, m=m, mrdn=mrdn) )
        _i = i

    if logging:
        with open("layer0_log_py.csv", "a") as csvFile_0:
            write = csv.writer(csvFile_0, delimiter=",")
            for id, val in enumerate(dert_):
                write.writerow([val.i, val.p, val.d, val.m, val.mrdn])

    # form patterns, evaluate them for rng+ and der+ sub-recursion of cross_comp:
    Pm_ = form_P_(None, dert_, rdn=1, rng=1, fPd=False)  # rootP=None, eval intra_P_ (calls form_P_)
    Pd_ = form_P_(None, dert_, rdn=1, rng=1, fPd=True)

    return [Pm_, Pd_]  # input to 2nd level


def form_P_(rootP, dert_, rdn, rng, fPd):  # accumulation and termination, rdn and rng are pass-through to rng+ and der+
    # initialization:
    P_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0
        if sign != _sign:
            # sign change, initialize and append P
            P = CP( L=1, I=dert.p, D=dert.d, M=dert.m, Rdn=dert.mrdn+1, x0=x, dert_=[dert], sublayers=[])  # Rdn starts from 1
            P_.append(P)  # updated with accumulation below
        else:
            # accumulate params:
            P.L += 1; P.I += dert.p; P.D += dert.d; P.M += dert.m; P.Rdn += dert.mrdn
            P.dert_ += [dert]
        x += 1
        _sign = sign
    '''
    due to separate aves, P may be processed by both or neither of r fork and d fork
    add separate rsublayers and dsublayers?
    '''
    # range_incr_P_(rootP, P_, rdn, rng)
    # deriv_incr_P_(rootP, P_, rdn, rng)
    if logging:

        if fPd == False:
            logfile_name = "layer1_Pm_log_py.csv"
        else:
            logfile_name = "layer1_Pd_log_py.csv"

        with open(logfile_name, "a") as csvFile_1:
            write = csv.writer(csvFile_1, delimiter=",")
            for id, val in enumerate(P_):
                write.writerow([val.L, val.I, val.D, val.M, val.Rdn, val.x0, val.dert_, val.sublayers])

    return P_  # used only if not rootP, else packed in rootP.sublayers


if __name__ == "__main__":
    ''' 
    Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    '''
    render = 0
    fline_PPs = 0
    frecursive = 0
    logging = 1  # logging of local functions variables

    if logging:
        import csv
        # with open("layer0_log_py.csv", "w") as csvFile_0:
        #     write = csv.writer(csvFile_0, delimiter=",")
        #     parameter_names = ["i", "p", "d", "m", "mrdn"]
        #     write.writerow(parameter_names)

        with open("layer1_Pm_log_py.csv", "w") as csvFile_1:
            write = csv.writer(csvFile_1, delimiter=",")
            parameter_names = ["L", "I", "D", "M", "Rdn", "x0", "dert_", "sublayers"]
            write.writerow(parameter_names)

        with open("layer1_Pd_log_py.csv", "w") as csvFile_1:
            write = csv.writer(csvFile_1, delimiter=",")
            parameter_names = ["L", "I", "D", "M", "Rdn", "x0", "dert_", "sublayers"]
            write.writerow(parameter_names)

    start_time = time()
    image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
    assert image is not None, "No image in the path"

    # Main
    Y, X = image.shape  # Y: frame height, X: frame width
    frame = []
    for y in range(init_y, min(halt_y, Y)):  # y is index of new row pixel_, we only need one row, use init_y=0, halt_y=Y for full frame

        line = line_Ps_root( image[y,:])  # line = [Pm_, Pd_]
        if fline_PPs:
            from line_PPs import line_PPs_root
            line = line_PPs_root([line])  # line = CPp, sublayers[0] is a flat 16-tuple of P_s,
            # but it is decoded by indices as 3-layer nested tuple P_ttt: (Pm_, Pd_, each:( Lmd, Imd, Dmd, Mmd, each:( Ppm_, Ppd_)))
            if frecursive:
                from line_recursive import line_level_root
                types_ = []
                for i in range(16):  # len(line.sublayers[0]
                    types_.append([i % 2, int(i % 8 / 2), int(i / 8) % 2])  # 2nd level output types: fPpd, param, fPd
                line = line_level_root(line, types_)  # line = CPp, sublayers[0] is a tuple of P_s, with implicit nesting decoded by types_

        frame.append(line)  # if fline_PPs: line is root CPp, else [Pm_, Pd_]

    end_time = time() - start_time
    print(end_time)