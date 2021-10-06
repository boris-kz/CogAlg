'''
  line_patterns is a principal version of 1st-level 1D algorithm
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
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
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
    mrdn = int  # -> Rdn: rdn counter per P

class CP(ClusterStructure):
    L = int
    I = int
    D = int
    M = int
    Rdn = int  # mrdn counter
    x0 = int
    dert_ = list  # contains (i, p, d, m)
    # for layer-parallel access and comp, ~ frequency domain, composition: 1st: dert_, 2nd: sub_P_[ dert_], 3rd: sublayers[ sub_P_[ dert_]]:
    sublayers = list  # multiple layers of sub_P_s from d segmentation or extended comp, nested to depth = sub_[n]
    subDerts = list  # conditionally summed [L,I,D,M] per sublayer, most are empty
    derDerts = list  # for compared subDerts only
    fPd = bool  # P is Pd if true, else Pm; also defined per layer

verbose = False
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = .5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_splice = 50  # to merge a kernel of 3 adjacent Ps
init_y = 0  # starting row, the whole frame doesn't need to be processed
halt_y = 999999999  # ending row
'''
    Conventions:
    postfix 't' denotes tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    capitalized variables are normally summed small-case variables
'''

def cross_comp(frame_of_pixels_):  # converts frame_of_pixels to frame_of_patterns, each pattern may be nested

    Y, X = frame_of_pixels_.shape  # Y: frame height, X: frame width
    frame_of_patterns_ = []
    '''
    if cross_comp_spliced:  # process all image rows as a single line, vertically consecutive and preserving horizontal direction:
        pixel_=[]; dert_=[]  
        for y in range(init_y, Y):  
            pixel_.append([ frame_of_pixels_[y, :]])  # splice all rows into pixel_
        _i = pixel_[0]
    else:
    '''
    for y in range(init_y, min(halt_y, Y)):  # y is index of new row pixel_, we only need one row, use init_y=0, halt_y=Y for full frame

        # initialization:
        dert_ = []  # line-wide i_, p_, d_, m_, mrdn_
        pixel_ = frame_of_pixels_[y, :]
        _i = pixel_[0]
        # pixel i is compared to prior pixel _i in a row:
        for i in pixel_[1:]:
            d = i - _i  # accum in rng
            p = i + _i  # accum in rng
            m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
            mrdn = m + ave < abs(d)
            dert_.append( Cdert( i=i, p=p, d=d, m=m, mrdn=mrdn) )
            _i = i
        # form patterns:
        rval_Pm_ = form_P_(None, dert_, rdn=1, rng=1, fPd=False)  # rootP=None, eval intra_P_, call form_rval_P_
        rval_Pd_ = form_P_(None, dert_, rdn=1, rng=1, fPd=True)

        frame_of_patterns_.append((rval_Pm_, rval_Pd_))  # add line of patterns to frame of patterns, skip if cross_comp_spliced

    return frame_of_patterns_  # frame of patterns, an input to level 2


def form_P_(rootP, dert_, rdn, rng, fPd):  # accumulation and termination, rdn and rng are pass-through intra_P_
    # initialization:
    P_ = []
    x = 0
    _sign = None  # to initialize 1st P, (None != True) and (None != False) are both True

    for dert in dert_:  # segment by sign
        if fPd: sign = dert.d > 0
        else:   sign = dert.m > 0
        if sign != _sign:
            # sign change, initialize and append P
            P = CP( L=1, I=dert.p, D=dert.d, M=dert.m, Rdn=dert.mrdn, x0=x, dert_=[dert], sublayers=[], fPd=fPd)
            P_.append(P)  # updated with accumulation below
        else:
            # accumulate params:
            P.L += 1; P.I += dert.p; P.D += dert.d; P.M += dert.m; P.Rdn += dert.mrdn
            P.dert_ += [dert]
        x += 1
        _sign = sign

    if rootP:  # call from intra_P_
        if len(P_) > 4:  # 2 * (rng+1) = 2*2 =4
            rootP.sublayers += intra_P_(P_, rdn, rng, fPd)  # deeper comb_layers feedback, subDerts is selective
    else:  # call from cross_comp
        intra_P_(P_, rdn, rng, fPd)

    rval_P_ = form_rval_P_(P_, fPd)
    return rval_P_  # used only if not rootP, else packed in rootP.sublayers and rootP.subDerts


def intra_P_(P_, rdn, rng, fPd):  # recursive cross-comp and form_P_ inside selected sub_Ps in P_

    adj_M_ = form_adjacent_M_(P_)  # compute adjacent Ms to evaluate contrastive borrow potential
    comb_sublayers = []

    for P, adj_M in zip(P_, adj_M_):
        if P.L > 2 * (rng + 1):  # vs. **? rng+1 because rng is initialized at 0, as all params
            rel_adj_M = adj_M / -P.M  # for allocation of -Pm' adj_M to each of its internal Pds?

            if fPd:  # P is Pd
                if min(abs(P.D), abs(P.D) * rel_adj_M) > ave_D * rdn:  # high-D span, level rdn, vs. param rdn in dert
                    sub_Pm_, sub_Pd_ = [],[]
                    rdn+=1; rng+=1  # brackets: 1st: param set, 2nd: sublayer concatenated from several root_Ps, 3rd: hierarchy of sublayers:
                    P.sublayers += [[(fPd, rdn, rng, sub_Pm_, [], sub_Pd_, [])]]
                    ddert_ = deriv_comp(P.dert_)  # i is d
                    sub_Pm_[:] = form_P_(P, ddert_, rdn, rng, fPd=False)  # cluster by mm sign
                    sub_Pd_[:] = form_P_(P, ddert_, rdn, rng, fPd=True)  # cluster by md sign
            else:  # P is Pm
                   # eval comp at rng=2^n: 1, 2, 3; kernel size 2, 4, 8..:
                if P.M > ave_M * rdn:  # high-M span, no -adj_M: lend to contrast is not adj only, reflected in ave?
                    ''' if local ave:
                    loc_ave = (ave + (P.M - adj_M) / P.L) / 2  # mean ave + P_ave, possibly negative?
                    loc_ave_min = (ave_min + (P.M - adj_M) / P.L) / 2  # if P.M is min?
                    rdert_ = range_comp(P.dert_, loc_ave, loc_ave_min, fid)
                    '''
                    sub_Pm_, sub_Pd_ = [],[]
                    rdn+=1; rng+=1  # brackets: 1st: param set, 2nd: sublayer concatenated from several root_Ps, 3rd: hierarchy of sublayers:
                    P.sublayers += [[(fPd, rdn, rng, sub_Pm_, [], sub_Pd_, [])]]
                    rdert_ = range_comp(P.dert_)  # rng+, skip predictable next dert, local ave? rdn to higher (or stronger?) layers
                    sub_Pm_[:] = form_P_(P, rdert_, rdn, rng, fPd=False)  # cluster by rm sign
                    sub_Pd_[:] = form_P_(P, rdert_, rdn, rng, fPd=True)  # cluster by rd sign

                else:  # to preserve index of sub_Pms and sub_Pds in P.sublayers
                    P.sublayers += [[]]

            # index 0 is sub_Pms, index 1 is sub_Pds, due to we form_P_ with fPd = false first
            if P.sublayers and P.sublayers[fPd]:
                new_sublayers = []
                for comb_subset_, subset_ in zip_longest(comb_sublayers, P.sublayers[fPd], fillvalue=([])):
                    # append combined subset_ (array of sub_P_ param sets):
                    new_sublayers.append(comb_subset_ + subset_)
                comb_sublayers = new_sublayers

    return comb_sublayers


def form_adjacent_M_(Pm_):  # compute array of adjacent Ms, for contrastive borrow evaluation
    '''
    Value is projected match, while variation has contrast value only: it matters to the extent that it interrupts adjacent match: adj_M.
    In noise, there is a lot of variation. but no adjacent match to cancel, so that variation has no predictive value.
    On the other hand, 2D outline or 1D contrast may have low gradient / difference, but it terminates some high-match span.
    Such contrast is salient to the extent that it can borrow predictive value from adjacent high-match area.
    adj_M is not affected by primary range_comp per Pm?
    no comb_m = comb_M / comb_S, if fid: comb_m -= comb_|D| / comb_S: alt rep cost
    same-sign comp: parallel edges, cross-sign comp: M - (~M/2 * rL) -> contrast as 1D difference?
    '''
    M_ = [0] + [Pm.M for Pm in Pm_] + [0]  # list of adj M components in the order of Pm_, + first and last M=0,

    adj_M_ = [ (abs(prev_M) + abs(next_M)) / 2  # mean adjacent Ms
               for prev_M, next_M in zip(M_[:-2], M_[2:])  # exclude first and last Ms
             ]
    ''' expanded:
    pri_M = Pm_[0].M  # deriv_comp value is borrowed from adjacent opposite-sign Ms
    M = Pm_[1].M
    adj_M_ = [abs(Pm_[1].M)]  # initial next_M, also projected as prior for first P
    for Pm in Pm_[2:]:
        next_M = Pm.M
        adj_M_.append((abs(pri_M / 2) + abs(next_M / 2)))  # exclude M
        pri_M = M
        M = next_M
    adj_M_.append(abs(pri_M))  # no / 2: projection for last P
    '''
    return adj_M_

def range_comp(dert_):  # cross-comp of 2**rng- distant pixels: 4,8,16.., skipping intermediate pixels
    rdert_ = []
    _i = dert_[0].i

    for dert in dert_[2::2]:  # all inputs are sparse, skip odd pixels compared in prior rng: 1 skip / 1 add to maintain 2x overlap
        d = dert.i -_i
        rp = dert.p + _i  # intensity accumulated in rng
        rd = dert.d + d   # difference accumulated in rng
        rm = dert.m + ave - abs(d)  # m accumulated in rng
        # for consistency with deriv_comp, else redundant
        rmrdn = rm + ave < abs(rd)  # use Ave?
        rdert_.append( Cdert( i=dert.i,p=rp,d=rd,m=rm,mrdn=rmrdn ))
        _i = dert.i

    return rdert_

def deriv_comp(dert_):  # cross-comp consecutive ds in same-sign dert_: sign match is partial d match
    # dd and md may match across d sign, spliced in compact?
    # initialization:
    ddert_ = []
    _d = abs( dert_[0].d)  # same-sign in Pd

    for dert in dert_[1:]:
        # same-sign in Pd
        d = abs( dert.d )
        rd = d + _d
        dd = d - _d
        md = min(d, _d) - abs( dd/2) - ave_min  # min_match because magnitude of derived vars corresponds to predictive value
        dmrdn = md + ave < abs(dd)  # use Ave?
        ddert_.append( Cdert( i=dert.d,p=rd,d=dd,m=md,dmrdn=dmrdn ))
        _d = d

    return ddert_


def form_rval_P_(iP_, fPd):  # cluster Ps by the sign of value adjusted for cross-param redundancy,
    Rval = 0
    rval_P__, rval_P_ = [], []
    _sign = None  # to initialize 1st rdn P, (None != True) and (None != False) are both True

    for P in iP_:
        # P.L-P.Rdn is inverted P.Rdn: max P.Rdn = P.L. Inverted because it counts mrdn, = (not drdn):
        if fPd: rval = abs(P.D) - (P.L-P.Rdn) * ave_D * P.L
        else:   rval = P.M - P.Rdn * ave_M * P.L
        # ave_D, ave_M are defined per dert: variable cost to adjust for rdn,
        # * Ave_D, Ave_M coef: fixed costs per P?
        sign = rval>0

        if sign != _sign:  # sign change, initialize rP and append it to rP_
            rval_P__.append([Rval, rval_P_])  # updated by accumulation below
        else:
            # accumulate params:
            Rval += rval
            rval_P_ += [(rval, P)]
        _sign = sign

    return rval_P__


if __name__ == "__main__":
    ''' 
    Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='.//raccoon.jpg')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    '''
    fpickle = 2  # 0: load; 1: dump; 2: no pickling
    render = 0
    fline_PPs = 0
    start_time = time()
    if fpickle == 0:
        # Read frame_of_patterns from saved file instead
        with open("frame_of_patterns_.pkl", 'rb') as file:
            frame_of_patterns_ = pickle.load(file)
    else:
        # Run functions
        image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        assert image is not None, "No image in the path"
        # Main
        frame_of_patterns_ = cross_comp(image)  # returns Pm__
        if fpickle == 1: # save the dump of the whole data_1D to file
            with open("frame_of_patterns_.pkl", 'wb') as file:
                pickle.dump(frame_of_patterns_, file)

    if render:
        image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
        plt.figure(); plt.imshow(image, cmap='gray'); plt.show()  # show the image below in gray

    if fline_PPs:  # debug line_PPs
        from line_PPs import *
        frame_Pp_t = []

        for y, P_t in enumerate(frame_of_patterns_):  # each line_of_patterns is (Pm_, Pd_)
            if len(P_) > 1: rval_Pp_t, Pp_t = line_PPs_root(P_t, 0)
            else:           rval_Pp_t, Pp_t = [], []
            frame_Pp_t.append(( rval_Pp_t, Pp_t ))

        draw_PP_(image, frame_Pp_t)  # debugging

    end_time = time() - start_time
    print(end_time)