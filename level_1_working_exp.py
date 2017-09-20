from scipy import misc
import numpy as np
'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.
'''

def inc_rng(a, aV, aD, min_r, A, AV, AD, r, p_):

    if r > min_r:  # A, AV, AD inc.to adjust for redundancy to patterns formed by prior comp:
        A += a     # a: min m for inclusion into positive vP
        AV += aV   # aV: min V for initial comp() recursion, AV: min V for higher recursions

    if r > min_r-1:  # default range is shorter for d_[w]: redundant ds are smaller than ps
        AD += aD     # aV: min |D| for comp() recursion over d_[w], AD: min |D| for recursion

    X = len(p_)
    ip_ = p_  # to differentiate from new p_

    vP_, dP_ = [],[]  # r was incremented in higher-scope p_
    pri_s, I, D, V, rv, olp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    for x in range(r+1, X):

        p, fd, fv = ip_[x]       # compared to a pixel at x-r-1:
        pp, pfd, pfv = ip_[x-r]  # previously compared p(ignored), its fd, fv to next p
        fv += pfv  # fuzzy v is summed over extended-comp range
        fd += pfd  # fuzzy d is summed over extended-comp range

        pri_p, pri_fd, pri_fv = ip_[x-r-1]  # for comp(p, pri_p), pri_fd and pri_fv ignored

        pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        comp(p, pri_p, fd, fv, x, X,
             pri_s, I, D, V, rv, p_, olp, olp_,
             pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_,
             a, aV, aD, min_r, A, AV, AD, r, vP_, dP_)

    return vP_, dP_  # local vPs and dPs to replace p_, A, AV, AD accumulated per comp recursion


def inc_der(a, aV, aD, min_r, A, AV, AD, r, d_):

    if r > min_r:
        A += a; AV += aV
    if r > min_r-1:
        AD += aD

    X = len(d_)
    ip_ = d_  # to differentiate from new d_

    fd, fv, r, vP_, dP_ = 0, 0, 0, [], []  # r is initialized for each d_
    pri_s, I, D, V, rv, olp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0,
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    pri_p = ip_[0]

    for x in range(1, X):

        p = ip_[x]  # better than pop()?

        pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        comp(p, pri_p, fd, fv, x, X,
             pri_s, I, D, V, rv, p_, olp, olp_,
             pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_,
             a, aV, aD, min_r, A, AV, AD, r, vP_, dP_)

        pri_p = p

    return vP_, dP_  # local vPs and dPs to replace d_


def comp(p, pri_p, fd, fv, x, X,  # input variables
         pri_s, I, D, V, rv, p_, olp, olp_,  # variables of vP
         pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_,  # variables of dP
         a, aV, aD, min_r, A, AV, AD, r, vP_, dP_):  # filter variables and output patterns

    d = p - pri_p      # difference between consecutive pixels
    m = min(p, pri_p)  # match between consecutive pixels
    v = m - A          # relative match (predictive value) between consecutive pixels

    fd += d  # fuzzy d includes all shorter + current- range ds between comparands
    fv += v  # fuzzy v includes all shorter + current- range vs between comparands


    # formation of value pattern vP: span of pixels forming same-sign v s:

    s = 1 if v > 0 else 0  # s: positive sign of v
    if x > r+2 and (s != pri_s or x == X-1):  # if derived pri_s miss, vP is terminated

        if len(p_) > r+3 and pri_s == 1 and V > AV:  # min 3 comp over extended distance within p_:

            r += 1  # r: incremental range-of-comp counter
            rv = 1  # rv: incremental range flag:
            p_.append(inc_rng(a, aV, aD, min_r, A, AV, AD, r, p_))

        p = I / len(p_); d = D / len(p_); v = V / len(p_)  # default to eval overlap, poss. div.comp?
        vP = pri_s, p, I, d, D, v, V, rv, p_, olp_
        vP_.append(vP)  # output of vP, related to dP_ by overlap only, no discont comp till Le3?

        o = len(vP_), olp  # len(P_) is index of current vP
        dolp_.append(o)  # indexes of overlapping vPs and olp are buffered at current dP

        I, D, V, rv, olp, dolp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # initialization of new vP and olp_

    pri_s = s   # vP (span of pixels forming same-sign v) is incremented:
    olp += 1    # overlap to concurrent dP
    I += pri_p  # ps summed within vP
    D += fd     # fuzzy ds summed within vP
    V += fv     # fuzzy vs summed within vP
    pri = pri_p, fd, fv  # inputs for recursive comp are tuples vs. pixels
    p_.append(pri)  # buffered within vP for selective extended comp


    # formation of difference pattern dP: span of pixels forming same-sign d s:

    sd = 1 if d > 0 else 0  # sd: positive sign of d;
    if x > r+2 and (sd != pri_sd or x == X-1):  # if derived pri_sd miss, dP is terminated

        if len(d_) > 3 and abs(Dd) > AD:  # min 3 comp within d_:

            rd = 1  # rd: incremental derivation flag:
            d_.append(inc_der(a, aV, aD, min_r, A, AV, AD, r, d_))

        pd = Id / len(d_); dd = Dd / len(d_); vd = Vd / len(d_)  # so all olp Ps can be directly evaluated
        dP = pri_sd, pd, Id, dd, Dd, vd, Vd, rd, d_, dolp_
        dP_.append(dP)  # output of dP

        o = len(dP_), dolp  # len(P_) is index of current dP
        olp_.append(o)  # indexes of overlapping dPs and dolps are buffered at current vP

        Id, Dd, Vd, rd, olp, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # initialization of new dP and dolp_

    pri_sd = sd  # dP (span of pixels forming same-sign d) is incremented:
    dolp += 1    # overlap to concurrent vP
    Id += pri_p  # ps summed within dP
    Dd += fd     # fuzzy ds summed within dP
    Vd += fv     # fuzzy vs summed within dP
    d_.append(fd)  # prior fds of the same sign are buffered within dP

    return pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP_, dP_
    # for next p comparison, vP and dP increment, and output


def level_1(input_frame_pixels): # last '_' distinguishes array name from element name

    output_frame = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    frame_height, frame_width = input_frame_pixels.shape  # Y: frame height, X: frame width

    ave_match = 127  # minimal filter for vP inclusion
    min_to_inc_range = 63  # minimal filter for incremental-range comp
    min_to_inc_derivation = 63  # minimal filter for incremental-derivation comp
    min_comp_range=0  # default range of fuzzy comparison, initially 0

    for frame_height_index in range(frame_height):

        current_pixels_row = input_frame_pixels[frame_height_index, :]  # frame_height_index is index of new line current_pixels_row

        if min_comp_range == 0: cum_min_match = ave_match; cum_min_to_inc_rng = min_to_inc_range  # actual filters, incremented per comp recursion
        else: cum_min_match = 0; cum_min_to_inc_rng = 0  # if comp_range > min_comp_range

        if min_comp_range <= 1: cum_min_to_inc_der = min_to_inc_derivation
        else: cum_min_to_inc_der = 0

        fuzzy_difference, fuzzy_value, comp_range, relative_match_patterns, difference_patterns = 0, 0, 0, [], []  # i/o tuple
        previous_relative_match_pattern_span, I, fuzzy_diff_span_sum_over_value_pattern, fuzzy_value_span_sum_over_value_pattern, value_pattern_comp_range, current_overlapping_value_pattern, pixels_, overlapping_value_pattern = 0, 0, 0, 0, 0, 0, [], []  # vP tuple
        previous_difference_pattern_span, Id, fuzzy_diff_span_sum_over_difference_pattern, fuzzy_value_span_sum_over_difference_pattern, difference_pattern_comp_range, current_overlapping_difference_pattern, d_, overlapping_difference_pattern = 0, 0, 0, 0, 0, 0, [], []  # dP tuple

        previous_pixel = current_pixels_row[0]

        for frame_width_index in range(1, frame_width):  # cross-compares consecutive pixels

            current_pixel = current_pixels_row[frame_width_index]  # new pixel for comp to prior pixel, could use pop()?

            previous_relative_match_pattern_span, I, fuzzy_diff_span_sum_over_value_pattern, fuzzy_value_span_sum_over_value_pattern, value_pattern_comp_range, pixels_, current_overlapping_value_pattern, overlapping_value_pattern, previous_difference_pattern_span, Id, fuzzy_diff_span_sum_over_difference_pattern, fuzzy_value_span_sum_over_difference_pattern, difference_pattern_comp_range, d_, current_overlapping_difference_pattern, overlapping_difference_pattern, vP_, dP_ = \
            comp(current_pixel, previous_pixel, fuzzy_difference, fuzzy_value, frame_width_index, frame_width,
                 previous_relative_match_pattern_span, I, fuzzy_diff_span_sum_over_value_pattern, fuzzy_value_span_sum_over_value_pattern, value_pattern_comp_range, pixels_, current_overlapping_value_pattern, overlapping_value_pattern,
                 previous_difference_pattern_span, Id, fuzzy_diff_span_sum_over_difference_pattern, fuzzy_value_span_sum_over_difference_pattern, difference_pattern_comp_range, d_, current_overlapping_difference_pattern, overlapping_difference_pattern,
                 ave_match, min_to_inc_range, min_to_inc_derivation, min_comp_range, cum_min_match, cum_min_to_inc_rng, cum_min_to_inc_der, comp_range, relative_match_patterns, difference_patterns)

            previous_pixel = current_pixel  # prior pixel, pri_ values are always derived before use

        current_pattern_line = relative_match_patterns, difference_patterns
        output_frame.append(current_pattern_line)  # line of patterns is added to frame of patterns, y = len(output_frame)
    return output_frame  # output to level 2

if __name__ == '__main__':
    input_frame_pixels = misc.face(gray=True)  # input frame of pixels
    input_frame_pixels = input_frame_pixels.astype(int)
    level_1(input_frame_pixels)

# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)
