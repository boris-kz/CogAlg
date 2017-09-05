from scipy import misc

'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.
'''

def incremental_range(a, aV, aD, min_range, A, AV, AD, r, p_):

    if r > min_range:  # A, AV, AD inc.to adjust for redundancy to patterns formed by prior compare:
        A += a     # a: min m for inclusion into positive vP
        AV += aV   # aV: min V for initial compare() recursion, AV: min V for higher recursions

    if r > min_range-1:  # default range is shorter for d_[w]: redundant ds are smaller than ps
        AD += aD     # aV: min |D| for compare() recursion over d_[w], AD: min |D| for recursion

    frame_width = len(p_)  
    # BK: this is recursing pattern width now, not a global frame width
    ip_ = p_  # to differentiate from new p_

    vP_, dP_ = [],[]  # r was incremented in higher-scope p_
    pri_s, I, D, V, rv, olp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    for x in range(r+1, frame_width):

        new_pixel, fuzzy_difference, fuzzy_variable = ip_[x]       # compared to a pixel at x-r-1:
        previous_pixel, pfd, pfv = ip_[x-r]  # previously compared p(ignored), its fuzzy_difference, fuzzy_variable to next p
        fuzzy_variable += pfv  # fuzzy v is summed over extended-compare range
        fuzzy_difference += pfd  # fuzzy d is summed over extended-compare range

        previous_pixel, pri_fd, pri_fv = ip_[x-r-1]  # for compare(p, previous_pixel), pri_fd and pri_fv ignored
        compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,  
                    '' BK: what does this repetition do? ''
                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_variable':fuzzy_variable,  
                    # no, this is fuzzy predictive predictive predictive predictive value: relative match = m - A
                    'x':x,
                    'frame_width':frame_width,
                },
                'vP': {
                    'pri_s':pri_s, # BK: this is boolean, but treated as integer for simplicity
                    # all other vars are integers, except for arrays which end with _
                    'I':I, # BK: capitalized variables are small-case variables summed over pattern width
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'p_':p_, 
                    'olp':olp, 
                    'olp_':olp_,
                },
                'dP': {
                    'pri_sd':pri_sd,
                    'Id':Id,
                    'Dd':Dd,
                    'Vd':Vd,
                    'rd':rd,
                    'd_':d_,
                    'dolp':dolp,
                    'dolp_':dolp_,
                },
                'filter_variable':{  
                    # BK: initialized | feedback value if starts with a
                    'a': a,
                    'aV': aV,
                    'aD': aD,
                    'min_range': min_range,
                     # BK: initialized *= number of inc_rng recursions if starts with A
                    'A': A, 
                    'AV': AV,
                    'AD': AD,
                    'r': r,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }


        pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        compare(compare_inputs)

    return vP_, dP_  # local vPs and dPs to replace p_, A, AV, AD accumulated per compare recursion


def incremental_derivation(a, aV, aD, min_range, A, AV, AD, r, d_):

    if r > min_range:
        A += a; AV += aV
    if r > min_range-1:
        AD += aD

    frame_width = len(d_)
    # BK: this is recursing pattern width, not a global frame width
    ip_ = d_  # to differentiate from new d_

    fuzzy_difference, fuzzy_variable, r, vP_, dP_ = 0, 0, 0, [], []  # r is initialized for each d_
    pri_s, I, D, V, rv, olp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0,
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    previous_pixel = ip_[0]

    for x in range(1, frame_width):

        new_pixel = ip_[x]  # better than pop()?
        compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,
                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_variable':fuzzy_variable,
                    # no, this is fuzzy predictive value: relative match = m - A
                    'x':x,
                    'frame_width':frame_width,
                },
                'vP': {
                    'pri_s':pri_s, 
                    'I':I, 
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'p_':p_, 
                    'olp':olp, 
                    'olp_':olp_,
                },
                'dP': {
                    'pri_sd':pri_sd,
                    'Id':Id,
                    'Dd':Dd,
                    'Vd':Vd,
                    'rd':rd,
                    'd_':d_,
                    'dolp':dolp,
                    'dolp_':dolp_,
                },
                'filter_variable':{
                    'a': a,
                    'aV': aV,
                    'aD': aD,
                    'min_range': min_range,
                    'A': A,
                    'AV': AV,
                    'AD': AD,
                    'r': r,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }

        pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        compare(compare_inputs)

        previous_pixel = new_pixel

    return vP_, dP_  # local vPs and dPs to replace d_


def compare(inputs):
    # input variables 
    '' BK: what does this repetition do? ''
    new_pixel = inputs['input_variable']['new_pixel']
    previous_pixel = inputs['input_variable']['previous_pixel']
    fuzzy_difference = inputs['input_variable']['fuzzy_difference']
    fuzzy_variable = inputs['input_variable']['fuzzy_variable']
    # no, this is fuzzy value: relative match = m - A
    x = inputs['input_variable']['x']
    frame_width = inputs['input_variable']['frame_width']

    # variables of vP
    pri_s = inputs['vP']['pri_s']
    I = inputs['vP']['I']
    D = inputs['vP']['D']
    V = inputs['vP']['V']
    rv = inputs['vP']['rv']
    p_ = inputs['vP']['p_']
    olp = inputs['vP']['olp']
    olp_ = inputs['vP']['olp_']

    # variables of dP
    pri_sd = inputs['dP']['pri_sd']
    Id = inputs['dP']['Id']
    Dd = inputs['dP']['Dd']
    Vd = inputs['dP']['Vd']
    rd = inputs['dP']['rd']
    d_ = inputs['dP']['d_']
    dolp = inputs['dP']['dolp']
    dolp_ = inputs['dP']['dolp_']

    # filter variables and output patterns
    a = inputs['filter_variable']['a']
    aV = inputs['filter_variable']['aV']
    aD = inputs['filter_variable']['aD']
    min_range = inputs['filter_variable']['min_range']
    A = inputs['filter_variable']['A']
    AV = inputs['filter_variable']['AV']
    AD = inputs['filter_variable']['AD']
    r = inputs['filter_variable']['r']
    vP_ = inputs['filter_variable']['vP_']
    dP_ = inputs['filter_variable']['dP_']

    difference_pixel = new_pixel - previous_pixel      # difference between consecutive pixels
    match_pixel = min(new_pixel, previous_pixel)  # match between consecutive pixels
    relative_match = match_pixel - A          # relative match (predictive value) between consecutive pixels

    fuzzy_difference += difference_pixel  # fuzzy difference_pixel includes all shorter + current- range ds between comparands
    fuzzy_variable += relative_match  # fuzzy relative_match includes all shorter + current- range vs between comparands
    # no, this is fuzzy predictive value = relative match

    # formation of value pattern vP: span of pixels forming same-sign relative_match s:

    s = 1 if relative_match > 0 else 0  # s: positive sign of relative_match
    if x > r+2 and (s != pri_s or x == frame_width-1):  # if derived pri_s miss, vP is terminated

        if len(p_) > r+3 and pri_s == 1 and V > AV:  # min 3 compare over extended distance within p_:

            r += 1  # r: incremental range-of-compare counter
            rv = 1  # rv: incremental range flag:
            p_.append(incremental_range(a, aV, aD, min_range, A, AV, AD, r, p_))

        p = I / len(p_); difference_pixel = D / len(p_); relative_match = V / len(p_)  # default to eval overlap, poss. div.compare?
        vP = pri_s, p, I, difference_pixel, D, relative_match, V, rv, p_, olp_
        vP_.append(vP)  # output of vP, related to dP_ by overlap only, no discont compare till Le3?

        o = len(vP_), olp  # len(P_) is index of current vP
        dolp_.append(o)  # indexes of overlapping vPs and olp are buffered at current dP

        I, D, V, rv, olp, dolp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # initialization of new vP and olp

    pri_s = s   # vP (span of pixels forming same-sign relative_match) is incremented:
    olp += 1    # overlap to current dP
    I += previous_pixel  # ps summed within vP
    D += fuzzy_difference     # fuzzy ds summed within vP
    V += fuzzy_variable     # fuzzy vs summed within vP
    # no, this is fuzzy predictive value: relative match = m - A
    
    pri = previous_pixel, fuzzy_difference, fuzzy_variable  # inputs for recursive compare are tuples vs. pixels
    p_.append(pri)  # buffered within vP for selective extended compare


    # formation of difference pattern dP: span of pixels forming same-sign difference_pixel s:

    sd = 1 if difference_pixel > 0 else 0  # sd: positive sign of difference_pixel;
    if x > r+2 and (sd != pri_sd or x == frame_width-1):  # if derived pri_sd miss, dP is terminated

        if len(d_) > 3 and abs(Dd) > AD:  # min 3 compare within d_:

            rd = 1  # rd: incremental derivation flag:
            d_.append(incremental_derivation(a, aV, aD, min_range, A, AV, AD, r, d_))

        pd = Id / len(d_); dd = Dd / len(d_); vd = Vd / len(d_)  # so all olp Ps can be directly evaluated
        dP = pri_sd, pd, Id, dd, Dd, vd, Vd, rd, d_, dolp_
        dP_.append(dP)  # output of dP

        o = len(dP_), dolp  # len(P_) is index of current dP
        olp_.append(o)  # indexes of overlapping dPs and dolps are buffered at current vP

        Id, Dd, Vd, rd, olp, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # initialization of new dP and olp

    pri_sd = sd  # dP (span of pixels forming same-sign difference_pixel) is incremented:
    dolp += 1    # overlap to current vP
    Id += previous_pixel  # ps summed within dP
    Dd += fuzzy_difference     # fuzzy ds summed within dP
    Vd += fuzzy_variable     # fuzzy vs summed within dP
    # no, this is fuzzy predictive value: relative match = m - A
    
    d_.append(fuzzy_difference)  # prior fds are buffered within dP, all of the same sign

    return pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP_, dP_
    # for next p comparison, vP and dP increment, and output

def level_1(frame_pixels):

    output_frame_pixels = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    # this is a frame of vPs, not of pixels
    
    frame_height, frame_width = frame_pixels.shape  # Y: frame height, frame_width: frame width

    a = 127  # minimal filter for vP inclusion
    aV = 63  # minimal filter for incremental-range compare
    aD = 63  # minimal filter for incremental-derivation compare
    min_range = 0  # default range of fuzzy comparison, initially 0

    for y in range(frame_height):

        new_line = frame_pixels[y, :]  # y is index of new line ip_

        if min_range == 0:
            A = a
            AV = aV  # actual filters, incremented per compare recursion
        else:
            A = 0
            AV = 0  # if r > min_range

        if min_range <= 1:
            AD = aD
        else:
            AD = 0

        fuzzy_difference, fuzzy_value, r, x, vP_, dP_ = 0, 0, 0, 0, [], []  # i/o tuple
        pri_s, I, D, V, rv, olp, p_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # vP tuple
        pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # dP tuple

        previous_pixel = new_line[0]

        for x in range(1, frame_width):  # cross-compares consecutive pixels

            new_pixel = new_line[x]  # new pixel for compare to prior pixel, could use pop()?

            compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,
                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_variable':fuzzy_value,
                    'x':x,
                    'frame_width':frame_width,
                },
                'vP': {
                    'pri_s':pri_s, 
                    'I':I, 
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'p_':p_, 
                    'olp':olp, 
                    'olp_':olp_,
                },
                'dP': {
                    'pri_sd':pri_sd,
                    'Id':Id,
                    'Dd':Dd,
                    'Vd':Vd,
                    'rd':rd,
                    'd_':d_,
                    'dolp':dolp,
                    'dolp_':dolp_,
                },
                'filter_variable':{
                    'a': a,
                    'aV': aV,
                    'aD': aD,
                    'min_range': min_range,
                    'A': A,
                    'AV': AV,
                    'AD': AD,
                    'r': r,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }

            pri_s, I, D, V, rv, p_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP_, dP_ = \
            compare(compare_inputs)

            previous_pixel = new_pixel  # prior pixel, pri_ values are always derived before use

        LP_ = vP_, dP_
        output_frame_patterns.append(LP_)  # line of patterns is added to frame of patterns, y = len(output_frame_pixels)

    return output_frame_patterns  # output to level 2


# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

if __name__ == '__main__':
    frame_pixels = misc.face(gray=True)  # input frame of pixels
    frame_pixels = frame_pixels.astype(int)
    level_1(frame_pixels)
