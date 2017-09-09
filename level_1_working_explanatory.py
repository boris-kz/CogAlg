from scipy import misc

'''
Level 1:

Cross-comparison between consecutive pixels within horizontal scan line (row).
Resulting difference patterns dPs (spans of pixels forming same-sign differences)
and relative match patterns vPs (spans of pixels forming same-sign predictive value)
are redundant representations of each line of pixels.
'''

def incremental_range(ave_match, min_to_inc_range, min_to_inc_derivation, min_comp_range, Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der, comp_range, pixels_):

    if comp_range > min_comp_range:  # Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der inc.to adjust for redundancy to patterns formed by prior compare:
        Ave_match += ave_match     # ave_match: min m for inclusion into positive vP
        cum_min_to_inc_rng += min_to_inc_range   # min_to_inc_range: min V for initial compare() recursion, cum_min_to_inc_rng: min V for higher recursions

    if comp_range > min_comp_range-1:  # default range is shorter for d_[w]: redundant ds are smaller than ps
        cum_min_to_inc_der += min_to_inc_derivation     # aV: min |D| for compare() recursion over d_[w], cum_min_to_inc_der: min |D| for recursion

    current_pattern_width = len(pixels_)  
    ip_ = pixels_  # to differentiate from new pixels_

    vP_, dP_ = [],[]  # comp_range was incremented in higher-scope pixels_
    pri_s, I, D, V, rv, olp, pixels_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    for index in range(comp_range+1, current_pattern_width):

        new_pixel, fuzzy_difference, fuzzy_value = ip_[index]       # compared to a pixel at index-comp_range-1:
        previous_pixel, pfd, pfv = ip_[index-comp_range]  # previously compared p(ignored), its fuzzy_difference, fuzzy_value to next p
        fuzzy_value += pfv  # fuzzy v is summed over extended-compare range
        fuzzy_difference += pfd  # fuzzy d is summed over extended-compare range

        previous_pixel, prior_pixel_comp_result_fd , prior_pixel_comp_result_fv = ip_[index-comp_range-1]  # for compare(p, previous_pixel), pri_fd and prior_pixel_comp_result_fv ignored
        compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,  
                    # ''' BK: what does this repetition do? '''
                    # KK: Basically it is encapsulating and grouping the arguments. 
                    # BK: so you can change all of their instances at once?

                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_value':fuzzy_value,  
                    'width_index':index, 
                    'current_pattern_width':current_pattern_width,
                },
                'vP': {
                    'pri_s':pri_s, # BK: this is boolean, but treated as integer for simplicity
                    # all other vars are integers, except for arrays which end with _
                    'I':I, # BK: capitalized variables are small-case variables summed over pattern width
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'pixels_':pixels_, 
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
                    'ave_match': ave_match,
                    'min_to_inc_range': min_to_inc_range,
                    'min_to_inc_derivation': min_to_inc_derivation,
                    'min_comp_range': min_comp_range,
                     # BK: initialized *= number of inc_rng recursions if starts with Ave_match
                    'Ave_match': Ave_match, 
                    'cum_min_to_inc_rng': cum_min_to_inc_rng,
                    'cum_min_to_inc_der': cum_min_to_inc_der,
                    'comp_range': comp_range,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }


        pri_s, I, D, V, rv, pixels_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        compare(compare_inputs)

    return vP_, dP_  # local vPs and dPs to replace pixels_, Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der accumulated per compare recursion


def incremental_derivation(ave_match, min_to_inc_range, min_to_inc_derivation, min_comp_range, Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der, comp_range, d_):

    if comp_range > min_comp_range:
        Ave_match += ave_match; cum_min_to_inc_rng += min_to_inc_range
    if comp_range > min_comp_range-1:
        cum_min_to_inc_der += min_to_inc_derivation

    current_pattern_width = len(d_)
    ip_ = d_  # to differentiate from new d_

    fuzzy_difference, fuzzy_value, comp_range, vP_, dP_ = 0, 0, 0, [], []  # comp_range is initialized for each d_
    pri_s, I, D, V, rv, olp, pixels_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple vP=0,
    pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # tuple dP=0

    previous_pixel = ip_[0]

    for index in range(1, current_pattern_width):
        # BK: isn't 'i' self-explanatory?

        new_pixel = ip_[index]  # better than pop()?
        compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,
                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_value':fuzzy_value,
                    'width_index':index,
                    'current_pattern_width':current_pattern_width,
                },
                'vP': {
                    'pri_s':pri_s, 
                    'I':I, 
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'pixels_':pixels_, 
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
                    'ave_match': ave_match,
                    'min_to_inc_range': min_to_inc_range,
                    'min_to_inc_derivation': min_to_inc_derivation,
                    'min_comp_range': min_comp_range,
                    'Ave_match': Ave_match,
                    'cum_min_to_inc_rng': cum_min_to_inc_rng,
                    'cum_min_to_inc_der': cum_min_to_inc_der,
                    'comp_range': comp_range,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }

        pri_s, I, D, V, rv, pixels_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP, dP_ = \
        compare(compare_inputs)

        previous_pixel = new_pixel

    return vP_, dP_  # local vPs and dPs to replace d_


def compare(inputs):
    # input variables 
    new_pixel = inputs['input_variable']['new_pixel']
    previous_pixel = inputs['input_variable']['previous_pixel']
    fuzzy_difference = inputs['input_variable']['fuzzy_difference']
    fuzzy_value = inputs['input_variable']['fuzzy_value']
    width_index = inputs['input_variable']['width_index']
    current_pattern_width = inputs['input_variable']['current_pattern_width']

    # variables of vP
    pri_s = inputs['vP']['pri_s']
    I = inputs['vP']['I']
    D = inputs['vP']['D']
    V = inputs['vP']['V']
    rv = inputs['vP']['rv']
    pixels_ = inputs['vP']['pixels_']
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
    ave_match = inputs['filter_variable']['ave_match']
    min_to_inc_range = inputs['filter_variable']['min_to_inc_range']
    min_to_inc_derivation = inputs['filter_variable']['min_to_inc_derivation']
    min_comp_range = inputs['filter_variable']['min_comp_range']
    Ave_match = inputs['filter_variable']['Ave_match']
    cum_min_to_inc_rng = inputs['filter_variable']['cum_min_to_inc_rng']
    cum_min_to_inc_der = inputs['filter_variable']['cum_min_to_inc_der']
    comp_range = inputs['filter_variable']['comp_range']
    vP_ = inputs['filter_variable']['vP_']
    dP_ = inputs['filter_variable']['dP_']

    difference_pixel = new_pixel - previous_pixel      # difference between consecutive pixels
    match_pixel = min(new_pixel, previous_pixel)  # match between consecutive pixels
    relative_match = match_pixel - Ave_match          # relative match (predictive value) between consecutive pixels

    fuzzy_difference += difference_pixel  # fuzzy difference_pixel includes all shorter + current- range ds between comparands
    fuzzy_value += relative_match  # fuzzy relative_match includes all shorter + current- range vs between comparands

    # formation of value pattern vP: span of pixels forming same-sign relative_match s:

    s = 1 if relative_match > 0 else 0  # s: positive sign of relative_match
    if width_index > comp_range+2 and (s != pri_s or width_index == current_pattern_width-1):  # if derived pri_s miss, vP is terminated

        if len(pixels_) > comp_range+3 and pri_s == 1 and V > cum_min_to_inc_rng:  # min 3 compare over extended distance within pixels_:

            comp_range += 1  # comp_range: incremental range-of-compare counter
            rv = 1  # rv: incremental range flag:
            pixels_.append(incremental_range(ave_match, min_to_inc_range, min_to_inc_derivation, min_comp_range, Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der, comp_range, pixels_))

        p = I / len(pixels_); difference_pixel = D / len(pixels_); relative_match = V / len(pixels_)  # default to eval overlap, poss. div.compare?
        vP = pri_s, p, I, difference_pixel, D, relative_match, V, rv, pixels_, olp_
        vP_.append(vP)  # output of vP, related to dP_ by overlap only, no discont compare till Le3?

        o = len(vP_), olp  # len(P_) is index of current vP
        dolp_.append(o)  # indexes of overlapping vPs and olp are buffered at current dP

        I, D, V, rv, olp, dolp, pixels_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # initialization of new vP and olp

    pri_s = s   # vP (span of pixels forming same-sign relative_match) is incremented:
    olp += 1    # overlap to current dP
    I += previous_pixel  # ps summed within vP
    D += fuzzy_difference     # fuzzy ds summed within vP
    V += fuzzy_value     # fuzzy vs summed within vP
    # no, this is fuzzy predictive value: relative match = m - Ave_match
    
    pri = previous_pixel, fuzzy_difference, fuzzy_value  # inputs for recursive compare are tuples vs. pixels
    pixels_.append(pri)  # buffered within vP for selective extended compare


    # formation of difference pattern dP: span of pixels forming same-sign difference_pixel s:

    sd = 1 if difference_pixel > 0 else 0  # sd: positive sign of difference_pixel;
    if width_index > comp_range+2 and (sd != pri_sd or width_index == current_pattern_width-1):  # if derived pri_sd miss, dP is terminated

        if len(d_) > 3 and abs(Dd) > cum_min_to_inc_der:  # min 3 compare within d_:

            rd = 1  # rd: incremental derivation flag:
            d_.append(incremental_derivation(ave_match, min_to_inc_range, min_to_inc_derivation, min_comp_range, Ave_match, cum_min_to_inc_rng, cum_min_to_inc_der, comp_range, d_))

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
    Vd += fuzzy_value     # fuzzy vs summed within dP
    
    d_.append(fuzzy_difference)  # prior fds are buffered within dP, all of the same sign

    return pri_s, I, D, V, rv, pixels_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP_, dP_
    # for next p comparison, vP and dP increment, and output

def level_1(frame_pixels):

    output_frame_patterns = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    # this is a frame of vPs, not of pixels
    
    frame_height, frame_width = frame_pixels.shape  # Y: frame height, frame_width: frame width

    ave_match = 127  # minimal filter for vP inclusion
    min_to_inc_range = 63  # minimal filter for incremental-range compare
    min_to_inc_derivation = 63  # minimal filter for incremental-derivation compare
    min_comp_range = 0  # default range of fuzzy comparison, initially 0

    for height_index in range(frame_height):

        new_line = frame_pixels[height_index, :]  # index: new line ip_

        if min_comp_range == 0:
            Ave_match = ave_match
            cum_min_to_inc_rng = min_to_inc_range  # actual filters, incremented per compare recursion
        else:
            Ave_match = 0
            cum_min_to_inc_rng = 0  # if comp_range > min_comp_range

        if min_comp_range <= 1:
            cum_min_to_inc_der = min_to_inc_derivation
        else:
            cum_min_to_inc_der = 0

        fuzzy_difference, fuzzy_value, comp_range, height_index, vP_, dP_ = 0, 0, 0, 0, [], []  # i/o tuple
        pri_s, I, D, V, rv, olp, pixels_, olp_ = 0, 0, 0, 0, 0, 0, [], []  # vP tuple
        pri_sd, Id, Dd, Vd, rd, dolp, d_, dolp_ = 0, 0, 0, 0, 0, 0, [], []  # dP tuple

        previous_pixel = new_line[0]

        for width_index in range(1, frame_width):  # cross-compares consecutive pixels

            new_pixel = new_line[width_index]  # new pixel for compare to prior pixel, could use pop()?

            compare_inputs = {
                'input_variable': {
                    'new_pixel':new_pixel,
                    'previous_pixel':previous_pixel,
                    'fuzzy_difference':fuzzy_difference,
                    'fuzzy_value':fuzzy_value,
                    'width_index':width_index,
                    'current_pattern_width':frame_width,
                },
                'vP': {
                    'pri_s':pri_s, 
                    'I':I, 
                    'D':D, 
                    'V':V, 
                    'rv':rv, 
                    'pixels_':pixels_, 
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
                    'ave_match': ave_match,
                    'min_to_inc_range': min_to_inc_range,
                    'min_to_inc_derivation': min_to_inc_derivation,
                    'min_comp_range': min_comp_range,
                    'Ave_match': Ave_match,
                    'cum_min_to_inc_rng': cum_min_to_inc_rng,
                    'cum_min_to_inc_der': cum_min_to_inc_der,
                    'comp_range': comp_range,
                    'vP_': vP_,
                    'dP_': dP_,
                },
            }

            pri_s, I, D, V, rv, pixels_, olp, olp_, pri_sd, Id, Dd, Vd, rd, d_, dolp, dolp_, vP_, dP_ = \
            compare(compare_inputs)

            previous_pixel = new_pixel  # prior pixel, pri_ values are formed by prior run of compare

        LP_ = vP_, dP_
        output_frame_patterns.append(LP_)  # line of patterns is added to frame of patterns, height_index = len(output_frame_patterns)

    return output_frame_patterns  # output to level 2



# at vP term: print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'pixels_', pixels_)
# at dP term: print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

if __name__ == '__main__':
    frame_pixels = misc.face(gray=True)  # input frame of pixels
    frame_pixels = frame_pixels.astype(int)
    level_1(frame_pixels)
