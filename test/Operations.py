def Axes_Rearrange(input_, axes=(0, 1)):

    ''' Sort inputs based on parameters chosen as coordinates

        - input_: list of inputs
        - axes: indices of parameters chosen as coordinates
        Return: a list sorted
            + First element is number of comps (not every input is compared twice)
            + Remaining elements: difference value for every axis

        For example:

            >> input_ = [(1, 0, 18), (0, 1, 17), (0, 0, 15), (1, 1, 20)]

            >> sorted_input_ = Axes_Rearrange(input_, axes=(0, 1))

            >> print(sorted_input_)

            [(0, 0, 15), (0, 1, 17), (1, 0, 18), (1, 1, 20)]

    '''

    output_ = list(input_)

    for axis in reversed(axes):
        output_.sort(key=lambda ip: ip[axis])    # sort based on each axis' coordinates, least significant axis first

    return output_

def Go_to(input_, target, from_idx=0, axes=(0, 1)):

    ''' Given a start index (is 0 by default), keep increasing it until it's >= target coordinate.
        Input list must be sorted by coordinates first '''

    i = from_idx
    t = False                               # signal variable, to check matching of higher order axes' coordinates
    for iaxis, axis in enumerate(axes):     # iterate through the axes
        sub_axes = axes[:iaxis]             # list of higher order axes

        while i < len(input_):
            # increase i by 1 if current axis coordinates < target coordinate and all previous coordinates are equal to target coordinates:
            t = not sum([input_[i][saxis] != target[isaxis] for isaxis, saxis in enumerate(sub_axes)])  # check matching of higher order axes' coordinates

            if not (t and input_[i][axis] < target[iaxis]):     # also look for matching of current axis coordinate
                break

            i += 1

        if i < len(input_):
            t = t and input_[i][axis] == target[iaxis] and i < len(input_)

            if not t:       # if there's a coordinate unmatch
                break
                
    return i, t    # return True if i points at the target coordinate
    
def Compare(input_, offset, comparand=2, axes=(0, 1)):

    ''' Compare imputs over indicated offset. Return differences between compared inputs.

        - input_: list of inputs
        - offset: coordinate displacement between compared inputs.
        - comparand: index of comparand in each input
        - axes: indices of axes' coordinates of each input
        Return: a list where each element is a list:
            + First element is number of comps (not every input is compared twice)
            + Remaining elements: difference value for every axis

        For example:

            >> input_ = [(0, 0, 15), (0, 1, 18), (1, 0, 19), (1, 1, 20)]

            >> output_ = Compare(input_, offset=(1, 0)) # Pure vertical comp

            >> print(output_)

            [[1, 4, 0], [1, 2, 0], [1, 4, 0], [1, 2, 0]]

    '''

    squared_mag = 0                         # squared magnitude of offset
    for d in offset:                        # offset between comparands
        squared_mag += d ** 2

    mag = squared_mag ** 0.5                # magnitude of offset
    coef_ = [d / mag for d in offset]       # decomposition coefficients

    # initialize output:
    output_ = [[0] * (len(axes) + 1)] * len(input_)   # for every outputs: results for each axis + number of comps

    # initialize indices:
    i = 0   # index of first comparand
    j = 0   # index of second comparand

    while i < len(input_) and j < len(input_):  # this loop is per i. j is updated accordingly. If either i or j is out of bound, no more comp could take place

        coords = [input_[i][axis] + offset[iaxis] for iaxis, axis in enumerate(axes)]   # coordinates of second comparand

        j, t = Go_to(input_, target=coords, from_idx=j, axes=axes)

        if j < len(input_) and t:  # compare input_[i] and input_[j] if input_[j] is in the target coordinate

            temp = input_[j][comparand] - input_[i][comparand]  # compare by subtraction
            d = [int(temp * coef) for coef in coef_]  # decomposition into value for each axis

            output_[i] = [val1 + val2 for val1, val2 in zip([1] + d, output_[i])]  # bilateral accumulation. First element is number of comps
            output_[j] = [val1 + val2 for val1, val2 in zip([1] + d, output_[j])]  # bilateral accumulation. First element is number of comps

        i += 1

    return output_

def Form_P_(input_, param=4, axes=(0, 1)):

    P_ = [] # list of P_

    param__ = zip(*input_) # put each param of inputs in a list
    s_ = [g > 0 for g in param__[param]]

    i = 0   # index of input at P's head
    j = 1   # index of input at P's tail

    while i < len(input_):

        # keep increasing j while coordinates are contiguous and signs are identical
        while j < len(input_) \
          and s[i] == s[j] \
          and not sum([input_[j][axis] != input_[j - 1][axis] for axis in axes[:-1]]) \
          and input_[j][axes[-1]] == input_[j - 1][axes[-1]] + 1:
            
            j += 1

        P_.append([s,
                   [sum([param for param in param_[i:j]]) for param_ in param__],
                   [dert for dert in input_[i:j]]
                    ])

        i = j
        j += 1
        
    return P_
