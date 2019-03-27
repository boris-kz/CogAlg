from Utilities import Go_to, Sum_Params

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
    k = 0   # _P's index

    while i < len(input_):

        # keep increasing j while coordinates are contiguous and signs are identical
        while j < len(input_) \
          and s_[i] == s_[j] \
          and not sum([input_[j][axis] != input_[j - 1][axis] for axis in axes[:-1]]) \
          and input_[j][axes[-1]] == input_[j - 1][axes[-1]] + 1:
            
            j += 1

        P = [s_[i],
             [j - i] + [Sum_Params(param_[i:j]) for param_ in param__],
             [dert for dert in input_[i:j]] ]

        # scan P for connections with higher-line Ps:

        fork_ = []
        y = P[2][0][0]  # first param (y coordinate) of first element of P
        x_first = P[2][0][1]  # second param (x coordinate)of first element of P
        x_last = P[2][-1][1]  # second param (x coordinate)of last element of P

        while k < len(P_):  # increases k until end of list is met or encounters stopping conditions below

            _P = P_[k]

            _y = _P[2][0][0]    # first param (y coordinate) of first element of _P

            if y == _y + 1:     # if _P is on the higher line of P

                _x_first = _P[2][0][1]  # second param (x coordinate) of first element of _P
                _x_last = _P[2][-1][1]  # second param (x coordinate)of last element of _P

                if P[0] == _P[0] and x_first <= _x_last and _x_first <= x_last:     # if s are the same and x coordinates overlap
                    fork_.append(_P)

                if _x_last >= x_last:
                    break

            elif _y >= y:
                break

            k += 1


        P_.append((P, fork_))

        i = j
        j += 1

        # scan P
        
    return P_