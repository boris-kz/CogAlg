def Sum_Params(param_):

    if type(param_[0]) not in {object, tuple}:
        return sum(param_)
    else:
        return [Sum_Params(sub_param_) for sub_param_ in zip(*param_)]

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
    t = False  # signal variable, to check matching of higher order axes' coordinates
    for iaxis, axis in enumerate(axes):  # iterate through the axes
        sub_axes = axes[:iaxis]  # list of higher order axes

        while i < len(input_):
            # increase i by 1 if current axis coordinates < target coordinate and all previous coordinates are equal to target coordinates:
            t = not sum([input_[i][saxis] != target[isaxis] for isaxis, saxis in
                         enumerate(sub_axes)])  # check matching of higher order axes' coordinates

            if not (t and input_[i][axis] < target[iaxis]):  # also look for matching of current axis coordinate
                break

            i += 1

        if i < len(input_):
            t = t and input_[i][axis] == target[iaxis] and i < len(input_)

            if not t:  # if there's a coordinate unmatch
                break

    return i, t  # return True if i points at the target coordinate
