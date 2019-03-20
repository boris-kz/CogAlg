def compare(input_, offset, comparand=2, axes=(0, 1)):
    ''' Compare imputs over indicated offset. Return differences between compared inputs.
        - input_: list of inputs
        - offset: offset between comparands.
        - comparand: index of comparand in each input
        - axes: indices of axes' coordinates of each input
        Return: a list where each element is a list:
            + First element is number of comps (not every input is compared twice)
            + Remaining elements: difference value for every axis

        For example: compare(input_, offset=(0, 1), comparand=2, axes=(0, 1))
            With each input: (y, x, p):
            + Coordinates: y, x (axes' indices = (0, 1))
            + Comparand: p (comparand index = 2)
            + Pure horizontal comp, with rng = 1: comp between input with coordinates: (y, x) and (y, x + 1)
            (offset = (0, 1))
            Return: each output: [num_comp, dy, dx] '''

    squared_mag = 0                         # squared magnitude of offset
    for d in offset:                        # offset between comparands
        mag += d ** 2

    mag = squared_mag ** 0.5                # magnitude of offset
    coef_ = [d / mag for d in offset]       # decomposition coefficients

    # initialize output:
    output_ = [[0] * (len(axes) + 1)] * len(input_)   # for every outputs: results for each axis + number of comps

    # initialize indices:
    i = 0   # index of first comparand
    j = 0   # index of second comparand

    while i < len(input_) and j < len(input_):  # this loop is per i. j is updated accordingly. If either i or j is out of bound, no more comp could take place

        coord = [input_[i][axis] + offset[iaxis] for iaxis, axis in enumerate(axes)]    # coordinates of second comparand
        comp = True

        for iaxis, axis in enumerate(axes):     # iterate through axes from most-significant to least-significant
            while j < len(input_) and input_[j][axis] < coord[iaxis]:
                j += 1                          # increment j until conditions are met

            if input_[j][axis] != coord[iaxis]:     # if stuck on coordinate of one axis
                pass                                # skip the rest
                comp = False                        # do not comp (offset between input_[i] and input_[j] is not correct

        if comp:
            temp = input_[j][comparand] - input_[i][comparand]  # compare by subtraction
            d =  [temp * coef for coef in coef_]                # decomposition into value for each axis

            output_[i] = [val1 + val2 for val1, val2 in zip([1] + d, output_[i])]   # bilateral accumulation. Last element is number of comps
            output_[j] = [val1 + val2 for val1, val2 in zip([1] + d, output_[j])]   # bilateral accumulation. Last element is number of comps

        i += 1

    return output_