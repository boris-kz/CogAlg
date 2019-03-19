def compare(input_, offset, comparand=2, axes=(0, 1)):
    " Compare imputs over indicated offset "

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

    while i < len(input_) and j < len(input_):  # this loop is per i

        coord = [input_[i][axis] + offset[iaxis] for iaxis, axis in enumerate(axes)]    # coordinates of second comparand

        for iaxis, axis in enumerate(axes):     # iterate through axes from most-significant to least-significant
            while j < len(input_) and input_[j][axis] < coord[iaxis]:
                j += 1                          # increment j until conditions are met

            if input_[j][axis] == coord[iaxis]:
                temp = input_[j][comparand] - input_[i][comparand]  # compare
                d =  [temp * coef for coef in coef_]                # decomposition into value for each axis

                output_[i] = [val1 + val2 for val1, val2 in zip(d + [1], output_[i])]   # bilateral accumulation

                output_[j] = [val1 + val2 for val1, val2 in zip(d + [1], output_[j])]   # bilateral accumulation

        i += 1

    return output_