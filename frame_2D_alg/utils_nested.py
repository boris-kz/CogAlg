from copy import deepcopy as dcopy
import numpy as np


# ----------------------------------------------------------------------------
# General purpose functions for nested operation


def nested(element__, function, *args):  # provided function operates on nested variable

    if isinstance(element__, list):
        if len(element__) > 1 and isinstance(element__[0], list):
            for i, element_ in enumerate(element__):
                element__[i] = nested(element_, function, *args)
        else:
            element__ = function(element__, *args)
    else:
        element__ = function(element__, *args)
    return element__


def nested2(element1__, element2__, function):  # provided function operates on 2 nested variables

    element__ = dcopy(element1__)
    if isinstance(element1__[0], list):
        for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
            element__[i] = nested2(element1_, element2_, function)
    else:
        element__ = function(element1__, element2__)
    return element__


# ----------------------------------------------------------------------------
# Single purpose function for nested operation in intra_comp


def calc_a(element1__, element2__, ave):  # nested compute a from gy,gx, g and ave

    element__ = dcopy(element1__)
    if isinstance(element2__[0], list):
        for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
            element__[i] = calc_a(element1_, element2_, ave)
    else:
        if isinstance(element2__, list):
            for i, (element1_, element2_) in enumerate(zip(element1__, element2__)):
                element__[i] = [element1_[0] / element2_, element1_[1] / element2_]
        else:
            element__ = [element1__[0] / element2__, element1__[1] / element2__]

    return element__


def shift_topleft(element_):  # shift variable in top left direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, :-1]
    else:
        element_ = element_[:-1, :-1]
    return element_


def shift_topright(element_):  # shift variable in top right direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[:-1, 1:]
    else:
        element_ = element_[:-1, 1:]
    return element_


def shift_botright(element_):  # shift variable in bottom right direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, 1:]
    else:
        element_ = element_[1:, 1:]
    return element_


def shift_botleft(element_):  # shift variable in bottom left direction

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[1:, :-1]
    else:
        element_ = element_[1:, :-1]
    return element_


def negative_nested(element_):  # complement all values in the variable

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = -element
    else:
        element_ = -element_
    return element_


def replace_zero_nested(element_):  # replace all 0 values in the variable with 1

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element[np.where(element == 0)] = 1
            element_[i] = element
    else:
        element_[np.where(element_ == 0)] = 1
    return element_


def hypot_nested(element1, element2):  # hypot of 2 elements

    return [np.hypot(element1[0], element2[0]), np.hypot(element1[1], element2[1])]


def hypot_add1_nested(element1, element2):  # hypot of 2 (elements+1)

    return [np.hypot(element1[0] + 1, element2[0] + 1), np.hypot(element1[1] + 1, element2[1] + 1)]


def add_nested(element1, element2):  # sum of 2 variables

    return [element1[0] + element2[0], element1[1] + element2[1]]


def subtract_nested(element1, element2):  # difference of 2 variables

    return [element1[0] - element2[0], element1[1] - element2[1]]


def arctan2_nested(element1, element2):  # arctan of 2 variables

    return [np.arctan2(element1[0], element2[0]), np.arctan2(element1[1], element2[1])]


# ----------------------------------------------------------------------------
# Single purpose function for nested operation in intra_blob


def nested_crop(element_, *args):  # crop element based on coordinates in box

    y0e = args[0][0]
    yne = args[0][1]
    x0e = args[0][2]
    xne = args[0][3]

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i] = element[y0e:yne, x0e:xne]
    else:
        element_ = element_[y0e:yne, x0e:xne]

    return element_


def nested_accum_blob_Dert(element_, *args):  # accumulate parameters based on the param value and their x,y coordinates

    param = args[0]
    y = args[1]
    x = args[2]

    if isinstance(element_, list):
        for i, element in enumerate(element_):
            element_[i][y, x] += param

    else:
        element_[y, x] += param

    return element_
