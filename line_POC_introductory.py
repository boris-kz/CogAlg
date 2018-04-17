import cv2
import argparse

from time import time
from collections import deque


def recursive_comparison(x,
                         pixel,
                         previous_pixel,
                         fuzzy_difference,
                         fuzzy_value,
                         value_pattern,
                         difference_pattern,
                         value_pattern_array,
                         difference_pattern_array,
                         overlap,
                         elements_array_length,
                         average_match,
                         comparison_range):

    difference = pixel - previous_pixel
    match = min(pixel, previous_pixel)
    value = match - average_match

    fuzzy_difference += difference
    fuzzy_value += value

    value_pattern,\
    difference_pattern,\
    value_pattern_array,\
    difference_pattern_array,\
    overlap = form_pattern(1,
                           value_pattern,
                           difference_pattern,
                           value_pattern_array,
                           difference_pattern_array,
                           overlap,
                           previous_pixel,
                           fuzzy_difference,
                           fuzzy_value,
                           x,
                           elements_array_length,
                           average_match,
                           comparison_range)

    difference_pattern,\
    value_pattern,\
    difference_pattern_array,\
    value_pattern_array,\
    overlap = form_pattern(0,
                           difference_pattern,
                           value_pattern,
                           difference_pattern_array,
                           value_pattern_array,
                           overlap,
                           previous_pixel,
                           fuzzy_difference,
                           fuzzy_value,
                           x,
                           elements_array_length,
                           average_match,
                           comparison_range)

    overlap += 1

    return fuzzy_difference,\
           fuzzy_value,\
           value_pattern,\
           difference_pattern,\
           value_pattern_array,\
           difference_pattern_array,\
           overlap


def recursive_comparison_control(pattern_type, elements_array, accumulated_average_match, comparison_range):
    accumulated_average_match += average_match
    elements_array_length = len(elements_array)

    overlap, value_pattern_array, difference_pattern_array = 0, [], []
    value_pattern = 0, 0, 0, 0, 0, [], []
    difference_pattern = 0, 0, 0, 0, 0, [], []

    if pattern_type:
        comparison_range += 1
        for x in range(comparison_range + 1, elements_array_length):

            pattern, internal_fuzzy_difference, internal_fuzzy_value = elements_array[x]
            previous_pixel, fuzzy_difference, fuzzy_value = elements_array[x - comparison_range]

            fuzzy_difference,\
            fuzzy_value,\
            value_pattern,\
            difference_pattern,\
            value_pattern_array,\
            difference_pattern_array,\
            overlap = recursive_comparison(x,
                                           pattern,
                                           previous_pixel,
                                           fuzzy_difference,
                                           fuzzy_value,
                                           value_pattern,
                                           difference_pattern,
                                           value_pattern_array,
                                           difference_pattern_array,
                                           overlap,
                                           elements_array_length,
                                           average_match,
                                           comparison_range)

    else:
        previous_difference = elements_array[0]
        fuzzy_difference, fuzzy_value = 0, 0

        for x in range(1, elements_array_length):
            difference = elements_array[x]
            fuzzy_difference,\
            fuzzy_value,\
            value_pattern,\
            difference_pattern,\
            value_pattern_array,\
            difference_pattern_array,\
            overlap = recursive_comparison(x,
                                           difference,
                                           previous_difference,
                                           fuzzy_difference,
                                           fuzzy_value,
                                           value_pattern,
                                           difference_pattern,
                                           value_pattern_array,
                                           difference_pattern_array,
                                           overlap,
                                           elements_array_length,
                                           average_match,
                                           comparison_range)

            previous_difference = difference

    return value_pattern_array, difference_pattern_array


def form_pattern(pattern_type,
                 pattern,
                 alternative_type_pattern,
                 pattern_array,
                 alternative_type_pattern_array,
                 overlap,
                 previous_pixel,
                 fuzzy_difference,
                 fuzzy_value,
                 pixel_index,
                 image_width,
                 average_match,
                 comparison_range):

    if pattern_type:
        sign = 1 if fuzzy_value >= 0 else 0
    else:
        sign = 1 if fuzzy_difference >= 0 else 0

    prior_sign,\
    summed_pixels,\
    summed_differences,\
    summed_values,\
    recursion_flag,\
    elements_array,\
    overlap_array = pattern

    if pixel_index > comparison_range + 2 and (sign != prior_sign or pixel_index == image_width - 1):
        if pattern_type:
            if len(elements_array) > comparison_range + 3\
                    and prior_sign == 1\
                    and summed_values > average_match + average_summed_values:
                recursion_flag = 1
                elements_array.append(recursive_comparison_control(1, elements_array, average_match, comparison_range))

        else:
            if len(elements_array) > 3 and abs(summed_differences) > average_match + average_summed_difference:
                recursion_flag = 1
                comparison_range = 1
                elements_array.append(recursive_comparison_control(0, elements_array, average_match, comparison_range))

        pattern = prior_sign, \
                summed_pixels, \
                summed_differences, \
                summed_values,\
                recursion_flag,\
                elements_array,\
                overlap_array

        pattern_array.append(pattern)

        o = len(pattern_array), overlap
        alternative_type_pattern[6].append(o)

        o = len(alternative_type_pattern_array), overlap
        overlap_array.append(o)

        prior_sign, \
        summed_pixels, \
        summed_differences, \
        summed_values,\
        recursion_flag,\
        elements_array,\
        overlap_array = 0, 0, 0, 0, 0, [], []

    prior_sign = sign
    summed_pixels += prior_sign
    summed_differences += fuzzy_difference
    summed_values += fuzzy_value

    if pattern_type:
        elements_array.append((previous_pixel, fuzzy_difference, fuzzy_value))
    else:
        elements_array.append(fuzzy_difference)

    pattern = prior_sign, \
            summed_pixels, \
            summed_differences, \
            summed_values,\
            recursion_flag,\
            elements_array,\
            overlap_array

    return pattern, \
           alternative_type_pattern, \
           pattern_array, \
           alternative_type_pattern_array, \
           overlap


def pixel_comparison(pixel_index,
                     pixel,
                     incomplete_tuples_array,
                     value_pattern,
                     difference_pattern,
                     value_pattern_array,
                     difference_pattern_array,
                     overlap,
                     image_width,
                     average_match,
                     comparison_range):
    index = 0

    for incomplete_tuple in incomplete_tuples_array:
        previous_pixel, fuzzy_difference, fuzzy_match = incomplete_tuple

        difference = pixel - previous_pixel
        match = min(pixel, previous_pixel)

        fuzzy_difference += difference
        fuzzy_match += match

        incomplete_tuples_array[index] = (previous_pixel, fuzzy_difference, fuzzy_match)
        index += 1

    if len(incomplete_tuples_array) == comparison_range:
        fuzzy_value = fuzzy_match - average_match

        value_pattern, \
        difference_pattern, \
        value_pattern_array, \
        difference_pattern_array, \
        overlap = form_pattern(1,
                               value_pattern,
                               difference_pattern,
                               value_pattern_array,
                               difference_pattern_array,
                               overlap,
                               previous_pixel,
                               fuzzy_difference,
                               fuzzy_value,
                               pixel_index,
                               image_width,
                               average_match,
                               comparison_range)

        difference_pattern,\
        value_pattern,\
        difference_pattern_array,\
        value_pattern_array,\
        overlap = form_pattern(0,
                               difference_pattern,
                               value_pattern,
                               difference_pattern_array,
                               value_pattern_array,
                               overlap,
                               previous_pixel,
                               fuzzy_difference,
                               fuzzy_value,
                               pixel_index,
                               image_width,
                               average_match,
                               comparison_range)

        overlap += 1

    incomplete_tuples_array.appendleft((pixel, 0, 0))

    return incomplete_tuples_array, value_pattern, difference_pattern, value_pattern_array, difference_pattern_array, overlap


def pixels_to_patterns(image):
    height, width = image.shape

    frame_of_patterns = []

    for line_index in range(height):
        line = image[line_index]

        overlap = 0
        value_pattern_array = []
        difference_pattern_array = []

        value_pattern = [0, 0, 0, 0, 0, [], []]
        difference_pattern = [0, 0, 0, 0, 0, [], []]

        incomplete_tuples_array = deque(maxlen=comparison_range)
        incomplete_tuples_array.append((line[0], 0, 0))

        for pixel_index in range(width):
            pixel = line[pixel_index]

            incomplete_tuples_array, \
            value_pattern, \
            difference_pattern, \
            value_pattern_array, \
            difference_pattern_array, \
            overlap = pixel_comparison(pixel_index,
                                       pixel,
                                       incomplete_tuples_array,
                                       value_pattern, difference_pattern,
                                       value_pattern_array,
                                       difference_pattern_array,
                                       overlap,
                                       width,
                                       average_match,
                                       comparison_range)

        line_of_patterns = value_pattern_array, difference_pattern_array
        frame_of_patterns.append(line_of_patterns)

    return frame_of_patterns


# add argument parser
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/racoon.jpg')
arguments = vars(argument_parser.parse_args())

# initialize constants
average_match = 63
average_summed_values = 63
average_summed_differences = 63
comparison_range = 3
average_summed_value = average_summed_values * comparison_range
average_summed_difference = average_summed_differences * comparison_range
average_match = average_match * comparison_range

# start time tracking
start_time = time()

# read image and form patterns
image = cv2.imread(arguments['image'], 0).astype(int)
frame_of_patterns = pixels_to_patterns(image)

# end time tracking
end_time = time() - start_time
print(end_time)
