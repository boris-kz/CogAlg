import cv2
import argparse
from time import time
from collections import deque

''' line_POC() in conventional notation: easier to understand variable names, but harder to follow operations. 
    This format is meant to ease transition to abbreviated variable names and more-2D format of line_POC() code.
    Incremental abbreviation and more in-line code and comments is the only escape from drowning in a sea of names 
    
    core algorithm level 1: 1D-only proof of concept, 
    applied here to process lines of grey-scale pixels but not effective in recognition of 2D images. 

    Cross-comparison between consecutive pixels within horizontal scan line (row).
    Resulting difference patterns dPs (spans of pixels forming same-sign differences)
    and relative match patterns vPs (spans of pixels forming same-sign predictive value)
    are redundant representations of each line of pixels.
'''

def recursive_comparison(x,
                         pixel,
                         prior_pixel,
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

    difference = pixel - prior_pixel
    match = min(pixel, prior_pixel)
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
                           prior_pixel,
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
                           prior_pixel,
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


def pre_recursive_comparison(pattern_type, elements_array, accumulated_average_match, comparison_range):
    accumulated_average_match += average_match
    elements_array_length = len(elements_array)

    overlap, value_pattern_array, difference_pattern_array = 0, [], []
    value_pattern = 0, 0, 0, 0, 0, [], []
    difference_pattern = 0, 0, 0, 0, 0, [], []

    if pattern_type:
        comparison_range += 1
        for x in range(comparison_range + 1, elements_array_length):

            pixel = elements_array[x][0]
            prior_pixel, fuzzy_difference, fuzzy_value = elements_array[x - comparison_range]

            fuzzy_difference,\
            fuzzy_value,\
            value_pattern,\
            difference_pattern,\
            value_pattern_array,\
            difference_pattern_array,\
            overlap = recursive_comparison(x,
                                           pixel,
                                           prior_pixel,
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
        prior_difference = elements_array[0]
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
                                           prior_difference,
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

            prior_difference = difference

    return value_pattern_array, difference_pattern_array


def form_pattern(pattern_type,
                 pattern,
                 alternative_type_pattern,
                 pattern_array,
                 alternative_type_pattern_array,
                 overlap,
                 prior_pixel,
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
    summed_difference,\
    summed_value,\
    recursion_flag,\
    elements_array,\
    overlap_array = pattern

    if pixel_index > comparison_range + 2 and (sign != prior_sign or pixel_index == image_width - 1):
        if pattern_type:
            if len(elements_array) > comparison_range + 3\
                    and prior_sign == 1\
                    and summed_value > average_match + average_summed_value:
                recursion_flag = 1
                elements_array.append(pre_recursive_comparison(1, elements_array, average_match, comparison_range))

        else:
            if len(elements_array) > 3 and abs(summed_difference) > average_match + average_summed_difference:
                recursion_flag = 1
                comparison_range = 1
                elements_array.append(pre_recursive_comparison(0, elements_array, average_match, comparison_range))

        pattern = prior_sign, \
                summed_pixels, \
                summed_difference, \
                summed_value,\
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
        summed_difference, \
        summed_value,\
        recursion_flag,\
        elements_array,\
        overlap_array = 0, 0, 0, 0, 0, [], []

    prior_sign = sign
    summed_pixels += prior_sign
    summed_difference += fuzzy_difference
    summed_value += fuzzy_value

    if pattern_type:
        elements_array.append((prior_pixel, fuzzy_difference, fuzzy_value))
    else:
        elements_array.append(fuzzy_difference)

    pattern = prior_sign, \
            summed_pixels, \
            summed_difference, \
            summed_value,\
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
                     comparison_range):
    index = 0

    for incomplete_tuple in incomplete_tuples_array:
        prior_pixel, fuzzy_difference, fuzzy_match = incomplete_tuple

        difference = pixel - prior_pixel
        match = min(pixel, prior_pixel)

        fuzzy_difference += difference
        fuzzy_match += match

        incomplete_tuples_array[index] = (prior_pixel, fuzzy_difference, fuzzy_match)
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
                               prior_pixel,
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
                               prior_pixel,
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
    image_height, image_width = image.shape

    frame_of_patterns = []

    for line_index in range(image_height):
        line = image[line_index]

        overlap = 0
        value_pattern_array = []
        difference_pattern_array = []

        value_pattern = [0, 0, 0, 0, 0, [], []]
        difference_pattern = [0, 0, 0, 0, 0, [], []]

        incomplete_tuples_array = deque(maxlen=comparison_range)
        incomplete_tuples_array.append((line[0], 0, 0))

        for pixel_index in range(image_width):
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
                                       image_width,
                                       comparison_range)

        line_of_patterns = value_pattern_array, difference_pattern_array
        frame_of_patterns.append(line_of_patterns)

    return frame_of_patterns


# add argument parser
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# initialize constants
comparison_range = 3
average_match = 64 * comparison_range
average_summed_value = 63
average_summed_difference = 63

start_time = time()
frame_of_patterns = pixels_to_patterns(image)
end_time = time() - start_time
print(end_time)
