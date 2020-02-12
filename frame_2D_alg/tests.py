"""
Script for testing 2D alg.
Change quickly in parallel with development.
"""

from time import time
import operator as op

from comp_pixel import comp_pixel
from intra_comp import (
    translated_operation, angle_diff,
)
from test_sets import (
    comp_pixel_test_pairs,
)
from utils import *


# ----------------------------------------------------------------------------
# Constants

# ----------------------------------------------------------------------------
# Functions

def test_function(func, test_pairs, eval_func=op.eq):
    """Meta test function. Run tests for a function."""
    start_time = time()
    print(f'Testing {func.__str__()}... ', end='')
    for inp, out in test_pairs:
        o = func(*inp)
        if not eval_func(o, out):
            print('\n\nTest failed on:')
            print('Input:')
            print(inp)
            print('Output:')
            print(out)
            print('Real output:')
            print(o)
            break
    else:
        print('Test success!')

    time_elapsed = time() - start_time
    test_cnt = len([*test_pairs])
    print(f'Finished {test_cnt} '
          f'{"tests" if test_cnt > 1 else "test"} '
          f'in {time_elapsed} seconds.\n')


# ----------------------------------------------------------------------------
# Module utils' test functions

def test_is_close(
        test_pairs=(
                (('asd', ''), False),
                ((1235, 1234.99), True),
                ((7, 'a'), False),
                ((np.array([12.1999]), np.array([12.2])), True),
                (('Dong Thap', 'Dong thap'), False),
                (('can tho', 'can tho'), True),
        ),
):
    """Run tests for is_close function in utils."""
    test_function(is_close, test_pairs)


# ----------------------------------------------------------------------------
# Module comp_pixel test functions

def test_comp_pixel(
        test_pairs=comp_pixel_test_pairs,
        eval_func=is_close,  # To check float equality
):
    """Test comp_pixels."""
    test_function(comp_pixel, test_pairs, eval_func)


# ----------------------------------------------------------------------------
# Module extend_comp test functions

def test_translated_operation(
        test_pairs=(
                ( # rng = 1, with subtraction as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         1, op.sub, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[-4, -3, -2, 1, 4, 3, 2, -1]]]),

                ),
                ( # rng = 1, with addition as the operator
                        # inputs
                        (np.array([[0, 1, 2],
                                   [3, 4, 5],
                                   [6, 7, 8]]),
                         1, op.add, # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[4, 5, 6, 9, 12, 11, 10, 7]]]),

                ),
                (  # rng = 2, with subtraction as the operator
                        # inputs (with edited value of 15)
                        (np.array([[ 0,  1,  2,  3,  4,  5],
                                   [ 6,  7,  8,  9, 10, 11],
                                   [12, 13, 14, 20, 16, 17],
                                   [18, 19, 20, 21, 22, 23],
                                   [24, 25, 26, 27, 28, 29]]),
                         2, op.sub,  # rng = 1 and operator = subtract
                         ),
                        # output
                        np.array([[[-14, -13, -12, -11, -10, -4,  2,   8,
                                     14,  13,  12,  11,  10,  4, -2,  -8],
                                   [-19, -18, -17, -16, -15, -9, -3,   3,
                                      9,   8,   7,   6,   5, -1, -7, -13]]]),

                ),
        ),
        eval_func=is_close,
):
    """Test translated_operation."""
    test_function(translated_operation, test_pairs, eval_func)


def test_angle_diff(
        test_pairs=(
                ( # 0 - 90 == -90 degrees
                        # inputs
                        ((0, 1),  # 0 degrees
                         (1, 0)), # 90 degrees
                        # output
                        (-1, 0), # 0 degrees
                ),
                ( # 135 - 90 = 45 degrees
                        # inputs
                        ((0.5**2, -0.5**2), # 135 degrees
                         (1, 0)),           # 90 degrees
                        # output
                        (0.5**2, 0.5**2), # 45 degrees
                ),
                ( # 90 - 30 = 60 degrees
                        # inputs
                        ((1, 0), # 135 degrees
                         (0.75**2, 0.5)),           # 90 degrees
                        # output
                        (0.5, 0.75**2), # 45 degrees
                ),
        ),
        eval_func=is_close,
):
    """Test angle_diff."""
    test_function(angle_diff, test_pairs, eval_func)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
    # Put function tests here
    # test_comp_pixel()
    test_angle_diff()
    test_translated_operation()