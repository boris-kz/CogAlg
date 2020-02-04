"""
Script for testing 2D alg.
Change quickly in parallel with development.
"""

from timeit import timeit

from comp_pixel import comp_pixel
from utils import *

# ----------------------------------------------------------------------------
# Constants

IMAGE_PATH = "./images/raccoon.jpg"
OUTPUT_PATH = "./images/outputs"
TIMES_TO_RUN = 10

# ----------------------------------------------------------------------------
# Test Functions

def read_image_for_tests(img_path=IMAGE_PATH):
    print(f"Reading image from '{img_path}'... ", end='')
    img = imread(img_path)
    print('Done!\n...')
    return img


def test_comp_pixel(img, output_path=OUTPUT_PATH):
    print('Testing comp_pixel...')

    print('Running comp_pixel on the image... ', end='')
    gdert__, rdert__ = comp_pixel(img)
    print('Done!')

    for name, v__ in zip(  # v__ take one of the following values each loop:
        ('gp', 'gg', 'gdy', 'gdx',
         'rp', 'rg', 'rdy', 'rdx'),
        (*gdert__, *rdert__),
    ):
        path = f'{output_path}/{name}.jpg'
        print(f"Writing {name} image to '{path}' ... ", end='')
        imwrite(path, array2image(v__))
        print('Done!')

    print('...')

def time_comp_pixel(img_path=IMAGE_PATH, number=TIMES_TO_RUN):
    print('Timing comp_pixel...')
    t = timeit(setup=f'from utils import imread\n'
                     f'from comp_pixel import comp_pixel\n'
                     f'img = imread("{img_path}")',
               stmt='comp_pixel(img)',
               number=number)
    print(f'comp_pixel ran in {t/number} seconds.')
    print('...')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    img = read_image_for_tests(IMAGE_PATH)
    test_comp_pixel(img, OUTPUT_PATH)
    time_comp_pixel(IMAGE_PATH)