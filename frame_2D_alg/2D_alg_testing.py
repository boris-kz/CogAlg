"""
Script for testing 2D alg.
Change quickly in parallel with development.

Currently testing: intra_blob.form_P__
"""

import numpy as np
import matplotlib.pyplot as plt
import frame_blobs

from utils import imread, draw, debug_segment
from comp_i import comp_i
from intra_blob import form_P__, scan_P__, form_segment_

# -----------------------------------------------------------------------------
# Adjustable parameters

image_path = "../images/raccoon_eye.jpg"
output_path = "../visualization/images/2D_alg_test_out"

# -----------------------------------------------------------------------------
# Adjustable parameters

def normalize(a):
    return (a - a.min()) / (a.max() - a.min())

if __name__ == "__main__":
    print('Reading image...')
    image = imread(image_path)
    print('Done!')

    print('Doing first comp...')
    frame = frame_blobs.image_to_blobs(image)
    print('Done!')

    print('Extracting best blob...')
    best_blob = sorted(frame['blob_'],
                       key=lambda blob: blob['Dert']['G'])[0]
    print('Done!')

    print('Doing angle comp on best blob...')
    derts = comp_i(best_blob['dert__'], 1, fa=1)
    print('Done!')

    print('Running form_P__...')
    y0, yn, x0, xn = best_blob['box']
    P__ = form_P__(x0, y0, derts, 50, fa=1, noM=1)
    print('Done!')

    print('Running scan_P__...')
    P_ = scan_P__(P__)
    print('Done!')

    print('Running form_seg_...')
    seg_ = form_segment_(P_, fa=1, noM=1)
    print('Done!')

    print('Debugging segments...')
    seg_ = seg_
    gray_scale = plt.get_cmap('gray')
    draw(output_path, debug_segment(derts.shape[1:], *seg_))
    for seg in seg_:
        plt.imshow(debug_segment(derts.shape[1:], seg), cmap=gray_scale)
        plt.show()
    print('Done!')