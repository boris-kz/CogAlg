"""
Script for testing 2D alg.
Change quickly in parallel with development.

Currently testing: intra_blob.form_P__
"""

import numpy as np
import matplotlib.pyplot as plt
import frame_blobs

from utils import imread, imwrite, debug_segment, debug_blob
from comparison import comp_v
from intra_blob import form_P__, scan_P__, form_segment_, form_blob_

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
    derts = comp_v(best_blob['dert__'], 1, fa=1)
    print('Done!')

    print('Running form_P__...')
    y0, yn, x0, xn = best_blob['box']
    P__ = form_P__(derts, 50, nI=4, dderived=0, x0=x0, y0=y0)
    print('Done!')

    print('Running scan_P__...')
    P_ = scan_P__(P__)
    print('Done!')

    print('Running form_segment_...')
    seg_ = form_segment_(P_, fa=1)
    print('Done!')

    print('Running form_blob_...')
    blob_ = form_blob_(seg_, derts, root_fork={'root_blob':None}, nI=4)
    print('Done!')

    print('Debugging blobs...')
    gray_scale = plt.get_cmap('gray')
    imwrite(output_path, debug_blob(derts.shape[1:], *blob_))
    print('Done!')
    for blob in blob_:
        plt.imshow(debug_blob(derts.shape[1:], blob), cmap=gray_scale)
        plt.show()