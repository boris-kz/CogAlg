from time import time
from collections import deque, defaultdict
import numpy as np
from comp_pixel import comp_pixel
from utils import *
from frame_blobs import *
import argparse

# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon.jpg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)

# Draw blobs --------------------------------------------------------------

from matplotlib import pyplot as plt


def draw_blobs(frame, dert__select):
    img_blobs = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2]))

    # loop across blobs
    for i, blob in enumerate(frame['blob__']):

        # if there is unmask dert
        if False in blob['dert__'][0].mask:
            # get dert value from blob
            dert__ = blob['dert__'][dert__select].data
            # get the index of mask
            mask_index = np.where(blob['dert__'][0].mask == True)
            # set masked area as 0
            dert__[mask_index] = 0

            # draw blobs into image
            img_blobs[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3]] += dert__

            # uncomment to enable draw animation
    #            plt.figure(dert__select); plt.clf()
    #            plt.imshow(img_blobs.astype('uint8'))
    #            plt.title('Blob Number ' + str(i))
    #            plt.pause(0.001)

    return img_blobs.astype('uint8')


# -----------------------------------------------------------------------------

# draw blobs
# select dert components
# 0 = i
# 1 = g
# 2 = dy
# 3 = dx
iblobs = draw_blobs(frame, dert__select=0)
gblobs = draw_blobs(frame, dert__select=1)

# save to disk
cv2.imwrite("images/iblobs_draft.png", iblobs)
cv2.imwrite("images/gblobs_draft.png", gblobs)
