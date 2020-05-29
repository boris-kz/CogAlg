from time import time
from collections import deque, defaultdict
import numpy as np
from comp_pixel import comp_pixel
from utils import *
#from frame_blobs_adj import *
from frame_blobs_adj2 import *
import argparse


# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpeg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)


# Draw blobs --------------------------------------------------------------

for i, blob in enumerate(frame['blob__']):

    # initialize image with 3 channels (colour image)
    img_blob_ = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2], 3))
    img_blob_box = img_blob_.copy()

    # check if there are adjacent blobs and there are unmasked values
    if blob['adj_blob_'] and False in blob['dert__'][0].mask:
        dert__mask = ~blob['dert__'][0].mask # get inverted mask value (we need plot mask = false)
        dert__mask = dert__mask*155 # set intensity of colour

        # draw blobs into image
        img_blob_[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3],2] += dert__mask
        img_blob_box[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3],2] += dert__mask
        # draw bounding box

        cv2.rectangle(img_blob_box, (blob['box'][2], blob['box'][0]),
                                                (blob['box'][3], blob['box'][1]),
                                                color=(0, 0 ,255), thickness=1)

        for j, adj_blob in enumerate(blob['adj_blob_']):

            # check if there are unmasked values
            if False in adj_blob['dert__'][0].mask:
                adj_dert__mask = ~adj_blob['dert__'][0].mask # get inverted mask value (we need plot mask = false)
                adj_dert__mask = adj_dert__mask*155 # set intensity of colour

                # draw blobs into image
                img_blob_[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3],1] += adj_dert__mask
                img_blob_box[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3],1] += adj_dert__mask

                # draw bounding box
                cv2.rectangle(img_blob_box, (adj_blob['box'][2], adj_blob['box'][0]),
                                            (adj_blob['box'][3], adj_blob['box'][1]),
                                            color=(0, 255 ,0), thickness=1)
            else:
                break

            cv2.imwrite("images/adj_blob_mask2/mask_adj_blob_"+str(i)+".png", img_blob_.astype('uint8'))
            cv2.imwrite("images/adj_blob_mask2/mask_adj_blob_"+str(i)+"_box.png", img_blob_box.astype('uint8'))

