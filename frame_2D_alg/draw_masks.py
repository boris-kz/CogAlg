from utils import *
from frame_blobs_alone import *
import argparse

def draw_masks(frame):

    for i in range(len(frame['blob__'])):  # masked = black, unmasked = white

        mask = 255 - (frame['blob__'][i]['dert__'].mask[0] * 255)
        img_mask = mask.astype('uint8')

        cv2.imwrite("images/masks/mask" + str(i) + ".bmp", img_mask)

# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpeg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)
draw_masks(frame)

# Draw blobs --------------------------------------------------------------

IMAGE_PATH = "./images/raccoon_eye.jpeg"
image = imread(IMAGE_PATH)

'''
def draw_blobs(frame, param):

    img_blob_ = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2]))
    box_ = []

    for i, blob in enumerate(frame['blob__']):
        if False in blob['dert__'][0].mask:  # if there is unmasked dert

            dert__ = blob['dert__'][param].data
            mask_index = np.where(blob['dert__'][0].mask == True)
            dert__[mask_index] = 0
            dert__ = dert__*255

            # draw blobs into image
            img_blob_[ blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3] ] += dert__
            box_.append(blob['box'])

    # uncomment to enable draw animation
    # plt.figure(dert__select); plt.clf()
    # plt.imshow(img_blobs.astype('uint8'))
    # plt.title('Blob Number ' + str(i))
    # plt.pause(0.001)

    return img_blob_.astype('uint8'), box_

iblob_, ibox_ = draw_blobs(frame, param=0)

for i in range(len(ibox_)):

    iblob_ = cv2.rectangle(iblob_, (ibox_[i][2], ibox_[i][0]), (ibox_[i][3], ibox_[i][1]), color=(750, 0 ,0),
                           thickness=1)  # cv2 rectangle: (xstart,ystart) (xstop, ystop)

cv2.imwrite("images/mask_blobs.png", iblob_)

# gblob_, gbox_ = draw_blobs(frame, param=1)
# dert params: 0 = i, 1 = g, 2 = dy, 3 = dx, but images should only show box masks
#    gblob_ = cv2.rectangle(gblob_, (gbox_[i][2], gbox_[i][0]), (gbox_[i][3], gbox_[i][1]), color=(750, 0 ,0),
#                           thickness=1)  # box = [ystart, ystop, xstart,xstop]
# cv2.imwrite("images/blobs1.png", gblob_)

# Check if all dert__s contain one blob ---------------------------------------

i_non_contiguous_top_bottom = []
blob_non_contiguous_top_bottom = []
i_empty = []
blob_empty = []
i_non_contiguous_within_dert= []
blob_non_contiguous_within_dert = []


# loop across all blobs
for i, blob in enumerate(frame['blob__']):
    dert__ = blob['dert__']
    
    # loop in each row
    for y in range(dert__.shape[1]):
        # if there is no unmasked dert in the row but there is some unmasked dert within the blob
        if ((False in dert__.mask[0][y,:]) == False) and ((False in dert__.mask[0]) == True):
            if y == 0 or y == dert__.shape[1]:
                # non-contiguous in top or bottom row of dert
                blob_non_contiguous_top_bottom.append(blob)
                i_non_contiguous_top_bottom.append(i)
                
        if ((False in dert__.mask[0][y,:]) == False) and ((False in dert__.mask[0]) == True):       
               
            if y != 0 or y != dert__.shape[1]:
                # non-contiguous in rows other than top and bottom row
                blob_non_contiguous_within_dert.append(blob)
                i_non_contiguous_within_dert.append(i)
            
        # if there is totally no unmasked dert in the blob
        if (False in dert__.mask[0]) == False:
            blob_empty.append(blob)
            i_empty.append(i)
            
            
i_gap_top_bottom = []
blob_gap_top_bottom = []
i_empty = []
blob_empty = []
i_gap_in_dert = []
blob_gap_in_dert = []

for i, blob in enumerate(frame['blob__']):
    dert__ = blob['dert__']
    for y in range(dert__.shape[0]):
        # if no unmasked dert in row but some unmasked dert in dert__:
        if (False in dert__.mask[0][y, :]) == False and (False in dert__.mask[0]) == True:

            if y == 0 or y == dert__.shape[0]:  # gap in top or bottom row of dert__
                blob_gap_top_bottom.append(blob)
                i_gap_top_bottom.append(i)
            else:
                blob_gap_in_dert.append(blob)  # gap in other rows
                i_gap_in_dert.append(i)
            break
        if (False in dert__.mask[0]) == False:  # no unmasked derts in the blob
            blob_empty.append(blob)
            i_empty.append(i)
            break
        break
'''