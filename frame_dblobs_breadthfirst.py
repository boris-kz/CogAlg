import cv2
import argparse
from scipy import misc
from time import time
from collections import deque
import numpy

'''
    This is a copy I wrote for revisualizing frame_dblobs, also being used for aiding in testing and debugging
    frame_dblobs.py. What it can do:
        - Computes individual fuzzy d of each pixel
        - Count number of blobs
        - Output the image of blobs, negative blob is black, positive one is white 
    Has 5 steps:
        - Computes and put individual fuzzy d of each pixel into a 2D array
        - Change elements of resulting array into 0s and 255s
        - Use Breadth-first search algorithm to count number of blobs (Can be extended to do more blobs related tasks)
        - Output number of blobs into a text file
        - [Optional] output the image of blobs, if required
'''

def black_white( img, threshold = 0 ):
    "Maps image into 2 black and white"
    if img > threshold:
        return 255
    else:
        return 0

def difference( p, pri_p ):
    "Get difference"
    return p - pri_p

def BFS( image ):
    "Counts the number of blobs"
    dir_x = [0, 1, 0, -1]
    dir_y = [-1, 0, 1, 0]   # Declared for neat spreading-search: go one step up, right, down or left
    counter = 0             # Output of the function
    has_been_here = numpy.array([[False] * X ] * Y) # flags to know if a location has been stepped on before
    for y in range(Y):
        for x in range(X):
            if not has_been_here[ y , x ]:
                # If the iterator steps on a pixel that hasn't been stepped on before, it has found a new blob
                counter += 1;
                # The rest of the code in this loop search through same d sign pixels that is from the same blob
                val = image[y,x]          # Initialize the value used to indicate if a pixel has the same d sign with the blob
                spread_queue = [(y,x)]      # Any adjacent, same d sign pixel will be added to this queue to continue the search
                has_been_here[y,x] = True   # Mark the current location as searched
                while spread_queue:
                    this_y, this_x = spread_queue.pop(0)    # Take a location from the queue to spread the search
                    for direction in range(4):              # Search in 4 directions
                        spread_y, spread_x = this_y + dir_y[direction], this_x + dir_x[direction]
                        # (spread_x, spread_y): coordinate of adjacent location of (this_x, this_y)
                        if spread_y >= 0 and spread_y < Y \
                        and spread_x >= 0 and spread_x < X \
                        and not has_been_here[spread_y, spread_x] \
                        and image[spread_y, spread_x] == val :
                        # Stop if:
                        #   - Out of border
                        #   - Has been here before
                        #   - Out of same d sign blob

                            spread_queue.append((spread_y, spread_x))   # Any adjacent, same d sign pixel will be added to this queue to continue the search
                            has_been_here[spread_y, spread_x] = True    # Mark the current location as searched

    return counter

def get_dm( image ):
    "Converts pixels into ds"
    d_image = numpy.array([[0] * X] * Y)    # Initialize
    m_image = numpy.array([[0] * X] * Y)  # Initialize
    for y in range(Y):
        for r in range( 1, rng + 1 ):       # Computation is made in increasing range, to rng
            for x in range (r, X):
                d = difference(image[y, x], image[y, x-r]) # Only computes d for now
                m = ave - abs(d)
                d_image[y, x] += d
                d_image[y, x-r] += d
                m_image[y, x] += m
                m_image[y, x - r] += m

    return d_image, m_image

# pattern filters: eventually updated by higher-level feedback, initialized here as constants:

# Init**************************************************

mapping = numpy.vectorize(black_white)

rng = 2  # number of pixels compared to each pixel in four directions
ave = 31  # |d| value that coincides with average match: value pattern filter
ave_rate = 0.25  # average match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
ini_y = 0  # that area in test image seems to be the most diverse

# image = misc.face(gray=True)  # read image as 2d-array of pixels (gray scale):
# image = image.astype(int)

# or:
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/test_blobs.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)
Y, X = image.shape  # image height and width

# blob_image = image.astype(int)

start_time = time()

# Core**************************************************
d_image, m_image = get_dm(image)          # Computes and put individual fuzzy d of each pixel into d_image

# TextOutput********************************************
fo = open( 'images/test_blobs/Match.txt', 'w+')
for y in range(Y):
    for x in range(X):
        fo.write('%6d' % (m_image[y,x]))
    fo.write('\n')
fo.close()
# ******************************************************

d_image, m_image = mapping(d_image), mapping(m_image)    # Change elements of d_image into 0s and 255s

counted_blob =  BFS(m_image)    # Count number of blobs

# Output************************************************
end_time = time() - start_time
print(end_time)

# TextOutput********************************************
fo = open( 'images/test_blobs/blob_count.txt', 'w+')
fo.write('%d\n' %(counted_blob))
fo.close()
# ******************************************************

'''
# TextOutput********************************************
fo = open( 'pixels.txt', 'w+')
for y in range(Y):
    for x in range(X):
        fo.write('%6d' % (image[y,x]))
    fo.write('\n')
fo.close()
# ******************************************************
'''

# ImageOutput*******************************************
cv2.imwrite( './images/test_blobs/dblobs.jpg', d_image)
cv2.imwrite( './images/test_blobs/mblobs.jpg', m_image)
# ******************************************************