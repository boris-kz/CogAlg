import cv2

def Read_Image(path):
    ''' Read image from file, return list of tuples of pixels and their coordinates.
        Input:  image file path
        Output: list of (y, x, p):
        - y, x: coordinate of pixel
        - p: pixel's grayscale value (0->255)
    '''

    image = cv2.imread(path, 0).astype(int) # Read image with cv2

    output_ = []                            # initialize output_

    for y, p_ in enumerate(image):          # iterate through lines
        output_ += [(y, x, p) for x, p in enumerate(p_)]

    return ouput_