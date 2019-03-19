import cv2

def Read_Image(path):
    " Read image from file, return list of tuples of pixels and their coordinates "

    image = cv2.imread(path, 0).astype(int)

    output = []

    for y, p_ in enumerate(image):
        output += [(y, x, p) for x, p in enumerate(p_)]

    return ouput