from scipy import misc
import numpy as np

# ************ MAIN FUNCTIONS *******************************************************************************************
# -draw(): output numpy array of pixel as image file.
# -map_sub_blobs(): given a blob and a traversing path, map all sub blobs of a specific branch belongs to that blob
# into a numpy array.
# -map_blobs(): map all blobs in blob_ into a numpy array
# -map_blob(): map all segments in blob.seg_ into a numpy array
# -map_segment(): map all Ps of a segment into a numpy array
# -over_draw(): used to draw sub-structure's map onto to current level structure
# -empty_map(): create a numpy array representing blobs' map
# ***********************************************************************************************************************

transparent_val = 127   # a pixel at this value is considered transparent

def draw(path, image, extension='.bmp'):
    ''' Output into an image file.
        Arguments:
        - path: path for saving image file.
        - image: input as numpy array.
        - extension: determine file-type of ouput image.
        Return: None '''

    misc.imsave(path + extension, image)
    # ---------- draw() end ---------------------------------------------------------------------------------------------

def map_sub_blobs(blob, traverse_path):
    ''' Given a blob and a traversing path, map image of all sub-blobs of a specific branch
        belonging to that blob into a numpy array.
        Arguments:
        - blob: contain all mapped sub-blobs.
        - traverse_path: list of values determine the derivation sequence of target sub-blobs.
        Return: numpy array of image's pixel '''

    image = empty_map(blob.box)

    return image
    # ---------- map_sub_blobs() end ------------------------------------------------------------------------------------

def map_frame(frame):
    ''' Map the whole frame of original image as computed blobs.
        Argument:
        - frame: frame object input (as a list).
        Return: numpy array of image's pixel '''

    blob_, (height, width) = frame[-2:]
    box = (0, height, 0, width)
    image = empty_map(box)

    for i, blob in enumerate(blob_):
        blob_map = map_blob(blob, original)

        over_draw(image, blob_map, blob.box, box)

    return image
    # ---------- map_frame() end ----------------------------------------------------------------------------------------

def map_blob(blob, original=False):
    ''' Map a single blob into an image.
        Argument:
        - blob: the input blob.
        - original: each pixel is the original image's pixel instead of just black or white to separate blobs.
        Return: numpy array of image's pixel '''

    blob_img = empty_map(blob.box)

    for seg in blob.seg_:


        y0s = seg[0]
        yns = y0s + seg[1][-1]
        x0s = min([P[1] for P in seg[2]])
        xns = max([P[1] + P[-2] for P in seg[2]])

        sub_box = (y0s, yns, x0s, xns)

        seg_map = map_segment(seg, sub_box, original)

        over_draw(blob_img, seg_map, sub_box, blob.box)

    return blob_img
    # ---------- map_blob() end -----------------------------------------------------------------------------------------

def map_segment(seg, box, original=False):
    ''' Map a single segment of a blob into an image.
        Argument:
        - seg: the input segment.
        - box: the input segment's bounding box.
        - original: each pixel is the original image's pixel instead of just black or white to separate blobs.
        Return: numpy array of image's pixel '''

    seg_img = empty_map(box)

    y0, yn, x0, xn = box

    for y, P in enumerate(seg[2], start= seg[0] - y0):
        x0P= P[1]
        x0P -= x0
        derts_ = P[-1]
        for x, derts in enumerate(derts_, start=x0P):
            if original:
                seg_img[y, x] = derts[0][0]
            else:
                seg_img[y, x] = 255 if P[0] else 0

    return seg_img

    # ---------- map_segment() end --------------------------------------------------------------------------------------

def over_draw(map, sub_map, sub_box, box = (0, 0, 0, 0)):
    ''' Over-write map of sub-structure onto map of parent-structure.
        Argument:
        - map: map of parent-structure.
        - sub_map: map of sub-structure.
        - sub_box: bounding box of sub-structure.
        - box: bounding box of parent-structure, for computing local coordinate of sub-structure.
        Return: over-written map of parent-structure '''

    y0, yn, x0, xn = box
    y0s, yns, x0s, xns = sub_box
    y0, yn, x0, xn = y0s - y0, yns - y0, x0s - x0, xns - x0
    map[y0:yn, x0:xn][sub_map != transparent_val] = sub_map[sub_map != transparent_val]
    return map
    # ---------- over_draw() end ----------------------------------------------------------------------------------------

def empty_map(shape):
    ''' Create an empty numpy array of desired shape.
        Argument:
        - shape: desired shape of the output.
        Return: over-written map of parent-structure '''

    if len(shape) == 2:
        height, width = shape
    else:
        y0, yn, x0, xn = shape
        height = yn - y0
        width = xn - x0

    return np.array([[transparent_val] * width] * height)
