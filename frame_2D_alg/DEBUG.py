from scipy import misc
import numpy as np

opacity_val = 127

def draw(path, image):
    " output into image file "

    misc.imsave(path + '.bmp', image)
    # ---------- draw() end ---------------------------------------------------------------------------------------------

def map_blobs(frame, original=False):
    " Rebuilt data of blobs into an image "

    if type(frame) == list:
        blob_, (height, width) = frame[-2:]
        box = 0, height, 0, width
    else:
        blob_ = frame.Derts[-1][-1]      # sub_blob_
        box = frame.box

    frame_img = empty_map(box)

    for i, blob in enumerate(blob_):
        boxes = blob.box
        blob_map = map_blob(blob, original)

        over_draw(frame_img, blob_map, boxes, box)

    return frame_img
    # ---------- map_blobs() end ----------------------------------------------------------------------------------------

def map_blob(blob, original=False):
    " map derts into a single blob "

    blob_img = empty_map(blob.box)

    for seg in blob.seg_:


        y0s = seg[0]
        yns = y0s + seg[1][-1]
        x0s = min([P[1] for P in seg[2]])
        xns = max([P[1] + P[-2] for P in seg[2]])

        boxes = (y0s, yns, x0s, xns)

        seg_map = map_segment(seg, boxes, original)

        over_draw(blob_img, seg_map, boxes, blob.box)

    return blob_img
    # ---------- map_blob() end -----------------------------------------------------------------------------------------

def map_segment(seg, box, original=False):
    " map derts into a single segment "

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
                if P[0]:
                    seg_img[y, x] = 255
                else:
                    seg_img[y, x] = 0


    return seg_img

    # ---------- map_segment() end --------------------------------------------------------------------------------------

def over_draw(map, sub_map, boxes, box = (0, 0, 0, 0)):
    " over-write a slice of an image "
    y0, yn, x0, xn = box
    y0s, yns, x0s, xns = boxes
    y0, yn, x0, xn = y0s - y0, yns - y0, x0s - x0, xns - x0
    map[y0:yn, x0:xn][sub_map != opacity_val] = sub_map[sub_map != opacity_val]
    return map
    # ---------- over_draw() end ----------------------------------------------------------------------------------------

def empty_map(shape):
    " create a gray map with predefined shape "

    if len(shape) == 2:
        height, width = shape
    else:
        y0, yn, x0, xn = shape
        height = yn - y0
        width = xn - x0

    return np.array([[opacity_val] * width] * height)