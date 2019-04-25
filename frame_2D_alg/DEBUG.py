from scipy import misc
import numpy as np

def draw(path, image):
    " output into image file "

    misc.imsave(path + '.bmp', image)
    # ---------- draw() end ---------------------------------------------------------------------------------------------

def map_blobs(frame):
    " Rebuilt data of blobs into an image "

    if type(frame) == list:
        blob_, (height, width) = frame[-2:]
        y0, yn, x0, xn = 0, height, 0, width
    else:
        blob_ = frame.Derts[-1][-1]      # sub_blob_
        y0, yn, x0, xn = frame.box

    frame_img = empty_map((y0, yn, x0, xn))

    for i, blob in enumerate(blob_):
        y0s, yns, x0s, xns = blob.box
        blob_map = map_blob(blob)

        over_draw(frame_img, blob_map, (y0s - y0, yns - y0, x0s - x0, xns - x0))

    return frame_img
    # ---------- map_blobs() end ----------------------------------------------------------------------------------------

def map_blob(blob):
    " map derts into a single blob "

    blob_img = empty_map(blob.box)

    y0, yn, x0, xn = blob.box

    for seg in blob.seg_:


        y0s = seg[0]
        yns = y0s + seg[1][0]
        x0s = min([P[1] for P in seg[2]])
        xns = max([P[1] + P[2] for P in seg[2]])

        seg_map = map_segment(seg, (y0s, yns, x0s, xns))

        over_draw(blob_img, seg_map, (y0s - y0, yns - y0, x0s - x0, xns - x0))

    return blob_img
    # ---------- map_blob() end -----------------------------------------------------------------------------------------

def map_segment(seg, box):
    " map derts into a single segment "

    seg_img = empty_map(box)

    y0, yn, x0, xn = box

    for y, P in enumerate(seg[2], start= seg[0] - y0):
        x0P, L = P[1:3]
        x0P -= x0
        for x in range(x0P, x0P + L):
            seg_img[y, x] = 255 if P[0] else 0
        # for x, derts in enumerate(P[-1], start=x0P):
        #     seg_img[y, x] = derts[-1][-1]
    return seg_img
    # ---------- map_segment() end --------------------------------------------------------------------------------------

def over_draw(map, sub_map, box, opacity_val = 127):
    " over-write a slice of an image "
    y0, yn, x0, xn = box
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

    return np.array([[127] * width] * height)