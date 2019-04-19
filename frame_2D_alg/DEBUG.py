import cv2
import numpy as np

def draw_blob(blob, img, localize):
    " draw a single blob "
    s = blob.sign
    y0, x0 = localize

    for seg in blob.seg_:
        for y, P in zip(range(seg[0], seg[0] + seg[1][0]), seg[2]):
            x0P, L = P[1:3]
            for x in range(x0P, x0P + L):
                img[y - y0, x - x0] = 255 if s else 0

def draw_blobs(path, frame):
    " Rebuilt data of blobs into an image "

    if type(frame) == list:
        blob_, (height, width) = frame[-2:]
        localize = 0, 0
    else:
        blob_ = frame.sub_blob_
        y0, yn, x0, xn = frame.box
        height = yn - y0
        width = xn - x0
        localize = y0, x0
        print(frame.box)

    frame_img = np.array([[127] * width] * height)

    for i, blob in enumerate(blob_):
        draw_blob(blob, frame_img, localize)

    cv2.imwrite(path + '.bmp', frame_img)
    # ---------- draw_blob() end ----------------------------------------------------------------------------------------
def map_dert___(path, dert___):

    Y = len(dert___)                                                            # height of frame
    X0 = min([dert__[0][0] for dert__ in dert___])
    Xn = max([dert__[-1][0] + len(dert__[-1][1]) for dert__ in dert___])
    X = Xn - X0                                                                 # width of frame

    image = np.array([[127] * X] * Y)

    for y, dert__ in enumerate(dert___):
        for x0, dert_ in dert__:
            for x, [(p, ncomp, dy, dx, g)] in enumerate(dert_, start= x0 - X0):
                image[y, x] = (g > 0) * 255

    cv2.imwrite(path + '.bmp', image)
    # ---------- draw_blob() end ----------------------------------------------------------------------------------------