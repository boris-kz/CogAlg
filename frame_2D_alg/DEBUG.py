import cv2
import numpy as np

def draw_blob(blob, img, globalize_coords = (0, 0)):
    " draw a single blob "
    s = blob.sign
    y0, x0 = globalize_coords
    for seg in blob.seg_:
        for y, P in zip(range(seg[0], seg[0] + seg[1][0]), seg[2]):
            x0P, L = P[1:3]
            for x in range(x0P, x0P + L):
                img[y+y0, x+x0] = 255 if s else 0

def draw_blobs(path, frame, isb=-1):
    " Rebuilt data of blobs into an image "

    height, width = frame[-1]
    frame_img = np.array([[127] * width] * height)

    for i, blob in enumerate(frame[1]):
        if isb < 0:
            draw_blob(blob, frame_img)
        elif blob.sign:
            y0, yn, x0, xn = blob.box
            for sub_blob in blob.sub_blob_[isb]:
                draw_blob(sub_blob, frame_img, (y0, x0))

    cv2.imwrite(path + '.bmp', frame_img)
    # ---------- draw_blob() end ----------------------------------------------------------------------------------------