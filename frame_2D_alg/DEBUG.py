import cv2
import numpy as np

def draw_blob(blob, img, globalize_coords = (0, 0)):
    " draw a single blob "
    s = blob.sign
    y0, x0 = globalize_coords
    for seg in blob.e_:
        for P in seg[2]:
            y = P[1][1] // P[1][0]
            for dert in P[2]:
                x = dert[:2]
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